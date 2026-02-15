from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import partial
# from tiny_api_cuda import update_flatten_view
from trimkv.triton import update_flatten_view_triton

import torch

from transformers.cache_utils import Cache, DynamicCache


class TrimKVCache(DynamicCache):
    def __init__(
        self,
        max_seq_len: int = 20480,
        _distributed_cache_data: Iterable = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.retention_weights: List[torch.Tensor] = []
        self.kv_positions: List[torch.Tensor] = []
        self.n_seen_tokens: List[int] = []
        self.device = device
        self.offset = torch.tensor(0, dtype=torch.int64)
        self.block_mask = None
        self.sliding_window_size = 0

        if _distributed_cache_data is not None:
            raise NotImplementedError("Distributed cache data is not supported for TrimKVCache")

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        if layer_idx is None:
            return self._seen_tokens
        if layer_idx < len(self.n_seen_tokens):
            return self.n_seen_tokens[layer_idx]
        else:
            return 0

    def get_total_cached_tokens(self, num_key_value_heads: Optional[int] = None) -> int:
        if num_key_value_heads is None:
            num_key_value_heads = self.key_cache[0].shape[1] if self.key_cache else 0

        return sum(self.get_cache_length(layer_idx) * num_key_value_heads for layer_idx in range(len(self)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        retention_weights: torch.Tensor,
        cache_positions: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if key_states is not None:
            bsz, num_heads, seq_len, dim = key_states.shape
            # assert bsz == 1, "Current implementation does not support attention mask due to a conflict in transformers's version, thus we can only run with batch_size 1. Need to be fixed later."
            # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += seq_len

            cache_positions = cache_positions[None, None, :].expand_as(retention_weights) if cache_positions.dim() == 1 else cache_positions

            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    self.retention_weights.append(torch.tensor([]))
                    self.kv_positions.append(torch.tensor([]))
                    self.n_seen_tokens.append(0)

                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.retention_weights.append(retention_weights)
                self.kv_positions.append(cache_positions)
                self.n_seen_tokens.append(key_states.shape[-2])
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                self.retention_weights[layer_idx] = retention_weights
                self.kv_positions[layer_idx] = cache_positions
                self.n_seen_tokens[layer_idx] = key_states.shape[-2]
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.retention_weights[layer_idx] = torch.cat([self.retention_weights[layer_idx], retention_weights], dim=-1)
                self.kv_positions[layer_idx] = torch.cat([self.kv_positions[layer_idx], cache_positions], dim=-1)
                self.n_seen_tokens[layer_idx] += key_states.shape[-2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx], {}

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
            self.retention_weights[layer_idx] = self.retention_weights[layer_idx][indices, ...]
            self.kv_positions[layer_idx] = self.kv_positions[layer_idx][indices, ...]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = TrimKVCache(max_seq_len=self.max_seq_len, device=self.device)
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.retention_weights = [tensor[i : i + split_size] for tensor in self.retention_weights]
            current_split.kv_positions = [tensor[i : i + split_size] for tensor in self.kv_positions]
            out.append(current_split)
        return out

    def get_cache_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def compress(
        self,
        strategy: str = "alpha",
        memory_size: int = 2048,
        buffer_size: int = 512,
        floor_budget_ratio: float = 0.,
        alpha_threshold: float = 0.0,
        num_layers: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        skip_layers: int = 0,
    ):
        num_layers = len(self.key_cache) if num_layers is None else num_layers
        assert num_layers == len(self.value_cache) == len(self.retention_weights) == len(self.kv_positions), "All caches must have the same number of layers"
        num_key_value_heads = self.key_cache[0].shape[1] if num_key_value_heads is None else num_key_value_heads
        assert num_key_value_heads == self.value_cache[0].shape[1], "Key and value caches must have the same number of heads"
        # layer wise compression
        for layer_idx in range(skip_layers, num_layers):
            if memory_size + buffer_size <= self.get_cache_length(layer_idx):
                self.compress_layer(layer_idx, memory_size, strategy, self.sliding_window_size)

    def compress_layer(
        self,
        layer_idx: int,
        memory_size: int,
        comrpess_strategy: str = "alpha",
        sliding_window_size: int = 0,
    ):
        key_states = self.key_cache[layer_idx]
        value_states = self.value_cache[layer_idx]
        kv_positions = self.kv_positions[layer_idx]
        retention_weights = self.retention_weights[layer_idx]
        
        log_beta = retention_weights.to(torch.float32)
        q_idx = self.get_seq_length(layer_idx) + 1
        log_alpha = (log_beta * (q_idx - kv_positions))

        if comrpess_strategy == "knorm_alpha":
            # get norm of the key states
            key_norm = key_states.norm(dim=-1, keepdim=False)
            # scores = torch.exp(log_alpha) * key_norm
            scores = log_alpha + torch.log(key_norm)
        elif comrpess_strategy == "alpha":
            scores = log_alpha
        else:
            raise ValueError(f"Unknown compression strategy: {comrpess_strategy}")

        if sliding_window_size > 0:
            scores[:, :, -sliding_window_size:] = float('inf')
        # get top-k (memory size) indices with highest alpha values to keep
        # top_k_indices = torch.topk(log_alpha, memory_size, dim=-1).indices
        top_k_indices = torch.topk(scores, memory_size, dim=-1).indices
        # sort the top-k indices to maintain order
        top_k_indices, _ = torch.sort(top_k_indices, dim=-1)

        # gather the top-k key and value states to the first position, using gather
        self.key_cache[layer_idx] = key_states.gather(-2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1]))
        self.value_cache[layer_idx] = value_states.gather(-2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, value_states.shape[-1]))
        self.retention_weights[layer_idx] = self.retention_weights[layer_idx].gather(-1, top_k_indices)
        self.kv_positions[layer_idx] = kv_positions.gather(-1, top_k_indices)
        
    def log(self, layer_idx: int = None):
        logs = {}
        if layer_idx is None:
            for layer_idx in range(len(self.key_cache)):
                logs[layer_idx] = {
                    "seen_tokens": self.get_seq_length(layer_idx),
                    "kv_positions": self.kv_positions[layer_idx].detach().cpu(),
                }
        else:
            logs["seen_tokens"] = self.get_seq_length(layer_idx)
            logs["kv_positions"] = self.kv_positions[layer_idx].detach().cpu()
        return logs

    def copy_to_device(self, device: str):
        new_cache = TrimKVCache(max_seq_len=self.max_seq_len, device=device)
        new_cache._seen_tokens = self._seen_tokens
        new_cache.offset = self.offset.to(device)
        new_cache.block_mask = self.block_mask.to(device) if self.block_mask is not None else None
        new_cache.key_cache = [tensor.to(device) for tensor in self.key_cache]
        new_cache.value_cache = [tensor.to(device) for tensor in self.value_cache]
        new_cache.retention_weights = [tensor.to(device) for tensor in self.retention_weights]
        new_cache.kv_positions = [tensor.to(device) for tensor in self.kv_positions]
        new_cache.n_seen_tokens = self.n_seen_tokens.copy()
        return new_cache


class DynamicBudgetTrimKVCache(TrimKVCache):
    def __init__(
        self,
        max_seq_len: int = 20480,
        _distributed_cache_data: Iterable = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.retention_weights: List[torch.Tensor] = []
        self.kv_positions: List[torch.Tensor] = []

        # Running parameters for update_flatten_view and flash_attn_varlen_func
        self.head_lens: List[torch.Tensor] = []
        self.cu_seqlens_k: List[torch.Tensor] = []

        self.device = device
        self.offset = torch.tensor(0, dtype=torch.int64)
        self.block_mask = None

        if _distributed_cache_data is not None:
            raise NotImplementedError("Distributed cache data is not supported for DynamicBudgetTrimKVCache")

    def get_total_cached_tokens(self) -> int:
        return sum(self.key_cache[layer_idx].shape[0] for layer_idx in range(len(self)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        retention_weights: torch.Tensor,
        cache_positions: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if key_states is not None:
            # Update the number of seen tokens
            bz, num_heads, seq_len, dim = key_states.shape
            assert bz == 1, "Batch size greater than 1 is not supported for DynamicBudgetTrimKVCache"

            if layer_idx == 0:
                self._seen_tokens += seq_len

            cache_positions = cache_positions[None, None, :].expand_as(retention_weights) if cache_positions.dim() == 1 else cache_positions

            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    self.retention_weights.append(torch.tensor([]))
                    self.kv_positions.append(torch.tensor([]))

                key_states = key_states.contiguous().view(-1, dim) # (B*H*S,D)
                value_states = value_states.contiguous().view(-1, dim) # (B*H*S,D)

                # use (-1, 1) just to be compatible with update_flatten_view
                cache_positions = cache_positions.contiguous().view(-1, 1) # (B*H*S, 1)
                retention_weights = retention_weights.contiguous().view(-1, 1) # (B*H*S, 1)

                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.retention_weights.append(retention_weights)
                self.kv_positions.append(cache_positions)
                # Update head_lens and cu_seqlens_k
                self.head_lens.append(torch.tensor([seq_len] * num_heads, device=self.device, dtype=torch.int32))
                self.cu_seqlens_k.append(torch.arange(0, (seq_len * num_heads) + 1, step=seq_len, device=self.device, dtype=torch.int32))
            else:
                self.key_cache[layer_idx] = update_flatten_view_triton(
                    self.key_cache[layer_idx], key_states.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )
                self.value_cache[layer_idx] = update_flatten_view_triton(
                    self.value_cache[layer_idx], value_states.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )

                retention_weights = retention_weights.unsqueeze(-1)
                cache_positions = cache_positions.unsqueeze(-1)
                self.retention_weights[layer_idx] = update_flatten_view_triton(
                    self.retention_weights[layer_idx], retention_weights.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )
                self.kv_positions[layer_idx] = update_flatten_view_triton(
                    self.kv_positions[layer_idx], cache_positions.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )

                # Update head_lens and cu_seqlens_k
                self.head_lens[layer_idx] = self.head_lens[layer_idx] + seq_len
                cu_offset = torch.arange(0, (num_heads * seq_len) + 1, step=seq_len, device=self.device, dtype=torch.int32)
                self.cu_seqlens_k[layer_idx] = self.cu_seqlens_k[layer_idx] + cu_offset

        flash_attn_kwargs = {
            "head_lens": self.head_lens[layer_idx],
            "cu_seqlens_k": self.cu_seqlens_k[layer_idx],
        }

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx], flash_attn_kwargs

    def get_cache_length(self, layer_idx: int = 0, head_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.head_lens[layer_idx][head_idx].item()

    @torch.inference_mode()
    def compress(
        self,
        strategy: str = "alpha",
        memory_size: int = 2048,
        buffer_size: int = 512,
        floor_budget_ratio: float = 0.,
        alpha_threshold: float = 0.0,
        num_layers: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        skip_layers: int = 0,  # unused but kept for API compatibility
    ):
        # print(f"Compressing cache with strategy: {strategy}, memory_size: {memory_size}, buffer_size: {buffer_size}, floor_budget_ratio: {floor_budget_ratio}, alpha_threshold: {alpha_threshold}, num_layers: {num_layers}, num_key_value_heads: {num_key_value_heads}")
        if "layer_wise" not in strategy:
            self._compress_all_layers(
                strategy=strategy,
                memory_size=memory_size,
                buffer_size=buffer_size,
                floor_budget_ratio=floor_budget_ratio,
                alpha_threshold=alpha_threshold,
                num_layers=num_layers,
                num_key_value_heads=num_key_value_heads,
            )
        else:
            num_layers = len(self.key_cache) if num_layers is None else num_layers
            num_key_value_heads = self.head_lens[0].shape[0] if num_key_value_heads is None else num_key_value_heads
            for l in range(num_layers):
                if num_key_value_heads * (memory_size + buffer_size) >= self.key_cache[l].shape[0]:
                    continue

                self._compress_layer(
                    layer_idx=l,
                    memory_size=memory_size,
                    floor_budget_ratio=floor_budget_ratio,
                    strategy=strategy.replace("layer_wise_", ""),
                    num_key_value_heads=num_key_value_heads,
                )

    def _compress_layer(
        self,
        layer_idx: int,
        memory_size: int,
        floor_budget_ratio: float = 0.,
        strategy: str = "alpha",
        num_key_value_heads: int = None,
    ):
        device = self.device
        num_key_value_heads = self.head_lens[layer_idx].shape[0] if num_key_value_heads is None else num_key_value_heads

        rw = self.retention_weights[layer_idx].squeeze(-1)  # (T,)
        kv_pos = self.kv_positions[layer_idx].squeeze(-1)  # (T,)

        q_idx = self.get_seq_length() + 1
        log_alpha = (q_idx - kv_pos) * rw

        if strategy == "knorm_alpha":
            key_norm = self.key_cache[layer_idx].norm(dim=-1).reshape(-1)           # (T,)
            scores = log_alpha + torch.log(key_norm.clamp_min(1e-12))
        elif strategy == "alpha":
            scores = log_alpha
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        layer_memory_size = num_key_value_heads * memory_size
        topk_idx = torch.topk(scores, layer_memory_size, largest=True, sorted=False).indices  # (K,)
        topk_mask = torch.zeros_like(scores, dtype=torch.bool)
        topk_mask.index_fill_(0, topk_idx, True)

        self.key_cache[layer_idx] = self.key_cache[layer_idx][topk_mask, ...]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][topk_mask, ...]
        self.retention_weights[layer_idx] = self.retention_weights[layer_idx][topk_mask, ...]
        self.kv_positions[layer_idx] = self.kv_positions[layer_idx][topk_mask, ...]

        self.head_lens[layer_idx] = torch.tensor(
            [topk_mask[self.cu_seqlens_k[layer_idx][h]:self.cu_seqlens_k[layer_idx][h+1]].sum() for h in range(num_key_value_heads)],
            device=device,
            dtype=torch.int32,
        )

        self.cu_seqlens_k[layer_idx] = torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=torch.int32), self.head_lens[layer_idx]], dim=0),
            dim=0,
            dtype=torch.int32,
        )

    def _compress_all_layers(
        self,
        strategy: str = "alpha",
        memory_size: int = 2048,
        buffer_size: int = 512,
        floor_budget_ratio: float = 0.,
        alpha_threshold: float = 0.0,
        num_layers: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
    ):
        device = self.device
        num_layers = len(self.key_cache) if num_layers is None else num_layers
        num_key_value_heads = self.head_lens[0].shape[0] if num_key_value_heads is None else num_key_value_heads
        total_memory_size = num_layers * num_key_value_heads * memory_size
        if num_layers * num_key_value_heads * (memory_size + buffer_size) > self.get_total_cached_tokens() and "threshold" not in strategy:
            return

        rw = torch.cat([self.retention_weights[l] for l in range(num_layers)], dim=0).squeeze(-1)  # (T,)
        kv_pos = torch.cat([self.kv_positions[l] for l in range(num_layers)], dim=0).squeeze(-1)  # (T,)
        layer_lens = torch.tensor([self.retention_weights[l].shape[0] for l in range(num_layers)])
        cu_layer_lens = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), layer_lens], dim=0), dim=0)

        q_idx = self.get_seq_length() + 1
        log_alpha = (q_idx - kv_pos) * rw

        if strategy in ["knorm_alpha", "knorm_alpha_threshold"]:
            # Concatenate keys along sequence dim once, then take norms (also 1D)
            key_chunks = []
            for l in range(num_layers):
                key_chunks.append(self.key_cache[l])        # (1,S,D)
            all_keys = torch.cat(key_chunks, dim=-2)               # (1,T,D)
            key_norm = all_keys.norm(dim=-1).reshape(-1)           # (T,)
            scores = log_alpha + torch.log(key_norm.clamp_min(1e-12))
        elif strategy in ["alpha", "alpha_threshold"]:
            scores = log_alpha
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        if floor_budget_ratio == 0:
            if "threshold" not in strategy:
                topk_idx = torch.topk(scores, total_memory_size, largest=True, sorted=False).indices  # (K,)
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.index_fill_(0, topk_idx, True)
            else:
                topk_mask = scores >= torch.log(torch.tensor(alpha_threshold, device=device))

            # compute budget for each head and layer
            for l in range(num_layers):
                start = cu_layer_lens[l].item()
                end = cu_layer_lens[l + 1].item()
                layer_mask = topk_mask[start:end]  # (S_l,)
                self.key_cache[l] = self.key_cache[l][layer_mask, ...]
                self.value_cache[l] = self.value_cache[l][layer_mask, ...]
                self.retention_weights[l] = self.retention_weights[l][layer_mask, ...]
                self.kv_positions[l] = self.kv_positions[l][layer_mask, ...]

                self.head_lens[l] = torch.tensor(
                    [layer_mask[self.cu_seqlens_k[l][h]:self.cu_seqlens_k[l][h+1]].sum() for h in range(num_key_value_heads)],
                    device=device,
                    dtype=torch.int32,
                )

                self.cu_seqlens_k[l] = torch.cumsum(
                    torch.cat([torch.zeros(1, device=device, dtype=torch.int32), self.head_lens[l]], dim=0),
                    dim=0,
                    dtype=torch.int32,
                )
        else:
            head_lens = torch.cat([self.head_lens[l] for l in range(num_layers)], dim=0) # (L*H,)
            cu_head_lens = torch.cumsum(head_lens, dim=0)

            assert "threshold" not in strategy, "Thresholding strategy is not compatible with floor_budget_ratio > 0"

            adaptive_memory_size = int(total_memory_size * (1 - floor_budget_ratio))
            topk_idx = torch.topk(scores, adaptive_memory_size, largest=True, sorted=False).indices  # (K,)

            # use bucketize to compute the number of keys selected in each head
            cu_head_lens = torch.cumsum(head_lens, dim=0)
            lh_idx = torch.bucketize(topk_idx, cu_head_lens)
            head_cnt = torch.zeros_like(head_lens, dtype=torch.long)
            head_cnt.index_add_(0, lh_idx, torch.ones_like(topk_idx))
            head_cnt = head_cnt + int(memory_size * floor_budget_ratio)
            # reshape head_cnt to (L, H)
            head_cnt = head_cnt.view(num_layers, num_key_value_heads)
            # compute topk mask for each layer, head
            for l in range(num_layers):
                start = cu_layer_lens[l].item()
                end = cu_layer_lens[l + 1].item()
                layer_scores = scores[start:end]
                layer_topk_mask = []
                for h in range(num_key_value_heads):
                    head_start = self.cu_seqlens_k[l][h].item()
                    head_end = self.cu_seqlens_k[l][h + 1].item()
                    head_scores = layer_scores[head_start:head_end]
                    k = head_cnt[l, h].item()

                    if k >= head_scores.shape[0]:
                        layer_topk_mask.append(torch.ones_like(head_scores, dtype=torch.bool))
                        continue

                    head_topk_idx = torch.topk(head_scores, k, largest=True, sorted=False).indices
                    head_topk_mask = torch.zeros_like(head_scores, dtype=torch.bool)
                    head_topk_mask.index_fill_(0, head_topk_idx, True)
                    layer_topk_mask.append(head_topk_mask)
                layer_topk_mask = torch.cat(layer_topk_mask, dim=0)
                self.key_cache[l] = self.key_cache[l][layer_topk_mask, ...]
                self.value_cache[l] = self.value_cache[l][layer_topk_mask, ...]
                self.retention_weights[l] = self.retention_weights[l][layer_topk_mask, ...]
                self.kv_positions[l] = self.kv_positions[l][layer_topk_mask, ...]
                self.head_lens[l] = torch.tensor(
                    [layer_topk_mask[self.cu_seqlens_k[l][h]:self.cu_seqlens_k[l][h+1]].sum() for h in range(num_key_value_heads)],
                    device=device,
                    dtype=torch.int32,
                )
                self.cu_seqlens_k[l] = torch.cumsum(
                    torch.cat([torch.zeros(1, device=device, dtype=torch.int32), self.head_lens[l]], dim=0),
                    dim=0,
                    dtype=torch.int32,
                )
                


    def log(self):
        num_layers = len(self.head_lens)
        num_key_value_heads = self.head_lens[0].shape[0] if num_key_value_heads is None else num_key_value_heads
        
        head_wise_kv_positions = []
        for l in range(num_layers):
            for h in range(num_key_value_heads):
                head_start = self.cu_seqlens_k[l][h].item()
                head_end = self.cu_seqlens_k[l][h + 1].item()
                head_kv_positions = self.kv_positions[l][head_start:head_end].detach().cpu()
                head_wise_kv_positions.append(head_kv_positions)
            
        logs = {
            "head_wise_kv_positions": head_wise_kv_positions, # list of token positions cached for each head
            "flat_head_lens": torch.cat(self.head_lens).to('cpu').numpy(), # flattened head lens for all layers * heads
        }
        return logs


    def copy_to_device(self, device):
        raise NotImplementedError("copy_to_device not implemented for DynamicBudgetTrimKVCache")


class BatchedDynamicBudgetTrimKVCache(TrimKVCache):

    def __init__(
        self,
        max_seq_len: int = 20480,
        _distributed_cache_data: Iterable = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.retention_weights: List[torch.Tensor] = []
        self.kv_positions: List[torch.Tensor] = []

        self.head_lens: List[torch.Tensor] = []
        self.cu_seqlens_k: List[torch.Tensor] = []

        self.batch_size: Optional[int] = None
        self.num_kv_heads: Optional[int] = None
        self._per_batch_seen_tokens: Optional[torch.Tensor] = None  # (B,) - yejun

        self.device = device
        self.offset = torch.tensor(0, dtype=torch.int64)
        self.block_mask = None

        if _distributed_cache_data is not None:
            raise NotImplementedError("Distributed cache data is not supported for BatchedDynamicBudgetTrimKVCache")

    def get_total_cached_tokens(self) -> int:
        return sum(self.key_cache[layer_idx].shape[0] for layer_idx in range(len(self)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        retention_weights: torch.Tensor,
        cache_positions: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states is not None:
            B, H, S, D = key_states.shape

            if self.batch_size is None:
                self.batch_size = B
                self.num_kv_heads = H
            else:
                assert B == self.batch_size, f"Batch size changed: {self.batch_size} -> {B}"
                assert H == self.num_kv_heads, f"Num KV heads changed: {self.num_kv_heads} -> {H}"

            if layer_idx == 0:
                self._seen_tokens += S
                if self._per_batch_seen_tokens is not None:
                    self._per_batch_seen_tokens += S

            if cache_positions.dim() == 1:
                cache_positions = cache_positions[None, None, :].expand(B, H, S)

            num_seqs = B * H

            if len(self.key_cache) <= layer_idx:
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    self.retention_weights.append(torch.tensor([]))
                    self.kv_positions.append(torch.tensor([]))
                    self.head_lens.append(torch.tensor([]))
                    self.cu_seqlens_k.append(torch.tensor([]))

                # filter out left-padding tokens on prefill - yejun
                position_ids = self._position_ids  # (B, S)
                real_mask = self._prefill_padding_mask.bool()  # (B, S)
                cache_positions = position_ids.unsqueeze(1).expand(B, H, S)  # (B, H, S)

                real_lens = real_mask.sum(dim=1).int()  # (B,)
                per_seq_lens = real_lens.unsqueeze(1).expand(B, H).reshape(-1)  # (B*H,)

                key_chunks = []
                val_chunks = []
                rw_chunks = []
                pos_chunks = []
                for b in range(B):
                    m = real_mask[b]  # (S,)
                    key_chunks.append(key_states[b, :, m, :].reshape(-1, D))
                    val_chunks.append(value_states[b, :, m, :].reshape(-1, D))
                    rw_chunks.append(retention_weights[b, :, m].reshape(-1, 1))
                    pos_chunks.append(cache_positions[b, :, m].reshape(-1, 1))

                key_flat = torch.cat(key_chunks, dim=0)
                value_flat = torch.cat(val_chunks, dim=0)
                retention_weights_flat = torch.cat(rw_chunks, dim=0)
                cache_positions_flat = torch.cat(pos_chunks, dim=0)

                self.head_lens.append(per_seq_lens.to(device=self.device, dtype=torch.int32))
                self.cu_seqlens_k.append(torch.cumsum(
                    torch.cat([torch.zeros(1, device=self.device, dtype=torch.int32), per_seq_lens.to(device=self.device, dtype=torch.int32)]),
                    dim=0, dtype=torch.int32,
                ))

                # track per-batch seen tokens for correct compression q_idx - yejun
                if layer_idx == 0:
                    self._per_batch_seen_tokens = real_lens.to(device=self.device, dtype=torch.long)
                    self._seen_tokens = self._seen_tokens - S + real_lens.max().item()

                self.key_cache.append(key_flat)
                self.value_cache.append(value_flat)
                self.retention_weights.append(retention_weights_flat)
                self.kv_positions.append(cache_positions_flat)

            else:
                key_states_reshaped = key_states.reshape(1, num_seqs, S, D)
                value_states_reshaped = value_states.reshape(1, num_seqs, S, D)

                self.key_cache[layer_idx] = update_flatten_view_triton(
                    self.key_cache[layer_idx],
                    key_states_reshaped.contiguous(),
                    self.head_lens[layer_idx],
                    self.cu_seqlens_k[layer_idx]
                )
                self.value_cache[layer_idx] = update_flatten_view_triton(
                    self.value_cache[layer_idx],
                    value_states_reshaped.contiguous(),
                    self.head_lens[layer_idx],
                    self.cu_seqlens_k[layer_idx]
                )

                retention_weights_reshaped = retention_weights.reshape(1, num_seqs, S, 1)
                # use per-batch position_ids for correct positions with unequal prompts - yejun
                cache_positions = self._position_ids.unsqueeze(1).expand(B, H, S)  # (B, H, S)
                cache_positions_reshaped = cache_positions.reshape(1, num_seqs, S, 1)

                self.retention_weights[layer_idx] = update_flatten_view_triton(
                    self.retention_weights[layer_idx],
                    retention_weights_reshaped.contiguous(),
                    self.head_lens[layer_idx],
                    self.cu_seqlens_k[layer_idx]
                )
                self.kv_positions[layer_idx] = update_flatten_view_triton(
                    self.kv_positions[layer_idx],
                    cache_positions_reshaped.contiguous(),
                    self.head_lens[layer_idx],
                    self.cu_seqlens_k[layer_idx]
                )

                self.head_lens[layer_idx] = self.head_lens[layer_idx] + S
                cu_offset = torch.arange(0, num_seqs * S + 1, S, device=self.device, dtype=torch.int32)
                self.cu_seqlens_k[layer_idx] = self.cu_seqlens_k[layer_idx] + cu_offset

        flash_attn_kwargs = {
            "head_lens": self.head_lens[layer_idx],
            "cu_seqlens_k": self.cu_seqlens_k[layer_idx],
            "batch_size": self.batch_size,
            "num_kv_heads": self.num_kv_heads,
        }

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx], flash_attn_kwargs

    def get_cache_length(self, layer_idx: int = 0, batch_idx: int = 0, head_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        seq_idx = batch_idx * self.num_kv_heads + head_idx
        return self.head_lens[layer_idx][seq_idx].item()

    @torch.inference_mode()
    def compress(
        self,
        strategy: str = "alpha",
        memory_size: int = 2048,
        buffer_size: int = 512,
        floor_budget_ratio: float = 0.,
        alpha_threshold: float = 0.0,
        num_layers: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        skip_layers: int = 0,
    ):
        num_layers = len(self.key_cache) if num_layers is None else num_layers
        num_key_value_heads = self.num_kv_heads if num_key_value_heads is None else num_key_value_heads
        B = self.batch_size
        H = num_key_value_heads

        tokens_per_batch = H * (memory_size + buffer_size)
        total_tokens = self.get_total_cached_tokens()
        if total_tokens <= B * tokens_per_batch * num_layers and "threshold" not in strategy:
            return

        if "layer_wise" in strategy:
            for l in range(num_layers):
                layer_tokens = self.key_cache[l].shape[0]
                if layer_tokens <= B * H * (memory_size + buffer_size):
                    continue
                self._compress_layer_batched(
                    layer_idx=l,
                    memory_size=memory_size,
                    floor_budget_ratio=floor_budget_ratio,
                    strategy=strategy.replace("layer_wise_", ""),
                )
        else:
            self._compress_all_layers_batched(
                strategy=strategy,
                memory_size=memory_size,
                buffer_size=buffer_size,
                floor_budget_ratio=floor_budget_ratio,
                alpha_threshold=alpha_threshold,
                num_layers=num_layers,
            )

    def _compress_layer_batched(
        self,
        layer_idx: int,
        memory_size: int,
        floor_budget_ratio: float = 0.,
        strategy: str = "alpha",
    ):
        device = self.device
        B = self.batch_size
        H = self.num_kv_heads
        num_seqs = B * H

        rw = self.retention_weights[layer_idx].squeeze(-1)   # (T,)
        kv_pos = self.kv_positions[layer_idx].squeeze(-1)     # (T,)
        T = rw.shape[0]

        # broadcast per-batch q_idx to every token without looping - yejun
        cu = self.cu_seqlens_k[layer_idx]                     # (B*H+1,)
        # map each token to its seq index, then to its batch index
        seq_idx = torch.bucketize(torch.arange(T, device=device), cu[1:], right=True)  # (T,)
        batch_idx = seq_idx // H                              # (T,)
        q_idx = self._per_batch_seen_tokens[batch_idx] + 1    # (T,)

        log_alpha = (q_idx - kv_pos) * rw

        if strategy == "knorm_alpha":
            key_norm = self.key_cache[layer_idx].norm(dim=-1).reshape(-1)
            scores = log_alpha + torch.log(key_norm.clamp_min(1e-12))
        elif strategy == "alpha":
            scores = log_alpha
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        # pad scores into (B, max_tokens_per_batch) for batched topk - yejun
        batch_starts = cu[::H]                                # (B+1,)
        batch_lens = batch_starts[1:] - batch_starts[:-1]     # (B,)
        max_batch_len = batch_lens.max().item()
        k = memory_size * H

        # scatter flat scores into padded (B, max_batch_len) tensor - yejun
        within_batch_pos = torch.arange(T, device=device, dtype=torch.long) - batch_starts[batch_idx]
        padded_scores = torch.full((B, max_batch_len), float('-inf'), device=device, dtype=scores.dtype)
        padded_scores[batch_idx, within_batch_pos] = scores

        topk_indices = torch.topk(padded_scores, min(k, max_batch_len), dim=-1, largest=True, sorted=False).indices  # (B, k)

        # convert padded topk indices back to a flat mask - yejun
        # filter out indices that point to -inf padding columns
        valid_mask = topk_indices < batch_lens.unsqueeze(1)   # (B, k)
        flat_idx = batch_starts[:-1].unsqueeze(1) + topk_indices  # (B, k)
        topk_mask = torch.zeros(T, dtype=torch.bool, device=device)
        topk_mask[flat_idx[valid_mask]] = True

        # apply mask to all cache tensors at once - yejun
        self.key_cache[layer_idx] = self.key_cache[layer_idx][topk_mask]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][topk_mask]
        self.retention_weights[layer_idx] = self.retention_weights[layer_idx][topk_mask]
        self.kv_positions[layer_idx] = self.kv_positions[layer_idx][topk_mask]

        # recompute head_lens from mask using segment reduction - yejun
        per_seq_mask = torch.zeros(num_seqs, device=device, dtype=torch.int32)
        per_seq_mask.index_add_(0, seq_idx[topk_mask], torch.ones(topk_mask.sum(), device=device, dtype=torch.int32))
        self.head_lens[layer_idx] = per_seq_mask
        self.cu_seqlens_k[layer_idx] = torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=torch.int32), per_seq_mask]),
            dim=0, dtype=torch.int32,
        )

    def _compress_all_layers_batched(
        self,
        strategy: str = "alpha",
        memory_size: int = 2048,
        buffer_size: int = 512,
        floor_budget_ratio: float = 0.,
        alpha_threshold: float = 0.0,
        num_layers: Optional[int] = None,
    ):
        device = self.device
        num_layers = len(self.key_cache) if num_layers is None else num_layers
        B = self.batch_size
        H = self.num_kv_heads
        num_seqs = B * H

        total_budget_per_batch = num_layers * H * memory_size

        tokens_per_batch_current = sum(
            self.key_cache[l].shape[0] for l in range(num_layers)
        ) // B
        if tokens_per_batch_current <= num_layers * H * (memory_size + buffer_size) and "threshold" not in strategy:
            return

        q_idx_per_batch = self._per_batch_seen_tokens + 1  # (B,) - yejun

        # compute scores for all layers, broadcast q_idx per token - yejun
        all_scores = []       # flat scores per layer
        all_seq_idx = []      # seq index per token per layer (for head_lens later)
        all_batch_idx = []    # batch index per token per layer
        per_layer_batch_lens = []  # (num_layers, B) token count per batch per layer

        for l in range(num_layers):
            cu = self.cu_seqlens_k[l]
            T_l = self.key_cache[l].shape[0]
            seq_idx = torch.bucketize(torch.arange(T_l, device=device), cu[1:], right=True)
            batch_idx = seq_idx // H
            q_idx = q_idx_per_batch[batch_idx]

            rw = self.retention_weights[l].squeeze(-1)
            kv_pos = self.kv_positions[l].squeeze(-1)
            log_alpha = (q_idx - kv_pos) * rw

            if strategy in ["knorm_alpha", "knorm_alpha_threshold"]:
                key_norm = self.key_cache[l].norm(dim=-1).reshape(-1)
                scores = log_alpha + torch.log(key_norm.clamp_min(1e-12))
            elif strategy in ["alpha", "alpha_threshold"]:
                scores = log_alpha
            else:
                raise ValueError(f"Unknown compression strategy: {strategy}")

            all_scores.append(scores)
            all_seq_idx.append(seq_idx)
            all_batch_idx.append(batch_idx)

            batch_lens = torch.zeros(B, device=device, dtype=torch.long)
            batch_lens.index_add_(0, batch_idx, torch.ones(T_l, device=device, dtype=torch.long))
            per_layer_batch_lens.append(batch_lens)

        # concatenate scores across layers, grouped by batch for padded topk - yejun
        # total tokens per batch across all layers
        per_batch_total = torch.stack(per_layer_batch_lens).sum(dim=0)  # (B,)
        max_total = per_batch_total.max().item()

        # build (B, max_total) padded score tensor and track flat indices
        padded_scores = torch.full((B, max_total), float('-inf'), device=device, dtype=all_scores[0].dtype)
        # flat_indices[b] maps padded column -> (layer, flat_token_idx)
        batch_offsets = torch.zeros(B, device=device, dtype=torch.long)

        # per-layer flat masks, to be filled after topk
        flat_masks = [torch.zeros(s.shape[0], device=device, dtype=torch.bool) for s in all_scores]
        # mapping: for each (layer, token), what padded column it went to
        layer_padded_cols = []

        for l in range(num_layers):
            T_l = all_scores[l].shape[0]
            batch_idx = all_batch_idx[l]
            cu = self.cu_seqlens_k[l]
            # tokens are contiguous per batch, batch b starts at cu[b*H] - yejun
            batch_starts = cu[::H]  # (B+1,)
            within_batch_idx = torch.arange(T_l, device=device, dtype=torch.long) - batch_starts[batch_idx]
            padded_cols = within_batch_idx + batch_offsets[batch_idx]
            padded_scores[batch_idx, padded_cols] = all_scores[l]
            layer_padded_cols.append(padded_cols)
            batch_offsets += per_layer_batch_lens[l]

        # batched topk or threshold - yejun
        if "threshold" in strategy:
            padded_mask = padded_scores >= torch.log(torch.tensor(alpha_threshold, device=device))
        else:
            k = min(total_budget_per_batch, max_total)
            topk_indices = torch.topk(padded_scores, k, dim=-1, largest=True, sorted=False).indices  # (B, k)
            padded_mask = torch.zeros_like(padded_scores, dtype=torch.bool)
            padded_mask.scatter_(1, topk_indices, True)

        # scatter padded mask back to per-layer flat masks - yejun
        for l in range(num_layers):
            batch_idx = all_batch_idx[l]
            padded_cols = layer_padded_cols[l]
            flat_masks[l] = padded_mask[batch_idx, padded_cols]

        # apply masks and recompute head_lens per layer - yejun
        for l in range(num_layers):
            mask = flat_masks[l]
            self.key_cache[l] = self.key_cache[l][mask]
            self.value_cache[l] = self.value_cache[l][mask]
            self.retention_weights[l] = self.retention_weights[l][mask]
            self.kv_positions[l] = self.kv_positions[l][mask]

            per_seq_mask = torch.zeros(num_seqs, device=device, dtype=torch.int32)
            per_seq_mask.index_add_(0, all_seq_idx[l][mask], torch.ones(mask.sum(), device=device, dtype=torch.int32))
            self.head_lens[l] = per_seq_mask
            self.cu_seqlens_k[l] = torch.cumsum(
                torch.cat([torch.zeros(1, device=device, dtype=torch.int32), per_seq_mask]),
                dim=0, dtype=torch.int32,
            )

    def log(self):
        num_layers = len(self.head_lens)
        B = self.batch_size
        H = self.num_kv_heads

        head_wise_kv_positions = []
        for l in range(num_layers):
            for b in range(B):
                for h in range(H):
                    seq_idx = b * H + h
                    head_start = self.cu_seqlens_k[l][seq_idx].item()
                    head_end = self.cu_seqlens_k[l][seq_idx + 1].item()
                    head_kv_positions = self.kv_positions[l][head_start:head_end].detach().cpu()
                    head_wise_kv_positions.append(head_kv_positions)

        logs = {
            "batch_size": B,
            "num_kv_heads": H,
            "head_wise_kv_positions": head_wise_kv_positions,
            "flat_head_lens": torch.cat(self.head_lens).to('cpu').numpy(),
        }
        return logs

    def copy_to_device(self, device):
        raise NotImplementedError("copy_to_device not implemented for BatchedDynamicBudgetTrimKVCache")

class PagedDynamicBudgetTrimKVCache(TrimKVCache):

    def __init__(
        self,
        max_seq_len: int = 20480,
        page_size: int = 256,
        _distributed_cache_data: Iterable = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        assert page_size % 256 == 0, f"page_size must be divisible by 256 (flash_attn requirement), got {page_size}"

        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.retention_weights: List[torch.Tensor] = []
        self.kv_positions: List[torch.Tensor] = []

        self.block_tables: List[torch.Tensor] = []
        self.cache_seqlens: List[torch.Tensor] = []
        self.free_blocks: List[List[int]] = []

        self.batch_size: Optional[int] = None
        self.num_kv_heads: Optional[int] = None
        self._per_batch_seen_tokens: Optional[torch.Tensor] = None

        self.device = device
        self.offset = torch.tensor(0, dtype=torch.int64)
        self.block_mask = None

        if _distributed_cache_data is not None:
            raise NotImplementedError("Distributed cache data is not supported for PagedDynamicBudgetTrimKVCache")

    def _allocate_blocks(self, layer_idx: int, num_blocks: int) -> torch.Tensor:
        """Pop num_blocks from the free list. Returns int32 tensor of block indices."""
        free = self.free_blocks[layer_idx]
        assert len(free) >= num_blocks, f"Out of free blocks: need {num_blocks}, have {len(free)}"
        allocated = free[-num_blocks:]
        del free[-num_blocks:]
        return torch.tensor(allocated, dtype=torch.int32, device=self.device)

    def _release_blocks(self, layer_idx: int, block_indices):
        """Return blocks to the free list."""
        if isinstance(block_indices, torch.Tensor):
            block_indices = block_indices.tolist()
        self.free_blocks[layer_idx].extend(block_indices)

    def get_total_cached_tokens(self) -> int:
        return sum(self.cache_seqlens[l].sum().item() for l in range(len(self)))

    def get_cache_length(self, layer_idx: int = 0, batch_idx: int = 0, head_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        seq_idx = batch_idx * self.num_kv_heads + head_idx
        return self.cache_seqlens[layer_idx][seq_idx].item()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        retention_weights: torch.Tensor,
        cache_positions: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states is not None:
            B, H, S, D = key_states.shape

            if self.batch_size is None:
                self.batch_size = B
                self.num_kv_heads = H
            else:
                assert B == self.batch_size, f"Batch size changed: {self.batch_size} -> {B}"
                assert H == self.num_kv_heads, f"Num KV heads changed: {self.num_kv_heads} -> {H}"

            if layer_idx == 0:
                self._seen_tokens += S
                if self._per_batch_seen_tokens is not None:
                    self._per_batch_seen_tokens += S

            if cache_positions.dim() == 1:
                cache_positions = cache_positions[None, None, :].expand(B, H, S)

            num_seqs = B * H

            if len(self.key_cache) <= layer_idx:
                # Fill skipped layers with empty tensors
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    self.retention_weights.append(torch.tensor([]))
                    self.kv_positions.append(torch.tensor([]))
                    self.block_tables.append(torch.tensor([]))
                    self.cache_seqlens.append(torch.tensor([]))
                    self.free_blocks.append([])

                # --- Prefill: pre-allocate page pools and write tokens ---
                max_blocks_per_seq = (self.max_seq_len + self.page_size - 1) // self.page_size
                total_blocks = num_seqs * max_blocks_per_seq

                self.key_cache.append(torch.zeros(total_blocks, self.page_size, 1, D, device=self.device, dtype=key_states.dtype))
                self.value_cache.append(torch.zeros(total_blocks, self.page_size, 1, D, device=self.device, dtype=value_states.dtype))
                self.retention_weights.append(torch.zeros(total_blocks, self.page_size, device=self.device, dtype=retention_weights.dtype))
                self.kv_positions.append(torch.zeros(total_blocks, self.page_size, device=self.device, dtype=cache_positions.dtype))
                self.block_tables.append(torch.full((num_seqs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device))
                self.cache_seqlens.append(torch.zeros(num_seqs, dtype=torch.int32, device=self.device))
                self.free_blocks.append(list(range(total_blocks)))

                # Filter out left-padding tokens on prefill - yejun
                position_ids = self._position_ids  # (B, S)
                real_mask = self._prefill_padding_mask.bool()  # (B, S)
                cache_positions = position_ids.unsqueeze(1).expand(B, H, S)  # (B, H, S)

                real_lens = real_mask.sum(dim=1).int()  # (B,)

                if layer_idx == 0:
                    self._per_batch_seen_tokens = real_lens.to(device=self.device, dtype=torch.long)
                    self._seen_tokens = self._seen_tokens - S + real_lens.max().item()

                # Write tokens into pages per (batch, head)
                for b in range(B):
                    m = real_mask[b]  # (S,)
                    real_len = real_lens[b].item()
                    if real_len == 0:
                        continue

                    blocks_per_head = (real_len + self.page_size - 1) // self.page_size
                    allocated = self._allocate_blocks(layer_idx, H * blocks_per_head)
                    allocated = allocated.view(H, blocks_per_head)

                    token_idx = torch.arange(real_len, device=self.device)
                    block_local = token_idx // self.page_size
                    offsets = token_idx % self.page_size

                    for h in range(H):
                        seq_idx = b * H + h
                        self.block_tables[layer_idx][seq_idx, :blocks_per_head] = allocated[h]

                        block_global = allocated[h][block_local]  # (real_len,)
                        self.key_cache[layer_idx][block_global, offsets, 0, :] = key_states[b, h, m, :]
                        self.value_cache[layer_idx][block_global, offsets, 0, :] = value_states[b, h, m, :]
                        self.retention_weights[layer_idx][block_global, offsets] = retention_weights[b, h, m]
                        self.kv_positions[layer_idx][block_global, offsets] = cache_positions[b, h, m]

                    self.cache_seqlens[layer_idx][b * H:(b + 1) * H] = real_len

            else:
                # --- Decode: append tokens to existing pages ---
                cache_positions = self._position_ids.unsqueeze(1).expand(B, H, S)  # (B, H, S)

                for s in range(S):
                    cur_lens = self.cache_seqlens[layer_idx]  # (num_seqs,)
                    page_indices = cur_lens // self.page_size
                    offsets = cur_lens % self.page_size

                    # Allocate new pages where current page is full
                    need_new_page = (offsets == 0)
                    if need_new_page.any():
                        num_new = need_new_page.sum().item()
                        new_blocks = self._allocate_blocks(layer_idx, num_new)
                        seqs_needing = need_new_page.nonzero(as_tuple=True)[0]
                        self.block_tables[layer_idx][seqs_needing, page_indices[seqs_needing]] = new_blocks

                    # Get physical block index for each sequence
                    seq_arange = torch.arange(num_seqs, device=self.device)
                    block_indices = self.block_tables[layer_idx][seq_arange, page_indices]

                    # Write token s for all (batch, head) pairs at once
                    k_flat = key_states[:, :, s, :].reshape(num_seqs, D)
                    v_flat = value_states[:, :, s, :].reshape(num_seqs, D)
                    rw_flat = retention_weights[:, :, s].reshape(num_seqs)
                    pos_flat = cache_positions[:, :, s].reshape(num_seqs)

                    self.key_cache[layer_idx][block_indices, offsets, 0, :] = k_flat
                    self.value_cache[layer_idx][block_indices, offsets, 0, :] = v_flat
                    self.retention_weights[layer_idx][block_indices, offsets] = rw_flat
                    self.kv_positions[layer_idx][block_indices, offsets] = pos_flat

                    self.cache_seqlens[layer_idx] += 1

        flash_attn_kwargs = {
            "block_table": self.block_tables[layer_idx],
            "cache_seqlens": self.cache_seqlens[layer_idx],
            "batch_size": self.batch_size,
            "num_kv_heads": self.num_kv_heads,
        }

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx], flash_attn_kwargs

    @torch.inference_mode()
    def compress(
        self,
        strategy: str = "alpha",
        memory_size: int = 2048,
        buffer_size: int = 512,
        floor_budget_ratio: float = 0.,
        alpha_threshold: float = 0.0,
        num_layers: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        skip_layers: int = 0,
    ):
        num_layers = len(self.key_cache) if num_layers is None else num_layers
        num_key_value_heads = self.num_kv_heads if num_key_value_heads is None else num_key_value_heads
        B = self.batch_size
        H = num_key_value_heads

        tokens_per_batch = H * (memory_size + buffer_size)
        total_tokens = self.get_total_cached_tokens()
        if total_tokens <= B * tokens_per_batch * num_layers and "threshold" not in strategy:
            return

        if "layer_wise" in strategy:
            for l in range(num_layers):
                layer_tokens = self.cache_seqlens[l].sum().item()
                if layer_tokens <= B * H * (memory_size + buffer_size):
                    continue
                self._compress_layer_paged(
                    layer_idx=l,
                    memory_size=memory_size,
                    floor_budget_ratio=floor_budget_ratio,
                    strategy=strategy.replace("layer_wise_", ""),
                )
        else:
            self._compress_all_layers_paged(
                strategy=strategy,
                memory_size=memory_size,
                buffer_size=buffer_size,
                floor_budget_ratio=floor_budget_ratio,
                alpha_threshold=alpha_threshold,
                num_layers=num_layers,
            )

    def _read_scores_from_pages(self, layer_idx: int, strategy: str):
        """Read rw/pos from pages, compute importance scores.

        Returns:
            scores: (num_seqs, max_padded) with -inf for invalid positions
            valid_mask: (num_seqs, max_padded) bool
            max_padded: int
        """
        device = self.device
        B = self.batch_size
        H = self.num_kv_heads
        num_seqs = B * H
        page_size = self.page_size

        bt = self.block_tables[layer_idx]
        seqlens = self.cache_seqlens[layer_idx]
        max_seqlen = seqlens.max().item()
        max_blocks_used = (max_seqlen + page_size - 1) // page_size
        max_padded = max_blocks_used * page_size

        active_blocks = bt[:, :max_blocks_used].clamp(min=0).long()
        rw = self.retention_weights[layer_idx][active_blocks].reshape(num_seqs, max_padded)
        pos = self.kv_positions[layer_idx][active_blocks].reshape(num_seqs, max_padded)

        valid_mask = torch.arange(max_padded, device=device).unsqueeze(0) < seqlens.unsqueeze(1)

        batch_idx = torch.arange(num_seqs, device=device) // H
        q_idx = (self._per_batch_seen_tokens + 1)[batch_idx]

        scores = (q_idx.unsqueeze(1) - pos) * rw

        if strategy in ["knorm_alpha", "knorm_alpha_threshold"]:
            k_gathered = self.key_cache[layer_idx][active_blocks]
            key_norm = k_gathered.reshape(num_seqs, max_padded, -1).norm(dim=-1)
            scores = scores + torch.log(key_norm.clamp_min(1e-12))
        elif strategy not in ["alpha", "alpha_threshold"]:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        scores[~valid_mask] = float('-inf')
        return scores, valid_mask, max_padded

    def _apply_eviction_paged(self, layer_idx: int, keep_mask: torch.Tensor):
        """Fill-from-tail compaction: swap tail tokens into gaps.

        keep_mask: (num_seqs, max_padded) bool
        Cost: O(evicted_tokens) copies per sequence, not O(surviving_tokens).
        """
        bt = self.block_tables[layer_idx]
        seqlens = self.cache_seqlens[layer_idx]
        num_seqs = bt.shape[0]
        page_size = self.page_size

        for s in range(num_seqs):
            L = seqlens[s].item()
            mask_s = keep_mask[s, :L]
            K = mask_s.sum().item()

            if K == L:
                continue

            # Gaps: positions < K that are evicted (need to be filled)
            # Tails: positions >= K that are kept (need to move into gaps)
            gap_positions = (~mask_s[:K]).nonzero(as_tuple=True)[0]
            tail_positions = mask_s[K:].nonzero(as_tuple=True)[0] + K

            if gap_positions.shape[0] > 0:
                gap_page_idx = gap_positions // page_size
                gap_offsets = gap_positions % page_size
                tail_page_idx = tail_positions // page_size
                tail_offsets = tail_positions % page_size

                gap_blocks = bt[s, gap_page_idx].long()
                tail_blocks = bt[s, tail_page_idx].long()

                self.key_cache[layer_idx][gap_blocks, gap_offsets] = self.key_cache[layer_idx][tail_blocks, tail_offsets]
                self.value_cache[layer_idx][gap_blocks, gap_offsets] = self.value_cache[layer_idx][tail_blocks, tail_offsets]
                self.retention_weights[layer_idx][gap_blocks, gap_offsets] = self.retention_weights[layer_idx][tail_blocks, tail_offsets]
                self.kv_positions[layer_idx][gap_blocks, gap_offsets] = self.kv_positions[layer_idx][tail_blocks, tail_offsets]

            # Update seqlen and free empty tail pages
            self.cache_seqlens[layer_idx][s] = K
            new_num_blocks = (K + page_size - 1) // page_size if K > 0 else 0
            old_num_blocks = (L + page_size - 1) // page_size
            if new_num_blocks < old_num_blocks:
                freed = bt[s, new_num_blocks:old_num_blocks]
                self._release_blocks(layer_idx, freed[freed >= 0])
                bt[s, new_num_blocks:old_num_blocks] = -1

    def _compress_layer_paged(
        self,
        layer_idx: int,
        memory_size: int,
        floor_budget_ratio: float = 0.,
        strategy: str = "alpha",
    ):
        device = self.device
        B = self.batch_size
        H = self.num_kv_heads
        num_seqs = B * H

        scores, valid_mask, max_padded = self._read_scores_from_pages(layer_idx, strategy)

        # Per-batch topk: group H heads per batch
        scores_by_batch = scores.view(B, H * max_padded)
        valid_by_batch = valid_mask.view(B, H * max_padded)

        k = memory_size * H
        topk_k = min(k, scores_by_batch.shape[1])
        topk_indices = torch.topk(scores_by_batch, topk_k, dim=1, largest=True, sorted=False).indices

        keep_mask_flat = torch.zeros_like(scores_by_batch, dtype=torch.bool)
        keep_mask_flat.scatter_(1, topk_indices, True)
        keep_mask_flat &= valid_by_batch

        keep_mask = keep_mask_flat.view(num_seqs, max_padded)
        self._apply_eviction_paged(layer_idx, keep_mask)

    def _compress_all_layers_paged(
        self,
        strategy: str = "alpha",
        memory_size: int = 2048,
        buffer_size: int = 512,
        floor_budget_ratio: float = 0.,
        alpha_threshold: float = 0.0,
        num_layers: Optional[int] = None,
    ):
        device = self.device
        num_layers = len(self.key_cache) if num_layers is None else num_layers
        B = self.batch_size
        H = self.num_kv_heads

        total_budget_per_batch = num_layers * H * memory_size

        tokens_per_batch_current = sum(
            self.cache_seqlens[l].sum().item() for l in range(num_layers)
        ) // B
        if tokens_per_batch_current <= num_layers * H * (memory_size + buffer_size) and "threshold" not in strategy:
            return

        # Compute scores for all layers
        all_scores_by_batch = []
        all_valid_by_batch = []
        all_max_padded = []

        for l in range(num_layers):
            scores, valid_mask, max_padded = self._read_scores_from_pages(l, strategy)
            all_scores_by_batch.append(scores.view(B, H * max_padded))
            all_valid_by_batch.append(valid_mask.view(B, H * max_padded))
            all_max_padded.append(max_padded)

        # Pool per batch across all layers
        pooled_scores = torch.cat(all_scores_by_batch, dim=1)  # (B, total_cols)
        pooled_valid = torch.cat(all_valid_by_batch, dim=1)

        if "threshold" in strategy:
            keep_pooled = pooled_scores >= torch.log(torch.tensor(alpha_threshold, device=device))
        else:
            k = min(total_budget_per_batch, pooled_scores.shape[1])
            topk_indices = torch.topk(pooled_scores, k, dim=1, largest=True, sorted=False).indices
            keep_pooled = torch.zeros_like(pooled_scores, dtype=torch.bool)
            keep_pooled.scatter_(1, topk_indices, True)

        keep_pooled &= pooled_valid

        # Split back to per-layer masks and apply eviction
        col_offset = 0
        num_seqs = B * H
        for l in range(num_layers):
            cols = H * all_max_padded[l]
            layer_keep = keep_pooled[:, col_offset:col_offset + cols]
            keep_mask = layer_keep.view(num_seqs, all_max_padded[l])
            self._apply_eviction_paged(l, keep_mask)
            col_offset += cols

    def log(self):
        num_layers = len(self.cache_seqlens)
        B = self.batch_size
        H = self.num_kv_heads
        page_size = self.page_size

        head_wise_kv_positions = []
        for l in range(num_layers):
            for b in range(B):
                for h in range(H):
                    seq_idx = b * H + h
                    L = self.cache_seqlens[l][seq_idx].item()
                    if L == 0:
                        head_wise_kv_positions.append(torch.tensor([]))
                        continue
                    num_blocks = (L + page_size - 1) // page_size
                    blocks = self.block_tables[l][seq_idx, :num_blocks].long()
                    pos = self.kv_positions[l][blocks].reshape(-1)[:L]
                    head_wise_kv_positions.append(pos.detach().cpu())

        logs = {
            "batch_size": B,
            "num_kv_heads": H,
            "head_wise_kv_positions": head_wise_kv_positions,
            "flat_head_lens": torch.cat([self.cache_seqlens[l] for l in range(num_layers)]).to('cpu').numpy(),
        }
        return logs

    def copy_to_device(self, device):
        raise NotImplementedError("copy_to_device not implemented for PagedDynamicBudgetTrimKVCache")


__all__ = [
    "TrimKVCache",
    "DynamicBudgetTrimKVCache",
    "BatchedDynamicBudgetTrimKVCache",
    "PagedDynamicBudgetTrimKVCache",
]

