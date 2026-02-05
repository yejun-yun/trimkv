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

                key_flat = key_states.reshape(-1, D)
                value_flat = value_states.reshape(-1, D)
                cache_positions_flat = cache_positions.reshape(-1, 1)
                retention_weights_flat = retention_weights.reshape(-1, 1)

                self.key_cache.append(key_flat)
                self.value_cache.append(value_flat)
                self.retention_weights.append(retention_weights_flat)
                self.kv_positions.append(cache_positions_flat)

                self.head_lens.append(torch.full((num_seqs,), S, device=self.device, dtype=torch.int32))
                self.cu_seqlens_k.append(torch.arange(0, num_seqs * S + 1, S, device=self.device, dtype=torch.int32))

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

        rw = self.retention_weights[layer_idx].squeeze(-1)
        kv_pos = self.kv_positions[layer_idx].squeeze(-1)

        q_idx = self.get_seq_length() + 1
        log_alpha = (q_idx - kv_pos) * rw

        if strategy == "knorm_alpha":
            key_norm = self.key_cache[layer_idx].norm(dim=-1).reshape(-1)
            scores = log_alpha + torch.log(key_norm.clamp_min(1e-12))
        elif strategy == "alpha":
            scores = log_alpha
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        new_key_chunks = []
        new_value_chunks = []
        new_rw_chunks = []
        new_pos_chunks = []
        new_head_lens = []

        for b in range(B):
            batch_start_seq = b * H
            batch_end_seq = (b + 1) * H

            batch_token_start = self.cu_seqlens_k[layer_idx][batch_start_seq].item()
            batch_token_end = self.cu_seqlens_k[layer_idx][batch_end_seq].item()

            batch_scores = scores[batch_token_start:batch_token_end]
            batch_tokens = batch_token_end - batch_token_start

            k = min(memory_size * H, batch_tokens)
            if floor_budget_ratio > 0:
                k = min(int(memory_size * H * (1 - floor_budget_ratio)), batch_tokens)

            topk_idx = torch.topk(batch_scores, k, largest=True, sorted=False).indices
            topk_mask = torch.zeros(batch_tokens, dtype=torch.bool, device=device)
            topk_mask.index_fill_(0, topk_idx, True)

            if floor_budget_ratio > 0:
                floor_per_head = int(memory_size * floor_budget_ratio)
                for h in range(H):
                    seq_idx = b * H + h
                    head_start = self.cu_seqlens_k[layer_idx][seq_idx].item() - batch_token_start
                    head_end = self.cu_seqlens_k[layer_idx][seq_idx + 1].item() - batch_token_start
                    head_len = head_end - head_start

                    head_mask = topk_mask[head_start:head_end]
                    already_selected = head_mask.sum().item()

                    need = max(0, floor_per_head - already_selected)
                    if need > 0 and head_len > already_selected:
                        head_scores = batch_scores[head_start:head_end]
                        head_scores_masked = head_scores.clone()
                        head_scores_masked[head_mask] = float('-inf')
                        floor_idx = torch.topk(head_scores_masked, min(need, head_len - already_selected), largest=True).indices
                        topk_mask[head_start + floor_idx] = True

            batch_key = self.key_cache[layer_idx][batch_token_start:batch_token_end][topk_mask]
            batch_value = self.value_cache[layer_idx][batch_token_start:batch_token_end][topk_mask]
            batch_rw = self.retention_weights[layer_idx][batch_token_start:batch_token_end][topk_mask]
            batch_pos = self.kv_positions[layer_idx][batch_token_start:batch_token_end][topk_mask]

            new_key_chunks.append(batch_key)
            new_value_chunks.append(batch_value)
            new_rw_chunks.append(batch_rw)
            new_pos_chunks.append(batch_pos)

            for h in range(H):
                seq_idx = b * H + h
                head_start = self.cu_seqlens_k[layer_idx][seq_idx].item() - batch_token_start
                head_end = self.cu_seqlens_k[layer_idx][seq_idx + 1].item() - batch_token_start
                new_head_lens.append(topk_mask[head_start:head_end].sum().item())

        self.key_cache[layer_idx] = torch.cat(new_key_chunks, dim=0)
        self.value_cache[layer_idx] = torch.cat(new_value_chunks, dim=0)
        self.retention_weights[layer_idx] = torch.cat(new_rw_chunks, dim=0)
        self.kv_positions[layer_idx] = torch.cat(new_pos_chunks, dim=0)

        self.head_lens[layer_idx] = torch.tensor(new_head_lens, device=device, dtype=torch.int32)
        self.cu_seqlens_k[layer_idx] = torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=torch.int32), self.head_lens[layer_idx]], dim=0),
            dim=0,
            dtype=torch.int32,
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

        total_budget_per_batch = num_layers * H * memory_size

        tokens_per_batch_current = sum(
            self.key_cache[l].shape[0] for l in range(num_layers)
        ) // B
        if tokens_per_batch_current <= num_layers * H * (memory_size + buffer_size) and "threshold" not in strategy:
            return

        q_idx = self.get_seq_length() + 1

        for b in range(B):
            batch_scores_list = []
            batch_layer_lens = []

            for l in range(num_layers):
                batch_start_seq = b * H
                batch_end_seq = (b + 1) * H
                batch_token_start = self.cu_seqlens_k[l][batch_start_seq].item()
                batch_token_end = self.cu_seqlens_k[l][batch_end_seq].item()

                rw = self.retention_weights[l][batch_token_start:batch_token_end].squeeze(-1)
                kv_pos = self.kv_positions[l][batch_token_start:batch_token_end].squeeze(-1)
                log_alpha = (q_idx - kv_pos) * rw

                if strategy in ["knorm_alpha", "knorm_alpha_threshold"]:
                    key_norm = self.key_cache[l][batch_token_start:batch_token_end].norm(dim=-1)
                    layer_scores = log_alpha + torch.log(key_norm.clamp_min(1e-12))
                elif strategy in ["alpha", "alpha_threshold"]:
                    layer_scores = log_alpha
                else:
                    raise ValueError(f"Unknown compression strategy: {strategy}")

                batch_scores_list.append(layer_scores)
                batch_layer_lens.append(batch_token_end - batch_token_start)

            all_batch_scores = torch.cat(batch_scores_list, dim=0)
            cu_batch_layer_lens = torch.cumsum(
                torch.tensor([0] + batch_layer_lens, dtype=torch.long), dim=0
            )

            if "threshold" in strategy:
                topk_mask = all_batch_scores >= torch.log(torch.tensor(alpha_threshold, device=device))
            else:
                k = min(total_budget_per_batch, all_batch_scores.shape[0])
                topk_idx = torch.topk(all_batch_scores, k, largest=True, sorted=False).indices
                topk_mask = torch.zeros_like(all_batch_scores, dtype=torch.bool)
                topk_mask.index_fill_(0, topk_idx, True)

            for l in range(num_layers):
                batch_start_seq = b * H
                batch_end_seq = (b + 1) * H
                batch_token_start = self.cu_seqlens_k[l][batch_start_seq].item()
                batch_token_end = self.cu_seqlens_k[l][batch_end_seq].item()

                layer_mask_start = cu_batch_layer_lens[l].item()
                layer_mask_end = cu_batch_layer_lens[l + 1].item()
                layer_mask = topk_mask[layer_mask_start:layer_mask_end]

                if b == 0:
                    self._batch_layer_masks = {}
                self._batch_layer_masks[(b, l)] = (batch_token_start, batch_token_end, layer_mask)

        for l in range(num_layers):
            new_key_chunks = []
            new_value_chunks = []
            new_rw_chunks = []
            new_pos_chunks = []
            new_head_lens = []

            for b in range(B):
                batch_token_start, batch_token_end, layer_mask = self._batch_layer_masks[(b, l)]

                new_key_chunks.append(self.key_cache[l][batch_token_start:batch_token_end][layer_mask])
                new_value_chunks.append(self.value_cache[l][batch_token_start:batch_token_end][layer_mask])
                new_rw_chunks.append(self.retention_weights[l][batch_token_start:batch_token_end][layer_mask])
                new_pos_chunks.append(self.kv_positions[l][batch_token_start:batch_token_end][layer_mask])

                for h in range(H):
                    seq_idx = b * H + h
                    head_start = self.cu_seqlens_k[l][seq_idx].item() - batch_token_start
                    head_end = self.cu_seqlens_k[l][seq_idx + 1].item() - batch_token_start
                    new_head_lens.append(layer_mask[head_start:head_end].sum().item())

            self.key_cache[l] = torch.cat(new_key_chunks, dim=0)
            self.value_cache[l] = torch.cat(new_value_chunks, dim=0)
            self.retention_weights[l] = torch.cat(new_rw_chunks, dim=0)
            self.kv_positions[l] = torch.cat(new_pos_chunks, dim=0)

            self.head_lens[l] = torch.tensor(new_head_lens, device=device, dtype=torch.int32)
            self.cu_seqlens_k[l] = torch.cumsum(
                torch.cat([torch.zeros(1, device=device, dtype=torch.int32), self.head_lens[l]], dim=0),
                dim=0,
                dtype=torch.int32,
            )

        del self._batch_layer_masks

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


__all__ = [
    "TrimKVCache",
    "DynamicBudgetTrimKVCache",
    "BatchedDynamicBudgetTrimKVCache",
]

