import os
import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg

from torch.nn.attention.flex_attention import (
    create_block_mask,
)

from .configuration_trimkv_phi3 import TrimKVPhi3Config
from trimkv.attn import get_attention_interface 
from trimkv.cache_utils import TrimKVCache


logger = logging.get_logger(__name__)
create_block_mask_compiled = torch.compile(create_block_mask)


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        print(f"[NaN/Inf detected in {name}]")
        return False
    return True


class TrimKVBaseModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for outputs of TrimKV models with past key values.
    It extends `BaseModelOutputWithPast` to include the retention loss.
    """

    def __init__(
        self,
        retention_loss: Optional[torch.FloatTensor] = None,
        retention_weights: Optional[torch.FloatTensor] = None,
        summarized_retention_weights: Optional[torch.FloatTensor] = None,
        last_ori_hidden_state: Optional[torch.FloatTensor] = None,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...], None]
    ):
        super().__init__(**kwargs)
        self.retention_loss = retention_loss
        self.retention_weights = retention_weights
        self.summarized_retention_weights = summarized_retention_weights
        self.last_ori_hidden_state = last_ori_hidden_state


class TrimKVCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Base class for outputs of TrimKV models with past key values and language modeling head.
    It extends `CausalLMOutputWithPast` to include the retention loss.
    """

    def __init__(
        self,
        retention_loss: Optional[torch.FloatTensor] = None,
        base_loss: Optional[torch.FloatTensor] = None,
        retention_weights: Optional[torch.FloatTensor] = None,
        summarized_retention_weights: Optional[torch.FloatTensor] = None,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...], None]
    ):
        super().__init__(**kwargs)
        self.retention_loss = retention_loss
        self.base_loss = base_loss
        self.retention_weights = retention_weights
        self.summarized_retention_weights = summarized_retention_weights


class Phi3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed

class RetentionGate(nn.Module):
    """
    Projects each attention-head vector (head_dim) to a single scalar,
    using a separate learnable linear layer per head.

    Input shape : (batch_size, seq_len, input_dim)
    Output shape: (batch_size, seq_len, num_heads)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.retention_gate_intermediate_size = config.retention_gate_intermediate_size
        self.linear1 = nn.Linear(self.hidden_size, self.retention_gate_intermediate_size, bias=True)
        self.linear2 = nn.Linear(self.retention_gate_intermediate_size, config.num_key_value_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_key_value_heads))

        self.act_fn = ACT2FN[config.hidden_act]
        self.reset_parameters()

    def reset_parameters(self):
        initializer_range = getattr(self.config, "initializer_range", 0.02)
        self.linear1.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        self.linear2.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.bias is not None:
            self.bias.data.fill_(self.config.retention_gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        out = self.linear1(x)  # (B, S, D) -> (B, S, H')
        out = self.act_fn(out)  # (B, S, H')
        out = self.linear2(out)  # (B, S, H') -> (B, S, H)
        out = out + self.bias  # (B, S, H)
        # avoid sigmoid here to prevent numerical issues
        out = F.logsigmoid(out)  # (B, S, H)
        return out


class RetentionGate2(nn.Module):
    """
    Projects each attention-head vector (head_dim) to a single scalar,
    using a separate learnable linear layer per head.

    Input shape : (batch_size, seq_len, input_dim)
    Output shape: (batch_size, seq_len, num_heads)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = self.head_dim * config.num_key_value_heads * 2
        self.retention_gate_intermediate_size = config.retention_gate_intermediate_size
        self.linear1 = nn.Linear(self.hidden_size, self.retention_gate_intermediate_size, bias=True)
        self.linear2 = nn.Linear(self.retention_gate_intermediate_size, config.num_key_value_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_key_value_heads))

        self.act_fn = ACT2FN[config.hidden_act]
        self.reset_parameters()

    def reset_parameters(self):
        initializer_range = getattr(self.config, "initializer_range", 0.02)
        self.linear1.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        self.linear2.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.bias is not None:
            self.bias.data.fill_(self.config.retention_gate_bias_init)

    def forward(self, x: torch.Tensor, transpose=True) -> torch.Tensor:
        # x: (B, H, S, D)
        out = x.transpose(1, 2) if transpose else x
        out = out.reshape(out.shape[0], out.shape[1], -1)  # (B, S, H*D)
        out = self.linear1(out)  # (B, S, D) -> (B, S, H')
        out = self.act_fn(out)  # (B, S, H')
        out = self.linear2(out)  # (B, S, H') -> (B, S, H)
        out = out + self.bias  # (B, S, H)
        # avoid sigmoid here to prevent numerical issues
        out = F.logsigmoid(out)  # (B, S, H)
        return out


class RetentionGate3(nn.Module):
    """
    Projects each attention-head vector (head_dim) to a single scalar,
    using a separate learnable linear layer per head.

    Input shape : (batch_size, seq_len, input_dim)
    Output shape: (batch_size, seq_len, num_heads)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = self.head_dim * config.num_key_value_heads * 2
        self.retention_gate_intermediate_size = config.retention_gate_intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.retention_gate_intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.retention_gate_intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.retention_gate_intermediate_size, config.num_key_value_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_key_value_heads))

        self.act_fn = ACT2FN[config.hidden_act]
        self.reset_parameters()

    def reset_parameters(self):
        initializer_range = getattr(self.config, "initializer_range", 0.02)
        self.gate_proj.weight.data.normal_(mean=0.0, std=initializer_range)
        self.up_proj.weight.data.normal_(mean=0.0, std=initializer_range)
        self.down_proj.weight.data.normal_(mean=0.0, std=initializer_range)

        if self.bias is not None:
            self.bias.data.fill_(self.config.retention_gate_bias_init)

    def forward(self, x: torch.Tensor, transpose=True) -> torch.Tensor:
        # x: (B, H, S, D) ->   # (B, S, H, D)
        out = x.transpose(1, 2) if transpose else x
        out = out.reshape(out.shape[0], out.shape[1], -1)  # (B, S, H*D)
        out = self.down_proj(self.act_fn(self.gate_proj(out)) * self.up_proj(out))
        out = out + self.bias  # (B, S, H)
        # avoid sigmoid here to prevent numerical issues
        out = F.logsigmoid(out)  # (B, S, H)
        return out


class TrimKVPhi3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TrimKVPhi3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        if config.retention_gate == 'rg':
            self.retention_gate = RetentionGate(config)
        elif config.retention_gate == 'rg2':
            self.retention_gate = RetentionGate2(config)

        op_size = config.num_attention_heads * self.head_dim + 2 * (config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(config.hidden_size, op_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.config.num_attention_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if self.config.retention_gate == 'rg':
            retention_weights = self.retention_gate(hidden_states).transpose(1, 2)
        else:
            retention_weights = None
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.retention_gate == 'rg2':
            kv_states = torch.cat([key_states, value_states], dim=-1)  # (B, H, S, 2*D)
            retention_weights = self.retention_gate(kv_states).transpose(1, 2)

        offset = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states, retention_weights, kv_positions, flash_attn_kwargs = past_key_values.update(key_states, value_states, retention_weights, cache_position, self.layer_idx, cache_kwargs)
        else:
            kv_positions = None

        if not vanilla_forward and self.training:
            attn_impl = self.config.attn_impl
            assert attn_impl in ["rg_attn_flex"], f"Unsupported attention implementation during training: {attn_impl}"
        else:
            attn_impl = self.config._attn_implementation

        attention_interface: Callable = get_attention_interface(attn_impl)

        attn_output, attn_weights, summarized_retention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            retention_weights=retention_weights,
            kv_positions=kv_positions if past_key_values is not None else None,
            offset=offset,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            flash_attn_kwargs=flash_attn_kwargs if past_key_values is not None else {},
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if self.training:
            return attn_output, attn_weights, retention_weights, summarized_retention_weights
        else:
            return attn_output, attn_weights, None, None


class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class TrimKVPhi3DecoderLayer(nn.Module):
    def __init__(self, config: TrimKVPhi3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TrimKVPhi3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config
        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        vanilla_forward: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, retention_weights, summarized_retention_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            vanilla_forward=vanilla_forward,
            **kwargs,
        )
        hidden_states = residual + self.resid_attn_dropout(hidden_states)  # main diff with Llama

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)  # main diff with Llama

        return (hidden_states, self_attn_weights if output_attentions else None, retention_weights, summarized_retention_weights)


class TrimKVPhi3PreTrainedModel(PreTrainedModel):
    config_class = TrimKVPhi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TrimKVPhi3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    _version = "0.0.5"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RetentionGate):
            if module.bias is not None:
                module.bias.data.fill_(self.config.retention_gate_bias_init)
        elif isinstance(module, RetentionGate2):
            if module.bias is not None:
                module.bias.data.fill_(self.config.retention_gate_bias_init)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        load_trimkv_weights=True,
        download_from='local',
        *model_args,
        **kwargs
    ):
        # Call the original method first
        if load_trimkv_weights:
            if download_from == 'wandb':
                import wandb
                api = wandb.Api()
                artifact = api.artifact(pretrained_model_name_or_path, type='model')
                if artifact is not None:
                    print(f"Using wandb artifact: {artifact.name}")
                    if not os.path.exists(artifact._default_root()) or not os.path.exists(os.path.join(artifact._default_root(), "trimkv_weights.pth")) or not os.path.exists(os.path.join(artifact._default_root(), "config.json")):
                        pretrained_model_name_or_path = artifact.download()
                        print(f"Downloaded model from wandb to: {pretrained_model_name_or_path}")
                    else:
                        pretrained_model_name_or_path = artifact._default_root()
                        print(f"Using existing local artifact at: {pretrained_model_name_or_path}")
                else:
                    raise ValueError(f"Artifact {pretrained_model_name_or_path} not found in wandb.")
            elif download_from == 'huggingface':
                from huggingface_hub import snapshot_download
                pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)
                print(f"Downloaded model from HuggingFace to: {pretrained_model_name_or_path}")
            elif download_from == 'local':
                print(f"Loading model from local path: {pretrained_model_name_or_path}")
            else:
                raise ValueError(f"Unsupported download_from value: {download_from}")

            config = TrimKVPhi3Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            if hasattr(config, "base_model"):
                base_model = config.base_model
            else:
                base_model = pretrained_model_name_or_path
                config.base_model = pretrained_model_name_or_path

            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            model = super().from_pretrained(base_model, config=config, *model_args, **kwargs)

            if os.path.exists(pretrained_model_name_or_path):
                gate_weights = torch.load(os.path.join(pretrained_model_name_or_path, "trimkv_weights.pth"))
                trainable_params = config.trainable_params.split("|")
                trainble_gate_state_keys = [
                    key for key in model.state_dict().keys() if any(
                        trainable_param in key for trainable_param in trainable_params
                    )
                ]
                # trainable_gate_state_keys and gate_weights.keys() should match
                if set(trainble_gate_state_keys) != set(gate_weights.keys()):
                    raise ValueError(
                        f"Mismatch between trainable gate state keys: {trainble_gate_state_keys} and loaded weights keys: {gate_weights.keys()}"
                    )

                model.load_state_dict(gate_weights, strict=False)
                print("Retention gate weights loaded successfully.")
            else:
                print("Could not load the trimkv gate weights.")
        else:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        return model


class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, config: TrimKVPhi3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class TrimKVPhi3Model(TrimKVPhi3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]

    Args:
        config: Phi3Config
    """

    def __init__(self, config: TrimKVPhi3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TrimKVPhi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Phi3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vanilla_forward: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = TrimKVCache(
                max_seq_len=self.config.max_seq_len,
                device=inputs_embeds.device,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        retention_weights = () if self.config.retention_gate is not None else None
        summarized_retention_weights = () if self.config.retention_gate is not None else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    vanilla_forward,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    vanilla_forward=vanilla_forward,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if self.config.retention_gate is not None and layer_outputs[2] is not None:
                retention_weights += (layer_outputs[2],)

            if self.config.retention_gate is not None and layer_outputs[3] is not None:
                summarized_retention_weights += (layer_outputs[3],)

        if retention_weights is not None and len(retention_weights) > 0:
            retention_weights = torch.stack(retention_weights, dim=1)
        else:
            retention_weights = None

        if summarized_retention_weights is not None and len(summarized_retention_weights) > 0:
            summarized_retention_weights = torch.stack(summarized_retention_weights, dim=1)
        else:
            summarized_retention_weights = None

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values is not None and self.config.compress_memory:
            past_key_values.compress(
                strategy=self.config.compress_strategy,
                memory_size=self.config.memory_size,
                buffer_size=self.config.buffer_size,
                floor_budget_ratio=self.config.floor_budget_ratio,
                num_layers=self.config.num_hidden_layers,
                num_key_value_heads=self.config.num_key_value_heads,
            )

        return TrimKVBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            retention_weights=retention_weights,
            summarized_retention_weights=summarized_retention_weights,
        )


class TrimKVPhi3ForCausalLM(TrimKVPhi3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = TrimKVPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        vanilla_forward: bool = False,
        base_logits: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            vanilla_forward=vanilla_forward,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        base_loss = None
        if self.training and not vanilla_forward:
            logits = None
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            slice_hidden_states = hidden_states[:, slice_indices, :]
            logits = self.lm_head(slice_hidden_states)

        retention_weights = outputs.retention_weights
        retention_loss = None

        loss = None
        if base_loss is not None and retention_loss is not None:
            loss = base_loss + self.config.retention_weight * retention_loss
        elif base_loss is not None:
            loss = base_loss
        elif retention_loss is not None:
            loss = retention_loss

        out = TrimKVCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            retention_loss=retention_loss,
            base_loss=base_loss,
        )
        return out

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- this model may need to switch between short and long rope, invalidating the cache in the
        # process

        # When the first time input length reached long and short factor switching point, enforce re-compute cache
        # It will cause downside of slower at this single token position, however, better than current failure.
        if (
            past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = TrimKVCache(
                    max_seq_len=past_key_values.max_seq_len,
                    device=past_key_values.device,
                )

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return model_inputs


__all__ = [
    "TrimKVPhi3PreTrainedModel",
    "TrimKVPhi3Model",
    "TrimKVPhi3ForCausalLM",
]
