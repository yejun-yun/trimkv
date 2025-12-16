from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn


def power_stair(a: torch.Tensor, q_len=None) -> torch.Tensor:
    """
    a: (bz, num_head, L)
    returns M: (bz, num_head, L, L) with
        M[..., i, j] = a[..., j] ** (i-j+1)  for i >= j
                       0                    otherwise
    """
    L = a.size(-1)
    device = a.device                      # keep everything on the same device

    # row/col grids
    i = torch.arange(L, device=device).unsqueeze(1)   # (L,1)
    j = torch.arange(L, device=device).unsqueeze(0)   # (1,L)

    # (L,L)  ⇒ i-j+1 below diag, 0 above
    exps  = (i - j)
    mask  = (exps >= 0)                   # True where we keep a power
    exps = exps.clamp(min=0)
    exps  = exps.unsqueeze(0).unsqueeze(0)          # (1,1,L,L) for broadcasting

    base  = a.unsqueeze(-2)               # (bz,num_head,1,L)
    M     = (base ** exps) * mask         # broadcast power, then zero upper-tri

    if q_len is not None:
        M = M[:, :, -q_len:, :]

    return M


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

    return attn_output, attn_weights, None



def retention_gated_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    retention_weights: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    q_len = query.shape[-2]

    retention_weights = retention_weights.to(torch.float32)
    retention_weights = torch.exp(retention_weights)
    retention_weights = power_stair(retention_weights, q_len)  # (batch, num_key_value_heads, seqlen, seqlen)

    summerized_retention_weights = retention_weights.sum(dim=-1) if module.training else None

    retention_weights = repeat_kv(retention_weights, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights * retention_weights

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights, summerized_retention_weights


__all__ = [
    "eager_attention_forward",
    "retention_gated_attention_forward",
]
