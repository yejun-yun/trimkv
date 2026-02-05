import torch
from typing import List
from einops import rearrange

from torch import nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func



def dynamic_kv_budget_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float = None,
    dropout_p: float = 0.0,
    flash_attn_kwargs: dict = {},
    **kwargs,
):
    """
    Flash Attention implementation with dynamic KV budget support.
    This function wraps around the original flash attention implementation to
    handle per-head dynamic key-value budgets during attention computation. 
    Because each head may have different sequence lengths due to trimming,
    we need to assume that we are working with standard attention with 1 KV head per batch and 
    then flatten the batch and head dimensions together. We then use flash attention with varlen
    support to compute the attention outputs. We then reshape the outputs back to the original dimensions.

    Args:
        query: Query tensor [B, N_HEADS, Q_LEN, HEAD_DIM].
        key: Key tensor. [seqlen_k, HEAD_DIM], head dimension is already flattened.
        value: Value tensor. [seqlen_k, HEAD_DIM], head dimension is already flattened.
        attention_mask: Attention mask tensor. (Not used here).

    Returns:
        attn_output: Attention output tensor.
        None, None: Placeholder for compatibility.
    """
    # split query into a list of tensors with grouped attention heads
    head_lens = flash_attn_kwargs.get("head_lens", None)
    cu_seqlens_k = flash_attn_kwargs.get("cu_seqlens_k", None)
    B, N_HEADS, Q_LEN, HEAD_DIM = query.shape
    N_KV_HEADS = head_lens.shape[0]
    N_Q_PER_GROUP = N_HEADS // N_KV_HEADS

    assert B == 1, "Batch size greater than 1 not supported in dynamic KV budget attention."

    # two things to remember: FA2 varlen works with non-transposed inputs and no batch size
    query = query.transpose(1, 2).squeeze(0)  # (B, N_HEADS, Q_LEN, HEAD_DIM) -> (Q_LEN, N_HEADS, HEAD_DIM)
    # maybe try to use rearrange from einops for better readability?
    query_list = torch.split(query, N_Q_PER_GROUP, dim=1) # list of (Q_LEN, N_Q_PER_GROUP, HEAD_DIM), len = N_KV_HEADS
    packed_query = torch.cat(query_list, dim=0)  # (Q_LEN * N_KV_HEADS, N_Q_PER_GROUP, HEAD_DIM)
    # somehow split and cat seems faster than reshape or rearrange here

    cu_seqlens_q = torch.arange(0, Q_LEN * N_KV_HEADS + 1, step=Q_LEN, device=query.device, dtype=torch.int32)
    max_seqlen_q = Q_LEN
    max_seqlen_k = head_lens.max().item()

    # call flash attention with varlen support
    attn_output = flash_attn_varlen_func(
        packed_query,
        key.unsqueeze(1),  # (seqlen_k, 1, HEAD_DIM), assuming we are working with 1 KV head
        value.unsqueeze(1),  # (seqlen_k, 1, HEAD_DIM)
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        causal=True,
    )
    # reshape attn_output back to (B, Q_LEN, N_HEADS HEAD_DIM)
    attn_output_list = torch.split(attn_output, Q_LEN, dim=0)  # list of (Q_LEN, N_Q_PER_GROUP, HEAD_DIM)
    attn_output = torch.cat(attn_output_list, dim=1)  # (Q_LEN, N_HEADS, HEAD_DIM)
    attn_output = attn_output.unsqueeze(0)
    return attn_output, None, None


def batched_dynamic_kv_budget_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float = None,
    dropout_p: float = 0.0,
    flash_attn_kwargs: dict = {},
    **kwargs,
):
    """
    Flash Attention with dynamic KV budget support for batch_size >= 1.
    Supports both prefill (Q_LEN > 1) and decode (Q_LEN = 1) stages.

    Args:
        query: Query tensor [B, N_HEADS, Q_LEN, HEAD_DIM].
        key: Key tensor [total_kv_tokens, HEAD_DIM], flattened across all (batch, head) pairs.
        value: Value tensor [total_kv_tokens, HEAD_DIM], same layout as key.
        attention_mask: Attention mask tensor. (Not used here).

    Returns:
        attn_output: Attention output tensor.
        None, None: Placeholder for compatibility.
    """
    head_lens = flash_attn_kwargs.get("head_lens", None)
    cu_seqlens_k = flash_attn_kwargs.get("cu_seqlens_k", None)
    batch_size = flash_attn_kwargs.get("batch_size", None)
    num_kv_heads = flash_attn_kwargs.get("num_kv_heads", None)

    B, N_HEADS, Q_LEN, HEAD_DIM = query.shape

    if batch_size is None:
        batch_size = B
    if num_kv_heads is None:
        num_kv_heads = head_lens.shape[0] // batch_size

    N_KV_HEADS = num_kv_heads
    N_Q_PER_GROUP = N_HEADS // N_KV_HEADS
    num_seqs = batch_size * N_KV_HEADS

    query = query.transpose(1, 2).reshape(B * Q_LEN, N_HEADS, HEAD_DIM)
    query_groups = torch.split(query, N_Q_PER_GROUP, dim=1)
    query_groups = [g.view(B, Q_LEN, N_Q_PER_GROUP, HEAD_DIM) for g in query_groups]
    query_stacked = torch.stack(query_groups, dim=0)
    query_stacked = query_stacked.permute(1, 0, 2, 3, 4)
    packed_query = query_stacked.reshape(num_seqs * Q_LEN, N_Q_PER_GROUP, HEAD_DIM)

    cu_seqlens_q = torch.arange(
        0, num_seqs * Q_LEN + 1, Q_LEN,
        device=query.device, dtype=torch.int32
    )

    max_seqlen_q = Q_LEN
    max_seqlen_k = head_lens.max().item()

    attn_output = flash_attn_varlen_func(
        packed_query,
        key.unsqueeze(1),
        value.unsqueeze(1),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        causal=True,
    )

    attn_output = attn_output.view(B, N_KV_HEADS, Q_LEN, N_Q_PER_GROUP, HEAD_DIM)
    attn_output = attn_output.permute(0, 2, 1, 3, 4)
    attn_output = attn_output.reshape(B, Q_LEN, N_HEADS, HEAD_DIM)

    return attn_output, None, None
