import torch
import triton
import triton.language as tl


@triton.jit
def _copy_old_values_kernel(
    dst_ptr,            # *[origin_len + H * state_len, D]
    src_ptr,            # *[origin_len, D]
    headlens_ptr,       # *[H] int32
    cu_headlens_ptr,    # *[H+1] int32
    D,                  # head_dim (runtime scalar)
    state_len,          # number of new tokens per head (runtime scalar)
    BLOCK: tl.constexpr # elements per program
):
    h = tl.program_id(0)          # which head
    b = tl.program_id(1)          # which block within this head's flattened segment

    # load per-head sizes/offsets
    headlen = tl.load(headlens_ptr + h, eviction_policy="evict_last")

    # old flattened segment (in elements) starts at cu_headlens[h] * D
    src_cum = tl.load(cu_headlens_ptr + h, eviction_policy="evict_last") * D

    # new flattened segment (in elements) starts at
    # (cu_headlens[h] + h * state_len) * D
    dst_cum = (
        tl.load(cu_headlens_ptr + h, eviction_policy="evict_last")
        + h * state_len
    ) * D

    # 1D block copy over this head's contiguous region of length (headlen * D)
    base = b * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < headlen * D

    vals = tl.load(src_ptr + src_cum + offs, mask=mask, other=0)
    tl.store(dst_ptr + dst_cum + offs, vals, mask=mask)


@triton.jit
def _insert_new_values_kernel(
    dst_ptr,            # *[origin_len + H * state_len, D]
    state_ptr,          # *[H * state_len, D]  (B=1 folded away)
    cu_headlens_ptr,    # *[H+1] int32
    D,                  # head_dim
    state_len,          # number of new tokens per head
    BLOCK_N: tl.constexpr
):
    # We use a 3D launch:
    #   pid0 -> head h
    #   pid1 -> new token index s in [0, state_len)
    #   pid2 -> block along D
    h   = tl.program_id(0)  # head id
    s   = tl.program_id(1)  # which new token for this head
    cb  = tl.program_id(2)  # block along D

    cols = cb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < D

    # For head h, its new segment starts at:
    #   new_cu[h] = cu_headlens[h] + h * state_len
    #
    # Old tokens occupy:
    #   rows [new_cu[h], new_cu[h] + headlen[h])
    #
    # First new token for this head is at:
    #   row0 = new_cu[h] + headlen[h] = cu_headlens[h+1] + h * state_len
    #
    # So token s goes to:
    #   row = cu_headlens[h+1] + h * state_len + s
    insert_row = (
        tl.load(cu_headlens_ptr + h + 1, eviction_policy="evict_last")
        + h * state_len
        + s
    ) * D

    # state is laid out as [H * state_len, D].
    # Row index for (h, s) is (h * state_len + s).
    state_row = (h * state_len + s) * D + cols

    vals = tl.load(state_ptr + state_row, mask=mask, other=0)
    tl.store(dst_ptr + insert_row + cols, vals, mask=mask)


def update_flatten_view_triton(
    cache: torch.Tensor,
    state: torch.Tensor,
    headlens: torch.Tensor,
    cu_headlens: torch.Tensor,
    *,
    COPY_BLOCK: int = 2048,
    INSERT_BLOCK: int = 256,
    num_warps_copy: int = 4,
    num_warps_insert: int = 4,
) -> torch.Tensor:
    """
    Args:
        cache:
            [origin_len, D] float16/float32/bfloat16
            (old flattened cache, sum(headlens) == origin_len)

        state:
            [B, H, state_len, D] same dtype as cache, *B must be 1*.
            We append `state_len` new tokens per head.

        headlens:
            [H] int32 (per-head lengths before appending)

        cu_headlens:
            [H+1] int32 (exclusive prefix-sum of headlens, cu_headlens[-1] == origin_len)

    Returns:
        out:
            [origin_len + H * state_len, D] same dtype as cache
            Flattened layout after appending `state_len` tokens per head.
    """
    # --- Checks & shapes -----------------------------------------------------
    assert cache.is_cuda and state.is_cuda, "Tensors must be on CUDA device"
    assert headlens.is_cuda and cu_headlens.is_cuda, "Index tensors must be on CUDA"
    assert headlens.dtype == torch.int32, "headlens must be int32"
    assert cu_headlens.dtype == torch.int32, "cu_headlens must be int32"
    assert cache.dtype == state.dtype, "cache/state dtypes must match"
    assert cache.is_contiguous(), "cache must be contiguous [origin_len, D]"
    assert headlens.is_contiguous() and cu_headlens.is_contiguous(), "index tensors must be contiguous"

    assert state.ndim == 4, "state must be [B, H, state_len, D]"
    B, H, state_len, D_state = state.shape
    assert B == 1, "This kernel currently assumes B == 1"
    origin_len, D = cache.shape
    assert D == D_state, "cache and state last dim (D) must match"

    assert headlens.shape[0] == H, "headlens must be length H"
    assert cu_headlens.numel() == H + 1, "cu_headlens must be length H+1"
    # (optional) sanity: cu_headlens[-1] == origin_len
    # assert cu_headlens[-1].item() == origin_len

    # Flatten away B and state_len for the kernel:
    #   [B, H, state_len, D] -> [H * state_len, D] (B==1)
    assert state.is_contiguous(), "state must be contiguous [B, H, state_len, D]"
    state_flat = state.view(H * state_len, D)

    out = torch.empty(
        (origin_len + H * state_len, D),
        dtype=cache.dtype,
        device=cache.device,
    )

    # --- Phase 1: copy old values per head into new flattened layout ---------
    max_headlen = int(headlens.max().item()) if H > 0 else 0
    max_blocks_per_head = (
        (max_headlen * D + COPY_BLOCK - 1) // COPY_BLOCK
        if max_headlen > 0 else 0
    )

    if max_blocks_per_head > 0:
        grid = (H, max_blocks_per_head)
        _copy_old_values_kernel[grid](
            out,
            cache,
            headlens,
            cu_headlens,
            D,
            state_len,
            BLOCK=COPY_BLOCK,
            num_warps=num_warps_copy,
            num_stages=2,
        )

    # --- Phase 2: insert `state_len` new rows per head -----------------------
    ncol_blocks = (D + INSERT_BLOCK - 1) // INSERT_BLOCK if D > 0 else 0
    if ncol_blocks > 0 and state_len > 0:
        # 3D grid: (H, state_len, blocks_along_D)
        grid_ins = (H, state_len, ncol_blocks)
        _insert_new_values_kernel[grid_ins](
            out,
            state_flat,
            cu_headlens,
            D,
            state_len,
            BLOCK_N=INSERT_BLOCK,
            num_warps=num_warps_insert,
            num_stages=2,
        )

    return out
