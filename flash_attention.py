import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr
):
    # range of values handled by this stage
    if STAGE == 1:
        # from 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q

    elif STAGE == 2:
        # Used only fo r  the block in where there is a transition between non-masked and masked keys
        lo, hi = block_index_q * \
            BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        # The multiple of function just tells triton that lo is a multiple of BLOCK_SIZE_Q so triton can make some optimizaitons.
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)

    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr(0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # Compute qk
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_in[:, None]
        else:
            # Compute the maxmimum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)

        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i, m_ij)

        # Apply the correction factor for the previous l_i and the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        # The 3rd arg is an accumulator. so 1 * 2 + 3
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))


@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUJM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUJM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUJM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUJM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUJM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which blockin the sequence length to process
    block_index_q = tl.program_id(0)

    # This indcates which head and batch to pocess. Each program is assiciated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicates which batch this program is assiciated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicates the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # This allows to get the (SEQ_LEN, HEAD_DIM) block in the Q, K, V by indexing it by batch and head
    # Notice how we treat the data to be 1D in nature and we calculate the strides accordingly.
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # offs_kv: the offsets for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) = float("inf")

    # l_i: the running sum. We have one for every query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # The accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # tl.load moves data from HBM to SMEM
    Q_block = tl.load(Q_block_ptr)

    # STAGE = 3 if causal, else 1.

    # The following way of splitting prevents having an if-statement inside the loop, making sure there is no warp divergence.
    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN
        )


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # ctx: context, used for storage for stuff required during backward pass

        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        # Pre-allocating the output
        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        def grid(args):
            return (
                # Which group of queries are we working with
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * NUM_HEADS,  # Which head of which batch are we working with
                1,
            )

        # M is the logsumexp for the backward pass, one for each query.
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return 0
