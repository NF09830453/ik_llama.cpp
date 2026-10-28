
#include "indexer_pack.cuh"

#define GSA_MAX_D_PER_LANE 16 // supports head_dim up to 512 (16 * 32)
#define GSA_N_WARPS 8
#define GSA_NTH (GSA_N_WARPS * 32)

static __device__ inline float gsa_dequant(ggml_type type, const char * row, int i) {
    switch (type) {
        case GGML_TYPE_F16:
            return __half2float(*(const half *)(row + i * sizeof(half)));
        case GGML_TYPE_F32:
            return *(const float *)(row + i * sizeof(float));
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * b = (const block_q8_0 *)(row + (i / QK8_0) * sizeof(block_q8_0));
            return __half2float(b->d) * (float)b->qs[i % QK8_0];
        }
        case GGML_TYPE_Q6_0: {
            const block_q6_0 * b = (const block_q6_0 *)(row + (i / QK6_0) * sizeof(block_q6_0));
            const int j = i % QK6_0;
            if (j < QK6_0 / 2) {
                const uint8_t h = b->qh[j % (QK6_0 / 4)] >> (4 * (j / (QK6_0 / 4)));
                return __half2float(b->d) * (float)(((b->qs[j] & 0x0F) | ((h << 4) & 0x30)) - 32);
            }
            const int j2 = j - QK6_0 / 2;
            const uint8_t h = b->qh[j2 % (QK6_0 / 4)] >> (4 * (j2 / (QK6_0 / 4)));
            return __half2float(b->d) * (float)(((b->qs[j2] >> 4) | ((h << 2) & 0x30)) - 32);
        }
        case GGML_TYPE_Q5_0: {
            const block_q5_0 * b = (const block_q5_0 *)(row + (i / QK5_0) * sizeof(block_q5_0));
            const int j = i % QK5_0;
            const int xh = ((b->qh[j >> 3] >> (j & 7)) & 1) << 4;
            const int xl = (j < QK5_0 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK5_0 / 2] >> 4);
            return __half2float(b->d) * (float)((xl | xh) - 16);
        }
        case GGML_TYPE_Q5_1: {
            const block_q5_1 * b = (const block_q5_1 *)(row + (i / QK5_1) * sizeof(block_q5_1));
            const int j = i % QK5_1;
            const int xh = ((b->qh[j >> 3] >> (j & 7)) & 1) << 4;
            const int xl = (j < QK5_1 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK5_1 / 2] >> 4);
            return __half2float(b->data.d) * (float)(xl | xh) + __half2float(b->data.m);
        }
        case GGML_TYPE_Q4_0: {
            const block_q4_0 * b = (const block_q4_0 *)(row + (i / QK4_0) * sizeof(block_q4_0));
            const int j = i % QK4_0;
            const int q = (j < QK4_0 / 2 ? (b->qs[j] & 0x0F) : (b->qs[j - QK4_0 / 2] >> 4)) - 8;
            return __half2float(b->d) * (float)q;
        }
        case GGML_TYPE_Q4_1: {
            const block_q4_1 * b = (const block_q4_1 *)(row + (i / QK4_1) * sizeof(block_q4_1));
            const int j = i % QK4_1;
            const int q = (j < QK4_1 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK4_1 / 2] >> 4);
            return __half2float(b->data.d) * (float)q + __half2float(b->data.m);
        }
        default:
            return 0.0f;
    }
}

static __global__ void k_gather_sparse_attn(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const int  * __restrict__ packed,
        char       * __restrict__ dst,
        const float scale, const int d, const int H, const int K,
        const ggml_type k_type, const ggml_type v_type,
        const int n_kv,
        const size_t nb_q1, const size_t nb_q2,
        const size_t nb_k1, const size_t nb_k2,
        const size_t nb_v1, const size_t nb_v2,
        const size_t nb_o1, const size_t nb_o2) {
    const int qh   = blockIdx.x;        // query head index [0 .. n_head - 1]
    const int t    = blockIdx.y;        // token index      [0 .. n_tokens - 1]
    const int ikvh = qh / H;            // kv head index    [0 .. n_head_kv - 1]
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Load Q into registers for this query head
    float q_reg[GSA_MAX_D_PER_LANE];
    float acc_reg[GSA_MAX_D_PER_LANE];
    const char * q_ptr = q + (size_t)t * nb_q1 + (size_t)qh * nb_q2;
    #pragma unroll
    for (int c = 0; c < GSA_MAX_D_PER_LANE; ++c) {
        const int dim = c * 32 + lane;
        q_reg[c]   = (dim < d) ? *(const float *)(q_ptr + (size_t)dim * sizeof(float)) : 0.0f;
        acc_reg[c] = 0.0f;
    }

    float m_w = -INFINITY;
    float l_w = 0.0f;

    const int  * packed_row = packed + (size_t)t * K;
    const char * k_head     = k + (size_t)ikvh * nb_k2;
    const char * v_head     = v + (size_t)ikvh * nb_v2;

    // Each warp independently processes every 8th slot with ZERO __syncthreads()!
    for (int base = warp; base < K; base += GSA_N_WARPS) {
        const int meta = packed_row[base];
        if (meta < 0) {
            break; // k_indexer_pack compacts valid cells to [0 .. valid_count - 1]
        }
        if ((meta >> 24) <= 0) {
            continue;
        }
        const int row = meta & 0xffffff;
        if (row >= n_kv) {
            continue;
        }

        const char * krow = k_head + (size_t)row * nb_k1;
        float dot = 0.0f;
        for (int c = 0; c * 32 < d; ++c) {
            const int dim = c * 32 + lane;
            if (dim < d) {
                dot += q_reg[c] * gsa_dequant(k_type, krow, dim);
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }

        const float sc     = scale * dot;
        const float m_prev = m_w;
        const float mn     = fmaxf(m_prev, sc);
        const float r      = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - mn);
        const float p      = expf(sc - mn);
        l_w = l_w * r + p;
        m_w = mn;

        const char * vrow = v_head + (size_t)row * nb_v1;
        for (int c = 0; c * 32 < d; ++c) {
            const int dim = c * 32 + lane;
            if (dim < d) {
                acc_reg[c] = acc_reg[c] * r + p * gsa_dequant(v_type, vrow, dim);
            }
        }
    }

    // Cross-warp Flash-Decoding combine in shared memory
    extern __shared__ float smem[];
    float * s_m   = smem;                           // [GSA_N_WARPS]
    float * s_l   = s_m + GSA_N_WARPS;              // [GSA_N_WARPS]
    float * s_acc = s_l + GSA_N_WARPS;              // [GSA_N_WARPS * d]

    if (lane == 0) {
        s_m[warp] = m_w;
        s_l[warp] = l_w;
    }
    for (int c = 0; c * 32 < d; ++c) {
        const int dim = c * 32 + lane;
        if (dim < d) {
            s_acc[warp * d + dim] = acc_reg[c];
        }
    }
    __syncthreads();

    // Warp 0 combines the 8 warp accumulators and writes dst
    if (warp == 0) {
        float m_global = -INFINITY;
        #pragma unroll
        for (int w = 0; w < GSA_N_WARPS; ++w) {
            m_global = fmaxf(m_global, s_m[w]);
        }

        float l_global = 0.0f;
        float w_scale[GSA_N_WARPS];
        #pragma unroll
        for (int w = 0; w < GSA_N_WARPS; ++w) {
            w_scale[w] = (s_m[w] == -INFINITY || m_global == -INFINITY) ? 0.0f : expf(s_m[w] - m_global);
            l_global  += s_l[w] * w_scale[w];
        }

        char * dst_ptr = dst + (size_t)t * nb_o2 + (size_t)qh * nb_o1;
        const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
        for (int c = 0; c * 32 < d; ++c) {
            const int dim = c * 32 + lane;
            if (dim < d) {
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < GSA_N_WARPS; ++w) {
                    sum += s_acc[w * d + dim] * w_scale[w];
                }
                *(float *)(dst_ptr + (size_t)dim * sizeof(float)) = sum * inv_l;
            }
        }
    }
}

void ggml_cuda_op_gather_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q      = dst->src[0];
    const ggml_tensor * k      = dst->src[1];
    const ggml_tensor * v      = dst->src[2];
    const ggml_tensor * packed = dst->src[3];

    const int d      = (int)q->ne[0];
    const int n_head = (int)q->ne[2];
    const int H      = n_head / (int)k->ne[2];
    const int K      = (int)packed->ne[0];

    GGML_ASSERT(d <= GSA_MAX_D_PER_LANE * 32);
    if (!ggml_gsa_kv_supported(k->type, v->type)) {
        GGML_ABORT("unsupported K/V type combo for gather_sparse_attn");
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    dim3 grid(n_head, (int)q->ne[1], 1);
    const size_t smem = (size_t)(2 * GSA_N_WARPS + GSA_N_WARPS * d) * sizeof(float);

    k_gather_sparse_attn<<<grid, GSA_NTH, smem, ctx.stream()>>>(
        (const char *)q->data, (const char *)k->data, (const char *)v->data,
        (const int *)packed->data, (char *)dst->data, scale, d, H, K,
        k->type, v->type, (int)k->ne[1],
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
        dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
