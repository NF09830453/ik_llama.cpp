#include "indexer_pack.cuh"

#define GSA_MAX_D_PER_LANE 16 // supports head_dim up to 512 (16 * 32)
#define GSA_N_WARPS 8
#define GSA_NTH (GSA_N_WARPS * 32)

// v2: one CTA per KV head, one warp per query head, split-K over the packed slots.
#define GSA_V2_MAX_WARPS 32 // head_dim / n_head_kv must be <= this
#define GSA_V2_TKV 8        // KV rows dequantized per shared-memory tile

#include "gather_sparse_dequant.cuh"
#include "gather_sparse_attn_prefill.cuh"

// Dequantize one packed row into `dst` with the type switch hoisted out of the
// element loop (lets the compiler specialize/vectorize each case). `tid`/`nth`
// cooperatively stride the row; multiple rows may run in parallel by passing the
// per-row thread id/count.
static __device__ inline void gsa_dequant_row(ggml_type type, const char * row, float * dst,
        int d, int tid, int nth) {
    switch (type) {
        case GGML_TYPE_F16:
            for (int i = tid; i < d; i += nth) {
                dst[i] = __half2float(*(const half *)(row + i * sizeof(half)));
            }
            break;
        case GGML_TYPE_F32:
            for (int i = tid; i < d; i += nth) {
                dst[i] = *(const float *)(row + i * sizeof(float));
            }
            break;
        case GGML_TYPE_Q8_0:
            for (int i = tid; i < d; i += nth) {
                const block_q8_0 * b = (const block_q8_0 *)(row + (i / QK8_0) * sizeof(block_q8_0));
                dst[i] = __half2float(b->d) * (float)b->qs[i % QK8_0];
            }
            break;
        case GGML_TYPE_Q6_0:
            for (int i = tid; i < d; i += nth) {
                const block_q6_0 * b = (const block_q6_0 *)(row + (i / QK6_0) * sizeof(block_q6_0));
                const int j = i % QK6_0;
                if (j < QK6_0 / 2) {
                    const uint8_t h = b->qh[j % (QK6_0 / 4)] >> (4 * (j / (QK6_0 / 4)));
                    dst[i] = __half2float(b->d) * (float)(((b->qs[j] & 0x0F) | ((h << 4) & 0x30)) - 32);
                } else {
                    const int j2 = j - QK6_0 / 2;
                    const uint8_t h = b->qh[j2 % (QK6_0 / 4)] >> (4 * (j2 / (QK6_0 / 4)));
                    dst[i] = __half2float(b->d) * (float)(((b->qs[j2] >> 4) | ((h << 2) & 0x30)) - 32);
                }
            }
            break;
        case GGML_TYPE_Q5_0:
            for (int i = tid; i < d; i += nth) {
                const block_q5_0 * b = (const block_q5_0 *)(row + (i / QK5_0) * sizeof(block_q5_0));
                const int j = i % QK5_0;
                const int xh = ((b->qh[j >> 3] >> (j & 7)) & 1) << 4;
                const int xl = (j < QK5_0 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK5_0 / 2] >> 4);
                dst[i] = __half2float(b->d) * (float)((xl | xh) - 16);
            }
            break;
        case GGML_TYPE_Q5_1:
            for (int i = tid; i < d; i += nth) {
                const block_q5_1 * b = (const block_q5_1 *)(row + (i / QK5_1) * sizeof(block_q5_1));
                const int j = i % QK5_1;
                const int xh = ((b->qh[j >> 3] >> (j & 7)) & 1) << 4;
                const int xl = (j < QK5_1 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK5_1 / 2] >> 4);
                dst[i] = __half2float(b->data.d) * (float)(xl | xh) + __half2float(b->data.m);
            }
            break;
        case GGML_TYPE_Q4_0:
            for (int i = tid; i < d; i += nth) {
                const block_q4_0 * b = (const block_q4_0 *)(row + (i / QK4_0) * sizeof(block_q4_0));
                const int j = i % QK4_0;
                const int q = (j < QK4_0 / 2 ? (b->qs[j] & 0x0F) : (b->qs[j - QK4_0 / 2] >> 4)) - 8;
                dst[i] = __half2float(b->d) * (float)q;
            }
            break;
        case GGML_TYPE_Q4_1:
            for (int i = tid; i < d; i += nth) {
                const block_q4_1 * b = (const block_q4_1 *)(row + (i / QK4_1) * sizeof(block_q4_1));
                const int j = i % QK4_1;
                const int q = (j < QK4_1 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK4_1 / 2] >> 4);
                dst[i] = __half2float(b->data.d) * (float)q + __half2float(b->data.m);
            }
            break;
        default:
            for (int i = tid; i < d; i += nth) {
                dst[i] = 0.0f;
            }
            break;
    }
}

// ---------------------------------------------------------------------------
// v1: original per-query-head kernel (kept as a fallback / A/B via IK_GSA_V1).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// v2: one CTA per (KV head, split, token). All query heads in the GQA group
// share the dequantized K/V tile in shared memory (dequant each KV row once
// instead of once per query head). Each warp owns one query head; partials are
// combined by k_gather_sparse_attn_combine.
// ---------------------------------------------------------------------------
static __global__ void k_gather_sparse_attn_v2(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const int  * __restrict__ packed,
        float      * __restrict__ partial_acc,  // [n_splits, n_tokens, n_head, d]
        float      * __restrict__ partial_m,    // [n_splits, n_tokens, n_head]
        float      * __restrict__ partial_l,    // [n_splits, n_tokens, n_head]
        const float scale, const int d, const int H, const int K,
        const ggml_type k_type, const ggml_type v_type,
        const int n_kv, const int n_splits, const int n_tokens,
        const size_t nb_q1, const size_t nb_q2,
        const size_t nb_k1, const size_t nb_k2,
        const size_t nb_v1, const size_t nb_v2) {
    const int ikvh = blockIdx.x;        // kv head
    const int t    = blockIdx.y;        // token
    const int s    = blockIdx.z;        // split
    const int tid  = threadIdx.x;
    const int nth  = blockDim.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (warp >= H) {
        return;
    }
    const int qh = ikvh * H + warp;

    const int chunk = (K + n_splits - 1) / n_splits;
    const int start = s * chunk;
    const int end   = min(K, start + chunk);

    // Q for this warp's query head.
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

    extern __shared__ float smem[];
    float * s_k = smem;                    // [GSA_V2_TKV * d]
    float * s_v = s_k + GSA_V2_TKV * d;    // [GSA_V2_TKV * d]

    const int  * packed_row = packed + (size_t)t * K;
    const char * k_head     = k + (size_t)ikvh * nb_k2;
    const char * v_head     = v + (size_t)ikvh * nb_v2;

    // Threads are grouped per KV row so all rows of a tile dequant in parallel.
    const int tpr = nth / GSA_V2_TKV;
    const int dr  = tid / tpr;
    const int di  = tid % tpr;

    for (int base = start; base < end; base += GSA_V2_TKV) {
        const int nrows = min(GSA_V2_TKV, end - base);

        if (dr < nrows) {
            const int meta = packed_row[base + dr];
            const int row  = meta & 0xffffff;
            float * kdst = s_k + dr * d;
            float * vdst = s_v + dr * d;
            if (meta < 0 || (meta >> 24) <= 0 || row >= n_kv) {
                for (int i = di; i < d; i += tpr) {
                    kdst[i] = 0.0f;
                    vdst[i] = 0.0f;
                }
            } else {
                gsa_dequant_row(k_type, k_head + (size_t)row * nb_k1, kdst, d, di, tpr);
                gsa_dequant_row(v_type, v_head + (size_t)row * nb_v1, vdst, d, di, tpr);
            }
        }
        __syncthreads();

        for (int r = 0; r < nrows; ++r) {
            const int meta = packed_row[base + r];
            if (meta < 0 || (meta >> 24) <= 0) {
                continue;
            }
            const int row = meta & 0xffffff;
            if (row >= n_kv) {
                continue;
            }

            const float * krow = s_k + r * d;
            float dot = 0.0f;
            for (int c = 0; c * 32 < d; ++c) {
                const int dim = c * 32 + lane;
                if (dim < d) {
                    dot += q_reg[c] * krow[dim];
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            }

            const float sc     = scale * dot;
            const float m_prev = m_w;
            const float mn     = fmaxf(m_prev, sc);
            const float rs     = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - mn);
            const float p      = expf(sc - mn);
            l_w = l_w * rs + p;
            m_w = mn;

            const float * vrow = s_v + r * d;
            for (int c = 0; c * 32 < d; ++c) {
                const int dim = c * 32 + lane;
                if (dim < d) {
                    acc_reg[c] = acc_reg[c] * rs + p * vrow[dim];
                }
            }
        }
        __syncthreads();
    }

    const int n_head = (int)gridDim.x * H;
    const int pidx = (int)(((size_t)s * n_tokens + t) * n_head + qh);
    partial_m[pidx] = m_w;
    partial_l[pidx] = l_w;
    float * pacc = partial_acc + (size_t)pidx * d;
    for (int c = 0; c * 32 < d; ++c) {
        const int dim = c * 32 + lane;
        if (dim < d) {
            pacc[dim] = acc_reg[c];
        }
    }
}

// Combine the split-K partials with the flash-decoding rescale.
static __global__ void k_gather_sparse_attn_combine(
        char       * __restrict__ dst,
        const float * __restrict__ partial_acc,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        const int d, const int H, const int n_head, const int n_splits, const int n_tokens,
        const size_t nb_o1, const size_t nb_o2) {
    const int qh   = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = blockDim.x >> 5;

    __shared__ float s_mg;
    __shared__ float s_lg;
    if (tid == 0) {
        float m_global = -INFINITY;
        for (int s = 0; s < n_splits; ++s) {
            const int pidx = (int)(((size_t)s * n_tokens + t) * n_head + qh);
            m_global = fmaxf(m_global, partial_m[pidx]);
        }
        float l_global = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const int pidx = (int)(((size_t)s * n_tokens + t) * n_head + qh);
            const float ms = partial_m[pidx];
            const float w = (ms == -INFINITY || m_global == -INFINITY) ? 0.0f : expf(ms - m_global);
            l_global += partial_l[pidx] * w;
        }
        s_mg = m_global;
        s_lg = l_global;
    }
    __syncthreads();

    const float m_global = s_mg;
    const float l_global = s_lg;
    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

    char * dst_ptr = dst + (size_t)t * nb_o2 + (size_t)qh * nb_o1;
    for (int c = warp; c * 32 < d; c += nwarp) {
        const int dim = c * 32 + lane;
        if (dim >= d) {
            continue;
        }
        float acc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const int pidx = (int)(((size_t)s * n_tokens + t) * n_head + qh);
            const float ms = partial_m[pidx];
            const float w = (ms == -INFINITY || m_global == -INFINITY) ? 0.0f : expf(ms - m_global);
            acc += partial_acc[(size_t)pidx * d + dim] * w;
        }
        *(float *)(dst_ptr + (size_t)dim * sizeof(float)) = acc * inv_l;
    }
}

void ggml_cuda_op_gather_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int64_t _t0 = ggml_time_us();
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

    const int n_kv     = (int)k->ne[1];
    const int n_tokens = (int)q->ne[1];

    // Same packed-cell contract; unsupported devices/shapes retain the existing
    // gather implementation, never dense attention on a discarded sparse mask.
    if (n_tokens > 8 && ggml_cuda_gather_sparse_attn_prefill(ctx, dst)) {
        gsa_time_log("gather_sparse_attn_prefill", _t0, n_kv);
        return;
    }

    const bool use_v1 = getenv("IK_GSA_V1") != nullptr || H > GSA_V2_MAX_WARPS || K <= 0;

    if (use_v1) {
        dim3 grid(n_head, n_tokens, 1);
        const size_t smem = (size_t)(2 * GSA_N_WARPS + GSA_N_WARPS * d) * sizeof(float);

        k_gather_sparse_attn<<<grid, GSA_NTH, smem, ctx.stream()>>>(
            (const char *)q->data, (const char *)k->data, (const char *)v->data,
            (const int *)packed->data, (char *)dst->data, scale, d, H, K,
            k->type, v->type, n_kv,
            q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
            dst->nb[1], dst->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        gsa_time_log("gather_sparse_attn", _t0, n_kv);
        return;
    }

    // ~32 slots per split: enough CTAs to fill the GPU without a huge combine.
    int n_splits = (K + 31) / 32;
    if (n_splits < 1)   n_splits = 1;
    if (n_splits > 128) n_splits = 128;
    // Prefill already supplies parallelism across query rows. Avoid multiplying
    // its scratch allocation and CTA count by the decode split count. Each CTA
    // streams the complete selected set with the same online softmax as decode.
    if (n_tokens > 8) n_splits = 1;
    const int n_head_kv = (int)k->ne[2];

    ggml_cuda_pool_alloc<float> partial_acc(ctx.pool(), (int64_t)n_splits * n_tokens * n_head * d);
    ggml_cuda_pool_alloc<float> partial_m(ctx.pool(), (int64_t)n_splits * n_tokens * n_head);
    ggml_cuda_pool_alloc<float> partial_l(ctx.pool(), (int64_t)n_splits * n_tokens * n_head);

    const dim3 grid(n_head_kv, n_tokens, n_splits);
    const int  nth  = H * 32;
    const size_t smem = (size_t)2 * GSA_V2_TKV * d * sizeof(float);

    k_gather_sparse_attn_v2<<<grid, nth, smem, ctx.stream()>>>(
        (const char *)q->data, (const char *)k->data, (const char *)v->data,
        (const int *)packed->data,
        partial_acc.get(), partial_m.get(), partial_l.get(),
        scale, d, H, K, k->type, v->type, n_kv, n_splits, n_tokens,
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    k_gather_sparse_attn_combine<<<dim3(n_head, n_tokens, 1), 256, 0, ctx.stream()>>>(
        (char *)dst->data, partial_acc.get(), partial_m.get(), partial_l.get(),
        d, H, n_head, n_splits, n_tokens, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    gsa_time_log("gather_sparse_attn", _t0, n_kv);
}
