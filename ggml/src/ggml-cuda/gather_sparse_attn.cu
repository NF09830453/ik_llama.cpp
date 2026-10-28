#include "indexer_pack.cuh"

#define GSA_MAX_H 16

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
        const size_t k_nbytes, const size_t v_nbytes, const int n_kv,
        const size_t nb_q1, const size_t nb_q2,
        const size_t nb_k1, const size_t nb_k2,
        const size_t nb_v1, const size_t nb_v2,
        const size_t nb_o1, const size_t nb_o2) {
    const int t    = blockIdx.y;
    const int ikvh = blockIdx.x;
    const int tid  = threadIdx.x;
    const int nth  = blockDim.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int n_warps = nth / 32;

    extern __shared__ float data[];
    float * q_s   = data;                  // [H * d]
    float * acc_s = q_s + H * d;           // [H * d]
    float * kv_s  = acc_s + H * d;         // [d] dequantized K or V row buffer
    float * red_s = kv_s + d;              // [H * n_warps]
    float * m_s   = red_s + H * n_warps;   // [H]
    float * l_s   = m_s + H;               // [H]
    float * r_s   = l_s + H;               // [H]
    float * p_s   = r_s + H;               // [H]

    for (int i = tid; i < H * d; i += nth) {
        const int h   = i / d;
        const int dim = i % d;
        q_s[i]   = *(const float *)(q + (size_t)t * nb_q1 + (size_t)(ikvh * H + h) * nb_q2 + (size_t)dim * sizeof(float));
        acc_s[i] = 0.0f;
    }
    if (tid < H) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    packed += (size_t)t * K;
    for (int base = 0; base < K; ++base) {
        const int meta = packed[base];
        if (meta < 0 || (meta >> 24) <= 0) {
            continue;
        }
        const int row = meta & 0xffffff;
        const size_t koff = (size_t)ikvh * nb_k2 + (size_t)row * nb_k1;
        const size_t voff = (size_t)ikvh * nb_v2 + (size_t)row * nb_v1;
        if (row >= n_kv || koff + nb_k1 > k_nbytes || voff + nb_v1 > v_nbytes) {
            continue;
        }

        // 1. Dequantize K row ONCE for all H query heads in the GQA group
        const char * krow = k + koff;
        for (int i = tid; i < d; i += nth) {
            kv_s[i] = gsa_dequant(k_type, krow, i);
        }
        __syncthreads();

        // 2. Compute Q*K dot products across all H heads
        for (int h = 0; h < H; ++h) {
            float partial = 0.0f;
            for (int i = tid; i < d; i += nth) {
                partial += q_s[h * d + i] * kv_s[i];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, offset);
            }
            if (lane == 0) {
                red_s[h * n_warps + warp] = partial;
            }
        }
        __syncthreads();

        // 3. Online softmax state update (1 thread per head in parallel) + dequantize V row ONCE
        if (tid < H) {
            float s = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                s += red_s[tid * n_warps + w];
            }
            const float sc = scale * s;
            const float m_prev = m_s[tid];
            const float mn = fmaxf(m_prev, sc);
            const float r  = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - mn);
            const float p  = expf(sc - mn);
            r_s[tid] = r;
            p_s[tid] = p;
            l_s[tid] = l_s[tid] * r + p;
            m_s[tid] = mn;
        }
        const char * vrow = v + voff;
        for (int i = tid; i < d; i += nth) {
            kv_s[i] = gsa_dequant(v_type, vrow, i);
        }
        __syncthreads();

        // 4. Rescale and accumulate V across all H heads
        for (int i = tid; i < H * d; i += nth) {
            const int h   = i / d;
            const int dim = i % d;
            acc_s[i] = acc_s[i] * r_s[h] + p_s[h] * kv_s[dim];
        }
        __syncthreads();
    }

    // Write output using byte-pointer `char * dst`
    for (int i = tid; i < H * d; i += nth) {
        const int h   = i / d;
        const int dim = i % d;
        *(float *)(dst + (size_t)t * nb_o2 + (size_t)(ikvh * H + h) * nb_o1 + (size_t)dim * sizeof(float)) =
            l_s[h] > 0.0f ? acc_s[i] / l_s[h] : 0.0f;
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

    GGML_ASSERT(H <= GSA_MAX_H);
    if (!ggml_gsa_kv_supported(k->type, v->type)) {
        GGML_ABORT("unsupported K/V type combo for gather_sparse_attn");
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    dim3 grid(k->ne[2], (int)q->ne[1], 1);
    constexpr int nth = 128;
    const int n_warps = nth / 32;
    const int smem = (2 * H * d + d + H * n_warps + 4 * H) * sizeof(float);

    k_gather_sparse_attn<<<grid, nth, smem, ctx.stream()>>>(
        (const char *)q->data, (const char *)k->data, (const char *)v->data,
        (const int *)packed->data, (char *)dst->data, scale, d, H, K,
        k->type, v->type,
        ggml_nbytes(k), ggml_nbytes(v), (int)k->ne[1],
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
        dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
