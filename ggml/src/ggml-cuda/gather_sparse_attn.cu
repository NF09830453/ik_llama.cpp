#include "indexer_pack.cuh"

// One program per (query token, kv head). Gathers the query's selected K/V rows (per-cell
// packed lists: 24-bit row id | 1<<24 valid) and runs a per-head online softmax over the
// gathered rows. Same attended set as the masked-dense path; only the fp reduction order
// differs.

#define GSA_MAX_H 16

// Per-element dequant accessors, one per supported KV-cache type. Each mirrors the
// matching ggml-quants.c dequantize_row_* EXACTLY -- including the asymmetric high-bit
// shifts between the first and second halves of q6_0/q5_0/q5_1 blocks (a classic
// silent-corruption trap; do not collapse into a single formula).
struct gsa_f16 { static __device__ inline float get(const char * row, int i) { return (float)*(const half *)(row + i*2); } };
struct gsa_f32 { static __device__ inline float get(const char * row, int i) { return *(const float *)(row + i*4); } };
struct gsa_q8_0 {
    static __device__ inline float get(const char * row, int i) {
        const block_q8_0 * b = (const block_q8_0 *)(row + (i/QK8_0)*sizeof(block_q8_0));
        return __half2float(b->d) * (float)b->qs[i % QK8_0];
    }
};
struct gsa_q6_0 {
    static __device__ inline float get(const char * row, int i) {
        const block_q6_0 * b = (const block_q6_0 *)(row + (i/QK6_0)*sizeof(block_q6_0));
        const int j = i % QK6_0;
        if (j < QK6_0/2) {
            const uint8_t h = b->qh[j % (QK6_0/4)] >> 4*(j / (QK6_0/4));
            return __half2float(b->d) * (float)(((b->qs[j] & 0x0F) | ((h << 4) & 0x30)) - 32);
        }
        const int j2 = j - QK6_0/2;
        const uint8_t h = b->qh[j2 % (QK6_0/4)] >> 4*(j2 / (QK6_0/4));
        return __half2float(b->d) * (float)(((b->qs[j2] >> 4) | ((h << 2) & 0x30)) - 32);
    }
};
struct gsa_q5_0 {
    static __device__ inline float get(const char * row, int i) {
        const block_q5_0 * b = (const block_q5_0 *)(row + (i/QK5_0)*sizeof(block_q5_0));
        const int j = i % QK5_0;
        const uint32_t qh = *(const uint32_t *)b->qh;
        if (j < QK5_0/2)
            return __half2float(b->d) * (float)(((b->qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16);
        return __half2float(b->d) * (float)(((b->qs[j-QK5_0/2] >> 4) | ((qh >> (j-4)) & 0x10)) - 16);
    }
};
struct gsa_q5_1 {
    static __device__ inline float get(const char * row, int i) {
        const block_q5_1 * b = (const block_q5_1 *)(row + (i/QK5_1)*sizeof(block_q5_1));
        const int j = i % QK5_1;
        const uint32_t qh = *(const uint32_t *)b->qh;
        const int x = (j < QK5_1/2)
            ? ((b->qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10))
            : ((b->qs[j-QK5_1/2] >> 4) | ((qh >> (j-4)) & 0x10));
        return __half2float(b->data.d) * (float)x + __half2float(b->data.m);
    }
};
struct gsa_q4_0 {
    static __device__ inline float get(const char * row, int i) {
        const block_q4_0 * b = (const block_q4_0 *)(row + (i/QK4_0)*sizeof(block_q4_0));
        const int j = i % QK4_0;
        const int q = (j < QK4_0/2 ? (b->qs[j] & 0x0F) : (b->qs[j-QK4_0/2] >> 4)) - 8;
        return __half2float(b->d) * (float)q;
    }
};
struct gsa_q4_1 {
    static __device__ inline float get(const char * row, int i) {
        const block_q4_1 * b = (const block_q4_1 *)(row + (i/QK4_1)*sizeof(block_q4_1));
        const int j = i % QK4_1;
        const int q = (j < QK4_1/2) ? (b->qs[j] & 0x0F) : (b->qs[j-QK4_1/2] >> 4);
        return __half2float(b->data.d) * (float)q + __half2float(b->data.m);
    }
};

template <typename k_t, typename v_t>
static __global__ void k_gather_sparse_attn(const char * __restrict__ q, const char * __restrict__ k,
        const char * __restrict__ v, const int * __restrict__ packed, float * __restrict__ dst,
        const float scale, const int d, const int H, const int K,
        const size_t k_nbytes, const size_t v_nbytes, const int n_kv,
        const size_t nb_q1, const size_t nb_q2,
        const size_t nb_k1, const size_t nb_k2,
        const size_t nb_v1, const size_t nb_v2,
        const size_t nb_o1, const size_t nb_o2) {
    const int t    = blockIdx.y;
    const int ikvh = blockIdx.x;
    const int tid  = threadIdx.x;
    const int nth  = blockDim.x;

    extern __shared__ float data[];
    float * q_s   = data;              // [H*d]
    float * acc_s = q_s + H*d;         // [H*d]
    float * red_s = acc_s + H*d;       // [nth/32] warp partials
    float * aux_s = red_s + nth/32;    // [H] m | [H] l | [H] rescale | [H] p
    float * m_s   = aux_s;
    float * r_s   = aux_s + H;
    float * l_s   = aux_s + 2*H;
    float * p_s   = aux_s + 3*H;

    for (int i = tid; i < H*d; i += nth) {
        const int h = i / d;
        q_s[i]   = *(const float *)(q + t*nb_q1 + (size_t)(ikvh*H + h)*nb_q2 + (size_t)(i % d)*sizeof(float));
        acc_s[i] = 0.0f;
        if (i < H) {
            m_s[i] = -INFINITY;
            l_s[i] = 0.0f;
        }
    }
    __syncthreads();

    packed += (size_t)t*K;
    for (int base = 0; base < K; ++base) {
        const int meta = packed[base];
        bool live = meta >= 0 && (meta >> 24) > 0;
        const int row = live ? (meta & 0xffffff) : 0;
        // bounds guard: a bad row/offset (stale packed slot) is treated as a dead slot
        // rather than faulting; live slots always satisfy row < n_kv by construction.
        if (live) {
            const size_t koff = (size_t)ikvh*nb_k2 + (size_t)row*nb_k1;
            const size_t voff = (size_t)ikvh*nb_v2 + (size_t)row*nb_v1;
            if (row >= n_kv || koff + nb_k1 > k_nbytes || voff + nb_v1 > v_nbytes) {
                live = false;
            }
        }

        for (int h = 0; h < H; ++h) {
            float partial = 0.0f;
            if (live) {
                const char * krow = k + (size_t)ikvh*nb_k2 + (size_t)row*nb_k1;
                for (int i = tid; i < d; i += nth) {
                    partial += q_s[h*d + i]*k_t::get(krow, i);
                }
            }
            partial += __shfl_xor_sync(0xffffffff, partial, 16);
            partial += __shfl_xor_sync(0xffffffff, partial, 8);
            partial += __shfl_xor_sync(0xffffffff, partial, 4);
            partial += __shfl_xor_sync(0xffffffff, partial, 2);
            partial += __shfl_xor_sync(0xffffffff, partial, 1);
            if (tid % 32 == 0) red_s[tid/32] = partial;
            __syncthreads();
            if (tid == 0) {
                float s = red_s[0];
                for (int w = 1; w < nth/32; ++w) s += red_s[w];
                if (live) {
                    const float sc = scale*s;
                    const float mn = fmaxf(m_s[h], sc);
                    r_s[h] = (m_s[h] == -INFINITY) ? 0.0f : expf(m_s[h] - mn);
                    p_s[h] = expf(sc - mn);
                    l_s[h] = l_s[h]*r_s[h] + p_s[h];
                    m_s[h] = mn;
                } else {
                    r_s[h] = 1.0f;
                    p_s[h] = 0.0f;
                }
            }
            __syncthreads();
            if (live) {
                const char * vrow = v + (size_t)ikvh*nb_v2 + (size_t)row*nb_v1;
                for (int i = tid; i < d; i += nth) {
                    acc_s[h*d + i] = acc_s[h*d + i]*r_s[h] + p_s[h]*v_t::get(vrow, i);
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < H*d; i += nth) {
        const int h = i / d;
        const int dim = i % d;
        // FA-compatible out layout {D,H,T}: element (dim,h,t) at (h + H*t)*d + dim
        *(float *)(dst + t*nb_o2 + (size_t)(ikvh*H + h)*nb_o1 + dim*sizeof(float)) =
            l_s[h] > 0.0f ? acc_s[i]/l_s[h] : 0.0f;
    }
}

void ggml_cuda_op_gather_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto q      = dst->src[0];
    auto k      = dst->src[1];
    auto v      = dst->src[2];
    auto packed = dst->src[3];

    const int d       = (int) q->ne[0];
    const int n_head  = (int) q->ne[2];
    const int H       = n_head / k->ne[2];
    const int K       = (int) packed->ne[0];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    dim3 grid(k->ne[2], (int) q->ne[1], 1);
    constexpr int nth = 128;
    const int smem = (2*H*d + nth/32 + 4*H)*sizeof(float);

    auto launch = [&](auto ktag, auto vtag) {
        k_gather_sparse_attn<decltype(ktag), decltype(vtag)><<<grid, nth, smem, ctx.stream()>>>(
                (const char *)q->data, (const char *)k->data, (const char *)v->data,
                (const int *)packed->data, (float *)dst->data, scale, d, H, K,
                ggml_nbytes(k), ggml_nbytes(v), (int)k->ne[1],
                q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                dst->nb[1], dst->nb[2]);
    };

    // full K x V cross product over the accessor set (keep in sync with GSA_TYPES
    // in indexer_pack.cuh -- regenerate this block when adding a type)
    #define GSA_TRY(kt_, ktag, vt_, vtag) if (k->type == kt_ && v->type == vt_) return launch(ktag{}, vtag{});
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_F16, gsa_f16, GGML_TYPE_Q4_1, gsa_q4_1)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_F32, gsa_f32, GGML_TYPE_Q4_1, gsa_q4_1)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_Q8_0, gsa_q8_0, GGML_TYPE_Q4_1, gsa_q4_1)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_Q6_0, gsa_q6_0, GGML_TYPE_Q4_1, gsa_q4_1)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_Q5_0, gsa_q5_0, GGML_TYPE_Q4_1, gsa_q4_1)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_Q5_1, gsa_q5_1, GGML_TYPE_Q4_1, gsa_q4_1)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_Q4_0, gsa_q4_0, GGML_TYPE_Q4_1, gsa_q4_1)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_F16, gsa_f16)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_F32, gsa_f32)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_Q8_0, gsa_q8_0)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_Q6_0, gsa_q6_0)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_Q5_0, gsa_q5_0)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_Q5_1, gsa_q5_1)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_Q4_0, gsa_q4_0)
    GSA_TRY(GGML_TYPE_Q4_1, gsa_q4_1, GGML_TYPE_Q4_1, gsa_q4_1)
    #undef GSA_TRY
    GGML_ABORT("unsupported K/V type combo for gather_sparse_attn");
}
