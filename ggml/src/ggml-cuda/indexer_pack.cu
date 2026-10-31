#include "indexer_pack.cuh"

// TODO(c12, follow-up): pass inp_pos/kv_seqlen into indexer_pack like the Triton
// _decode_pack_kernel reference does; derive the force-append set arithmetically from
// n_kv (current partial block) and drop the seed + band01 tensors from the decode graph
// entirely. Pilot the Bug A fix FIRST -- one change per pilot.

// Max n_kv supported by shared-memory bitmask: 131072 bits = 16 KB shared memory
#define GSA_PACK_MAX_WORDS 4096

template <typename seed_t>
static __global__ void k_indexer_pack(
        const int  * __restrict__ topk,
        const char * __restrict__ seed,
        int        * __restrict__ dst,
        int n_kv, int kept, int total, int B,
        size_t nb_topk1, size_t nb_seed1, size_t nb_dst1) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ uint32_t s_mem[];
    uint32_t * seen = s_mem; // [n_words]
    __shared__ int s_count;
    __shared__ int s_q_pos;
    __shared__ int s_scan[256];

    const int n_words = (n_kv + 31) >> 5;
    for (int w = tid; w < n_words; w += nth) {
        seen[w] = 0u;
    }
    if (tid == 0) {
        s_count = 0;
        s_q_pos = -1;
    }
    __syncthreads();

    const int    * topk_row = (const int    *)((const char *)topk + (size_t)row * nb_topk1);
    const seed_t * seed_row = (const seed_t *)(seed + (size_t)row * nb_seed1);
    int          * dst_row  = (int          *)((char *)dst + (size_t)row * nb_dst1);

    // 1. Mark all causally-visible topk cells in the bitmask
    for (int j = tid; j < kept; j += nth) {
        const int id = topk_row[j];
        if (id >= 0 && id < n_kv && !isinf((float)seed_row[id])) {
            atomicOr(&seen[id >> 5], 1u << (id & 31));
        }
    }

    // 2. Force-append the query's own (partial) block. The query position is
    //    the highest cell visible in this row's seed mask. Do NOT derive it from
    //    n_kv/n_tok: n_kv is padded to the FA alignment (256) while the write
    //    head is not, so that derivation targets an empty future block and
    //    silently force-appends nothing. Reading it from the (graph-updated) mask
    //    row is also safe under CUDA-graph replay. Own block = [own0, q_pos].
    int local_q_pos = -1;
    for (int c = tid; c < n_kv; c += nth) {
        if (!isinf((float)seed_row[c])) {
            local_q_pos = c;
        }
    }
    atomicMax(&s_q_pos, local_q_pos);
    __syncthreads();

    const int q_pos = s_q_pos;
    if (q_pos >= 0) {
        const int own0 = (q_pos / B) * B;
        for (int c = own0 + tid; c <= q_pos; c += nth) {
            if (!isinf((float)seed_row[c])) {
                atomicOr(&seen[c >> 5], 1u << (c & 31));
            }
        }
    }
    __syncthreads();

    // 3. Contiguous word range [w_start, w_end) per thread -> true globally ascending cell order!
    const int words_per_thread = (n_words + nth - 1) / nth;
    const int w_start = min(tid * words_per_thread, n_words);
    const int w_end   = min(w_start + words_per_thread, n_words);

    int cnt = 0;
    for (int w = w_start; w < w_end; ++w) {
        cnt += __popc(seen[w]);
    }
    s_scan[tid] = cnt;
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nth; ++i) {
            const int v = s_scan[i];
            s_scan[i] = acc;
            acc += v;
        }
        s_count = acc;
        if (acc > total) {
            printf("BUG in k_indexer_pack: s_count=%d > total=%d (kept=%d, n_kv=%d)!\n",
                   acc, total, kept, n_kv);
        }
    }
    __syncthreads();

    int out_idx = s_scan[tid];
    for (int w = w_start; w < w_end; ++w) {
        uint32_t mask = seen[w];
        while (mask) {
            const int bit = __ffs(mask) - 1;
            mask &= mask - 1;
            if (out_idx < total) {
                dst_row[out_idx] = ((w << 5) + bit) | (1 << 24);
            }
            ++out_idx;
        }
    }

    // 4. Fill remaining dead slots with -1
    const int valid_count = s_count;
    for (int j = valid_count + tid; j < total; j += nth) {
        dst_row[j] = -1;
    }
}

void ggml_cuda_op_indexer_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int64_t _t0 = ggml_time_us();
    const ggml_tensor * topk = dst->src[0];
    const ggml_tensor * seed = dst->src[1];

    const int n_kv  = (int)seed->ne[0];
    const int kept  = (int)topk->ne[0];
    const int total = (int)dst->ne[0];
    const int n_tok = (int)dst->ne[1];

    constexpr int k_block = 256;
    const int n_words = (n_kv + 31) >> 5;
    GGML_ASSERT(n_words <= GSA_PACK_MAX_WORDS);
    const size_t smem = (size_t)n_words * sizeof(uint32_t);
    dim3 grid(n_tok, 1, 1);

    // force-append budget = own partial block (<= B); total = kept + B by construction
    const int B = total - kept;
    GGML_ASSERT(B > 0);

    if (seed->type == GGML_TYPE_F16) {
        k_indexer_pack<half><<<grid, k_block, smem, ctx.stream()>>>(
            (const int *)topk->data, (const char *)seed->data,
            (int *)dst->data, n_kv, kept, total, B,
            topk->nb[1], seed->nb[1], dst->nb[1]);
    } else {
        k_indexer_pack<float><<<grid, k_block, smem, ctx.stream()>>>(
            (const int *)topk->data, (const char *)seed->data,
            (int *)dst->data, n_kv, kept, total, B,
            topk->nb[1], seed->nb[1], dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    gsa_time_log("indexer_pack", _t0, n_kv);
}
