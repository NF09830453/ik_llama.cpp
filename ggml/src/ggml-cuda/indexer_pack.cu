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
        const char * __restrict__ band,
        int        * __restrict__ dst,
        int n_kv, int kept, int total,
        size_t nb_topk1, size_t nb_seed1, size_t nb_band1, size_t nb_dst1) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ uint32_t s_mem[];
    uint32_t * seen = s_mem; // [n_words]
    __shared__ int s_count;

    const int n_words = (n_kv + 31) >> 5;
    for (int w = tid; w < n_words; w += nth) {
        seen[w] = 0u;
    }
    if (tid == 0) {
        s_count = 0;
    }
    __syncthreads();

    const int    * topk_row = (const int    *)((const char *)topk + (size_t)row * nb_topk1);
    const seed_t * seed_row = (const seed_t *)(seed + (size_t)row * nb_seed1);
    const float  * band_row = (const float  *)(band + (size_t)row * nb_band1);
    int          * dst_row  = (int          *)((char *)dst + (size_t)row * nb_dst1);

    // 1. Mark all causally-visible topk cells in the bitmask (automatically deduplicates!)
    for (int j = tid; j < kept; j += nth) {
        const int id = topk_row[j];
        if (id >= 0 && id < n_kv && !isinf((float)seed_row[id])) {
            atomicOr(&seen[id >> 5], 1u << (id & 31));
        }
    }

    // 2. Cooperatively scan band_row in parallel and mark all causally-visible band cells
    for (int c = tid; c < n_kv; c += nth) {
        if (band_row[c] > 0.5f && !isinf((float)seed_row[c])) {
            atomicOr(&seen[c >> 5], 1u << (c & 31));
        }
    }
    __syncthreads();

    // 3. Compact set bits into dst_row[0 .. valid_count - 1], globally ascending cell order
    //    (physical-major). Per-thread word popcounts + exclusive block scan give each thread
    //    a deterministic slot base -- no atomicAdd ordering, output is run-to-run stable.
    //    Thread tid owns words tid, tid+nth, tid+2*nth, ... so word ranges stay ascending.
    int cnt = 0;
    for (int w = tid; w < n_words; w += nth) {
        cnt += __popc(seen[w]);
    }
    __shared__ int s_scan[256]; // host launches k_block = 256
    s_scan[tid] = cnt;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nth; ++i) {
            const int v = s_scan[i];
            s_scan[i] = acc;
            acc += v;
        }
    }
    __syncthreads();
    int out_idx = s_scan[tid];
    for (int w = tid; w < n_words; w += nth) {
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
    if (tid == nth - 1) {
        s_count = s_scan[tid] + cnt; // total valid cells
    }
    __syncthreads();

    // 4. Fill remaining dead slots with -1
    const int valid_count = s_count;
    for (int j = valid_count + tid; j < total; j += nth) {
        dst_row[j] = -1;
    }
}

void ggml_cuda_op_indexer_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * topk = dst->src[0];
    const ggml_tensor * seed = dst->src[1];
    const ggml_tensor * band = dst->src[2];

    const int n_kv  = (int)seed->ne[0];
    const int kept  = (int)topk->ne[0];
    const int total = (int)dst->ne[0];
    const int n_tok = (int)dst->ne[1];

    constexpr int k_block = 256;
    const int n_words = (n_kv + 31) >> 5;
    GGML_ASSERT(n_words <= GSA_PACK_MAX_WORDS);
    const size_t smem = (size_t)n_words * sizeof(uint32_t);
    dim3 grid(n_tok, 1, 1);

    if (seed->type == GGML_TYPE_F16) {
        k_indexer_pack<half><<<grid, k_block, smem, ctx.stream()>>>(
            (const int *)topk->data, (const char *)seed->data, (const char *)band->data,
            (int *)dst->data, n_kv, kept, total,
            topk->nb[1], seed->nb[1], band->nb[1], dst->nb[1]);
    } else {
        k_indexer_pack<float><<<grid, k_block, smem, ctx.stream()>>>(
            (const int *)topk->data, (const char *)seed->data, (const char *)band->data,
            (int *)dst->data, n_kv, kept, total,
            topk->nb[1], seed->nb[1], band->nb[1], dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}
