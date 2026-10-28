#include "indexer_pack.cuh"

#undef GSA_PACK_DEBUG

template <typename seed_t>
static __global__ void k_indexer_pack(const int * __restrict__ topk, const seed_t * __restrict__ seed,
        const float * __restrict__ band, int * __restrict__ dst, int n_kv, int kept, int total) {
    const int row = blockIdx.x;
    const int j   = blockIdx.y*blockDim.x + threadIdx.x;
    if (j >= total) return;

    topk += (size_t)row*kept;
    dst  += (size_t)row*total;

    if (j < kept) {
        const int id = topk[j];
        // visible iff the causal seed at this cell is finite (padding/future/poisoned slots
        // hold -inf; own-region band cells hold a finite -BIG and count as visible).
        const bool ok = id >= 0 && id < n_kv && (float)seed[(size_t)row*n_kv + id] > -INFINITY;
        dst[j] = ok ? (id | (1 << 24)) : -1;
    } else {
        // (j - kept)-th force-visible own-region cell, scanned out of the band map.
        // Dedup: when kept >= n_kv - B the -BIG own-region cells can also sit in the topk
        // list; attending them twice would double their softmax weight (masked-dense is
        // immune -- mask 0 applied twice is the same mask), so a band cell already in the
        // kept slots is dropped here.
        int m = j - kept;
        int cell = -1;
        for (int c = 0; c < n_kv; ++c) {
            if (band[(size_t)row*n_kv + c] > 0.5f) {
                if (m == 0) { cell = c; break; }
                --m;
            }
        }
        bool dup = false;
        if (cell >= 0) {
            for (int s = 0; s < kept; ++s) {
                if (dst[s] == (cell | (1 << 24))) { dup = true; break; }
            }
        }
        dst[j] = (cell >= 0 && !dup) ? (cell | (1 << 24)) : -1;
    }
}

void ggml_cuda_op_indexer_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto topk = dst->src[0];
    auto seed = dst->src[1];
    auto band = dst->src[2];

    const int n_kv   = (int) seed->ne[0];
    const int kept   = (int) topk->ne[0];
    const int total  = (int) dst->ne[0];
    const int n_tok  = (int) dst->ne[1];

    constexpr int k_block = 256;
    const int per_row = (total + k_block - 1)/k_block;
    dim3 grid(n_tok, per_row, 1);

    if (seed->type == GGML_TYPE_F16) {
        k_indexer_pack<half><<<grid, k_block, 0, ctx.stream()>>>((const int *)topk->data,
                (const half *)seed->data, (const float *)band->data, (int *)dst->data, n_kv, kept, total);
    } else {
        k_indexer_pack<float><<<grid, k_block, 0, ctx.stream()>>>((const int *)topk->data,
                (const float *)seed->data, (const float *)band->data, (int *)dst->data, n_kv, kept, total);
    }
    CUDA_CHECK(cudaGetLastError());
}
