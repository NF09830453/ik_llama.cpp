#include "indexer_pack.cuh"

// TODO(c12, follow-up): pass inp_pos/kv_seqlen into indexer_pack like the Triton
// _decode_pack_kernel reference does; derive the force-append set arithmetically from
// n_kv (current partial block) and drop the seed + band01 tensors from the decode graph
// entirely. Pilot the Bug A fix FIRST -- one change per pilot.

template <typename seed_t>
static __global__ void k_indexer_pack(
        const int * __restrict__ topk,
        const char * __restrict__ seed,
        const char * __restrict__ band,
        int * __restrict__ dst,
        int n_kv, int kept, int total,
        size_t nb_topk1, size_t nb_seed1, size_t nb_band1, size_t nb_dst1) {
    const int row = blockIdx.x;
    const int j   = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= total) return;

    const int    * topk_row = (const int    *)((const char *)topk + (size_t)row * nb_topk1);
    const seed_t * seed_row = (const seed_t *)(seed + (size_t)row * nb_seed1);
    const float  * band_row = (const float  *)(band + (size_t)row * nb_band1);
    int          * dst_row  = (int          *)((char *)dst + (size_t)row * nb_dst1);

    if (j < kept) {
        const int id = topk_row[j];
        const bool ok = id >= 0 && id < n_kv && !isinf((float)seed_row[id]);
        dst_row[j] = ok ? (id | (1 << 24)) : -1;
    } else {
        int m = j - kept;
        int cell = -1;
        for (int c = 0; c < n_kv; ++c) {
            if (band_row[c] > 0.5f) {
                if (m == 0) { cell = c; break; }
                --m;
            }
        }
        bool dup = false;
        if (cell >= 0 && !isinf((float)seed_row[cell])) {
            // Read immutable topk_row instead of dst_row to avoid cross-block race
            for (int s = 0; s < kept; ++s) {
                if (topk_row[s] == cell) { dup = true; break; }
            }
        } else {
            cell = -1;
        }
        dst_row[j] = (cell >= 0 && !dup) ? (cell | (1 << 24)) : -1;
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
    const int per_row = (total + k_block - 1) / k_block;
    dim3 grid(n_tok, per_row, 1);

    if (seed->type == GGML_TYPE_F16) {
        k_indexer_pack<half><<<grid, k_block, 0, ctx.stream()>>>(
            (const int *)topk->data, (const char *)seed->data, (const char *)band->data,
            (int *)dst->data, n_kv, kept, total,
            topk->nb[1], seed->nb[1], band->nb[1], dst->nb[1]);
    } else {
        k_indexer_pack<float><<<grid, k_block, 0, ctx.stream()>>>(
            (const int *)topk->data, (const char *)seed->data, (const char *)band->data,
            (int *)dst->data, n_kv, kept, total,
            topk->nb[1], seed->nb[1], band->nb[1], dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}
