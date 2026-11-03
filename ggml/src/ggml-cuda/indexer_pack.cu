#include "indexer_pack.cuh"
#include <cmath>
#include <vector>
#include <atomic>

static std::vector<unsigned char> pack_check_read(ggml_backend_cuda_context & ctx, const ggml_tensor * x) {
    std::vector<unsigned char> bytes(ggml_nbytes(x));
    CUDA_CHECK(cudaMemcpyAsync(bytes.data(), x->data, bytes.size(), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    return bytes;
}

static float pack_check_value(const ggml_tensor * x, const std::vector<unsigned char> & bytes, int row, int c) {
    const auto p = bytes.data() + (size_t)row*x->nb[1] + (size_t)c*x->nb[0];
    if (x->type == GGML_TYPE_F32) { float v; memcpy(&v, p, sizeof(v)); return v; }
    GGML_ASSERT(x->type == GGML_TYPE_F16);
    ggml_fp16_t h; memcpy(&h, p, sizeof(h)); return ggml_fp16_to_fp32(h);
}

static void pack_check_dump(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int row,
        const std::vector<unsigned char> & reference, const std::vector<unsigned char> & seen) {
    const auto topk = dst->src[0], seed = dst->src[1], ref = dst->src[2];
    const auto force = dst->src[3], causal = dst->src[4];
    GGML_ASSERT(force && causal);
    const auto tk = pack_check_read(ctx, topk);
    const auto sd = pack_check_read(ctx, seed);
    const auto fv = pack_check_read(ctx, force);
    const auto cm = pack_check_read(ctx, causal);
    std::vector<int> selected(seed->ne[0], 0);
    for (int j = 0; j < topk->ne[0]; ++j) {
        int id; memcpy(&id, tk.data() + (size_t)row*topk->nb[1] + (size_t)j*topk->nb[0], sizeof(id));
        GGML_ASSERT(id >= 0 && id < seed->ne[0]);
        ++selected[id];
    }
    char path[256];
    snprintf(path, sizeof(path), "_c12_pack_witness_%lld.tsv", (long long)ggml_time_us());
    FILE * f = fopen(path, "wx");
    if (!f) { fprintf(stderr, "[PACK_CHECK] could not create witness %s\n", path); return; }
    fprintf(f, "# tensor=%s row=%d T=%lld B=%lld kept=%lld\n", dst->name, row,
            (long long)dst->ne[1], (long long)(dst->ne[0]-topk->ne[0]), (long long)topk->ne[0]);
    fprintf(f, "cell\ttopk_count\tseed\tcausal\tforce\treference\tpacked\n");
    for (int c = 0; c < seed->ne[0]; ++c) {
        fprintf(f, "%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n", c, selected[c],
                pack_check_value(seed, sd, row, c), pack_check_value(causal, cm, row, c),
                pack_check_value(force, fv, row, c), pack_check_value(ref, reference, row, c), (int)seen[c]);
    }
    fclose(f);
    fprintf(stderr, "[PACK_CHECK] witness=%s\n", path);
}

// Opt-in eager diagnostic. src[2] is produced by the real MASK_TOPK kernel,
// from the same topk and own-band tensors as the masked Step5 path.
static void check_packed_set(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ref = dst->src[2];
    if (!ref) return;
    GGML_ASSERT(ref->type == GGML_TYPE_F16 || ref->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int nk = (int)ref->ne[0], nt = (int)dst->ne[1], cap = (int)dst->ne[0];
    GGML_ASSERT(ref->ne[1] >= nt && ref->ne[2] == 1 && ref->ne[3] == 1);
    std::vector<int> ids((size_t)cap * nt);
    std::vector<unsigned char> masks(ggml_nbytes(ref));
    CUDA_CHECK(cudaMemcpyAsync(ids.data(), dst->data, ids.size()*sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(masks.data(), ref->data, masks.size(), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    for (int t = 0; t < nt; ++t) {
        std::vector<unsigned char> seen(nk, 0);
        bool dead = false;
        int count = 0, prev = -1;
        for (int j = 0; j < cap; ++j) {
            const int meta = ids[(size_t)t*cap+j];
            if (meta == -1) { dead = true; continue; }
            const int id = meta & 0xffffff;
            if (dead || meta < 0 || (meta >> 24) != 1 || id >= nk || id <= prev) {
                GGML_ABORT("[PACK_CHECK] malformed pack %s T=%d row=%d slot=%d meta=%d previous=%d n_kv=%d", dst->name, nt, t, j, meta, prev, nk);
            }
            seen[id] = 1; prev = id; ++count;
        }
        int expected = 0;
        for (int c = 0; c < nk; ++c) {
            const unsigned char * p = masks.data() + (size_t)t*ref->nb[1] + (size_t)c*ref->nb[0];
            float value;
            if (ref->type == GGML_TYPE_F32) { memcpy(&value, p, sizeof(value)); }
            else { ggml_fp16_t h; memcpy(&h, p, sizeof(h)); value = ggml_fp16_to_fp32(h); }
            if (std::isnan(value) || value == INFINITY) {
                GGML_ABORT("[PACK_CHECK] invalid reference %s row=%d cell=%d value=%g", dst->name, t, c, value);
            }
            const bool visible = std::isfinite(value);
            expected += visible;
            if (visible != (seen[c] != 0)) {
                pack_check_dump(ctx, dst, t, masks, seen);
                GGML_ABORT("[PACK_CHECK] SET MISMATCH %s T=%d row=%d cell=%d packed=%d reference=%g n_kv=%d live=%d", dst->name, nt, t, c, (int)seen[c], value, nk, count);
            }
        }
        static std::atomic<unsigned long long> checked{0};
        const auto n = ++checked;
        if (n <= 16 || n % 1024 == 0) {
            fprintf(stderr, "[PACK_CHECK] PASS rows=%llu tensor=%s T=%d row=%d n_kv=%d live=%d expected=%d\n", n, dst->name, nt, t, nk, count, expected);
        }
    }
}

// INDEXER_PACK position-only mode. Same visibility predicate and physical
// endpoint as the in-pack scan, but shared by every layer using this seed.
template<typename seed_t>
static __global__ void k_pack_positions(const char * seed, int * dst, int nk, size_t nb1) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const seed_t * values = (const seed_t *)(seed+(size_t)row*nb1);
    int last = -1;
    for (int c = tid; c < nk; c += blockDim.x) if (!isinf((float)values[c])) last = c;
    __shared__ int maxima[256];
    maxima[tid] = last;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (tid < offset) maxima[tid] = max(maxima[tid],maxima[tid+offset]);
        __syncthreads();
    }
    if (tid == 0) dst[row] = maxima[0];
}

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
        const int * __restrict__ positions,
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
    if (positions) {
        if (tid == 0) s_q_pos = positions[row];
    } else {
        int local_q_pos = -1;
        for (int c = tid; c < n_kv; c += nth) {
            if (!isinf((float)seed_row[c])) local_q_pos = c;
        }
        atomicMax(&s_q_pos, local_q_pos);
    }
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
    if (dst->op_params[0] == 1) {
        const ggml_tensor * seed = dst->src[0];
        GGML_ASSERT(seed && dst->type == GGML_TYPE_I32 && dst->ne[0] == seed->ne[1]);
        GGML_ASSERT(seed->nb[0] == ggml_type_size(seed->type));
        if (seed->type == GGML_TYPE_F16) {
            k_pack_positions<half><<<seed->ne[1],256,0,ctx.stream()>>>(
                (const char *)seed->data,(int *)dst->data,seed->ne[0],seed->nb[1]);
        } else {
            GGML_ASSERT(seed->type == GGML_TYPE_F32);
            k_pack_positions<float><<<seed->ne[1],256,0,ctx.stream()>>>(
                (const char *)seed->data,(int *)dst->data,seed->ne[0],seed->nb[1]);
        }
        CUDA_CHECK(cudaGetLastError());
        gsa_time_log("pack_positions",_t0,seed->ne[0]);
        return;
    }
    const ggml_tensor * topk = dst->src[0];
    const ggml_tensor * seed = dst->src[1];
    const ggml_tensor * positions = dst->src[5];
    GGML_ASSERT(!positions || (positions->type == GGML_TYPE_I32 &&
        ggml_is_contiguous(positions) && ggml_nelements(positions) == seed->ne[1]));

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
            (int *)dst->data, positions ? (const int *)positions->data : nullptr, n_kv, kept, total, B,
            topk->nb[1], seed->nb[1], dst->nb[1]);
    } else {
        k_indexer_pack<float><<<grid, k_block, smem, ctx.stream()>>>(
            (const int *)topk->data, (const char *)seed->data,
            (int *)dst->data, positions ? (const int *)positions->data : nullptr, n_kv, kept, total, B,
            topk->nb[1], seed->nb[1], dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    check_packed_set(ctx, dst);
    gsa_time_log("indexer_pack", _t0, n_kv);
}
