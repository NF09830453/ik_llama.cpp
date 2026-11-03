#include "gather_sparse_attn_prefill.cuh"
#include "gather_sparse_dequant.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
#include <mma.h>

// One warp per (query, KV group, 16-head tile). K/V are dequantized once
// per selected tile and reused by the GQA heads. All strides below are BYTES.
static __global__ void gather_prefill_mma(
        const char * q, const char * k, const char * v, const int * packed,
        char * dst, int d, int H, int slots, int nk, float scale,
        ggml_type kt, ggml_type vt,
        size_t q1, size_t q2, size_t k1, size_t k2, size_t v1, size_t v2,
        size_t o1, size_t o2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const int lane = threadIdx.x, t = blockIdx.x, group = blockIdx.y;
    const int h0 = blockIdx.z * 16;
    const int dp = (d + 15) / 16 * 16;
    extern __shared__ __align__(32) unsigned char arena[];
    half * sq = (half *)arena;
    half * sk = sq + 16*dp;
    half * sv = sk + 16*dp;
    half * prob = sv + 16*dp;
    float * acc = (float *)(prob + 256);
    float * tmp = acc + 16*dp;
    __shared__ float maximum[16], normalizer[16];
    __shared__ int ids[16];
    for (int i = lane; i < 16*dp; i += 32) {
        int h = i / dp, c = i % dp;
        sq[i] = __float2half(h0+h < H && c < d ?
            *(const float *)(q + t*q1 + (group*H+h0+h)*q2 + c*sizeof(float)) : 0.0f);
        acc[i] = 0.0f;
    }
    if (lane < 16) { maximum[lane] = -INFINITY; normalizer[lane] = 0.0f; }
    __syncthreads();
    for (int base = 0; base < slots; base += 16) {
        if (lane < 16) {
            int meta = base+lane < slots ? packed[(size_t)t*slots+base+lane] : -1;
            int id = meta & 0xffffff;
            ids[lane] = meta >= 0 && (meta >> 24) == 1 && id < nk ? id : -1;
        }
        __syncthreads();
        // Pack guarantees a contiguous live prefix. Early exit is uniform.
        if (ids[0] < 0) break;
        for (int i = lane; i < 16*dp; i += 32) {
            int r = i / dp, c = i % dp, id = ids[r];
            sk[i] = __float2half(id >= 0 && c < d ? gsa_dequant(kt, k + id*k1 + group*k2, c) : 0.0f);
            sv[i] = __float2half(id >= 0 && c < d ? gsa_dequant(vt, v + id*v1 + group*v2, c) : 0.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> scores;
        wmma::fill_fragment(scores, 0.0f);
        for (int c = 0; c < dp; c += 16) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
            wmma::load_matrix_sync(a, sq+c, dp);
            wmma::load_matrix_sync(b, sk+c, dp);
            wmma::mma_sync(scores,a,b,scores);
        }
        wmma::store_matrix_sync(tmp,scores,16,wmma::mem_row_major);
        __syncthreads();
        if (lane < 16) {
            float m = maximum[lane];
            for (int j = 0; j < 16; ++j) if (ids[j] >= 0) m = fmaxf(m,tmp[lane*16+j]*scale);
            // Empty rows/tiles never evaluate -inf - -inf.
            float alpha = m == -INFINITY ? 1.0f : expf(maximum[lane]-m);
            float sum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                float p = ids[j] >= 0 && m != -INFINITY ? expf(tmp[lane*16+j]*scale-m) : 0.0f;
                prob[lane*16+j] = __float2half(p);
                sum += p;
            }
            normalizer[lane] = normalizer[lane]*alpha + sum;
            maximum[lane] = m;
            for (int c = 0; c < dp; ++c) acc[lane*dp+c] *= alpha;
        }
        __syncthreads();
        for (int c = 0; c < dp; c += 16) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> out;
            wmma::load_matrix_sync(a,prob,16);
            wmma::load_matrix_sync(b,sv+c,dp);
            wmma::fill_fragment(out,0.0f);
            wmma::mma_sync(out,a,b,out);
            wmma::store_matrix_sync(tmp,out,16,wmma::mem_row_major);
            __syncthreads();
            for (int i = lane; i < 256; i += 32) acc[(i/16)*dp+c+i%16] += tmp[i];
            __syncthreads();
        }
    }
    for (int i = lane; i < 16*dp; i += 32) {
        int h = i / dp, c = i % dp;
        if (h0+h < H && c < d) {
            float value = normalizer[h] > 0.0f ? acc[i]/normalizer[h] : 0.0f;
            *(float *)(dst + t*o2 + (group*H+h0+h)*o1 + c*sizeof(float)) = value;
        }
    }
#endif
}


// v2: four warps cover a 64-cell tile. QK is split over selected rows;
// PV is split over output dimensions. Metadata and output contracts match v1.
// D is specialized so the compiler can fold padded strides and loop bounds.
template<int D, bool F16KV, bool ReuseKV, bool RegisterAccum>
static __global__ void gather_prefill_mma_v2(
        const char * q, const char * k, const char * v, const int * packed,
        char * dst, int H, int slots, int nk, float scale,
        ggml_type kt, ggml_type vt,
        size_t q1, size_t q2, size_t k1, size_t k2, size_t v1, size_t v2,
        size_t o1, size_t o2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int TK = 64, NW = 4, NTH = NW*32;
    const int tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int t = blockIdx.x, group = blockIdx.y, h0 = blockIdx.z*16;
    extern __shared__ __align__(32) unsigned char arena[];
    half * sq = (half *)arena;
    half * sk = sq + 16*D;
    // K is dead after QK; reuse its tile for V only after all readers finish.
    half * sv = ReuseKV ? sk : sk + TK*D;
    half * prob = sk + (ReuseKV ? 1 : 2)*TK*D;
    float * acc = (float *)(prob + 16*TK);
    // Same space is reused for QK (16 x TK) and NW independent PV tiles.
    float * tmp = acc + (RegisterAccum ? 0 : 16*D);
    __shared__ float maximum[16], normalizer[16], rescale[16];
    __shared__ int ids[TK];
    for (int i = tid; i < 16*D; i += NTH) {
        int h = i/D, c = i%D;
        sq[i] = __float2half(h0+h < H ?
            *(const float *)(q + t*q1 + (group*H+h0+h)*q2 + c*sizeof(float)) : 0.0f);
        if constexpr (!RegisterAccum) acc[i] = 0.0f;
    }
    if (tid < 16) { maximum[tid] = -INFINITY; normalizer[tid] = 0.0f; }
    // Obtain accumulator-element row IDs through the public WMMA load API.
    // No assumptions about the architecture-dependent fragment register layout.
    constexpr int NTILES = (D+63)/64;
    using fragment_c = wmma::fragment<wmma::accumulator,16,16,16,float>;
    fragment_c output_regs[NTILES];
    int output_rows[fragment_c::num_elements];
    if constexpr (RegisterAccum) {
        for (int i = lane; i < 256; i += 32) tmp[warp*256+i] = float(i/16);
    }
    __syncthreads();
    if constexpr (RegisterAccum) {
        fragment_c coordinates;
        wmma::load_matrix_sync(coordinates,tmp+warp*256,16,wmma::mem_row_major);
        #pragma unroll
        for (int e = 0; e < fragment_c::num_elements; ++e) output_rows[e] = int(coordinates.x[e]);
        #pragma unroll
        for (int j = 0; j < NTILES; ++j) wmma::fill_fragment(output_regs[j],0.0f);
    }
    // All row-ID reads must finish before any warp reuses tmp for QK scores.
    __syncthreads();
    for (int base = 0; base < slots; base += TK) {
        if (tid < TK) {
            int meta = base+tid < slots ? packed[(size_t)t*slots+base+tid] : -1;
            int id = meta & 0xffffff;
            ids[tid] = meta >= 0 && (meta >> 24) == 1 && id < nk ? id : -1;
        }
        __syncthreads();
        // Producer guarantees no holes in the live prefix.
        if (ids[0] < 0) break;
        for (int i = tid; i < TK*D; i += NTH) {
            int r = i/D, c = i%D, id = ids[r];
            if constexpr (F16KV) {
                sk[i] = id >= 0 ? ((const half *)(k+id*k1+group*k2))[c] : __float2half(0.0f);
                if constexpr (!ReuseKV) {
                    sv[i] = id >= 0 ? ((const half *)(v+id*v1+group*v2))[c] : __float2half(0.0f);
                }
            } else {
                sk[i] = __float2half(id >= 0 ? gsa_dequant(kt,k+id*k1+group*k2,c) : 0.0f);
                if constexpr (!ReuseKV) {
                    sv[i] = __float2half(id >= 0 ? gsa_dequant(vt,v+id*v1+group*v2,c) : 0.0f);
                }
            }
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> scores;
        wmma::fill_fragment(scores,0.0f);
        for (int c = 0; c < D; c += 16) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
            wmma::load_matrix_sync(a,sq+c,D);
            wmma::load_matrix_sync(b,sk+warp*16*D+c,D);
            wmma::mma_sync(scores,a,b,scores);
        }
        wmma::store_matrix_sync(tmp+warp*16,scores,TK,wmma::mem_row_major);
        __syncthreads();
        // A warp cooperates on each head's softmax and output rescaling.
        for (int h = warp; h < 16; h += NW) {
            float m = maximum[h];
            for (int j = lane; j < TK; j += 32)
                if (ids[j] >= 0) m = fmaxf(m,tmp[h*TK+j]*scale);
            for (int off = 16; off; off /= 2) m = fmaxf(m,__shfl_xor_sync(0xffffffff,m,off));
            const float alpha = m == -INFINITY ? 1.0f : expf(maximum[h]-m);
            float sum = 0.0f;
            for (int j = lane; j < TK; j += 32) {
                float p = ids[j] >= 0 && m != -INFINITY ? expf(tmp[h*TK+j]*scale-m) : 0.0f;
                prob[h*TK+j] = __float2half(p);
                sum += p;
            }
            for (int off = 16; off; off /= 2) sum += __shfl_xor_sync(0xffffffff,sum,off);
            // All lanes have consumed the previous maximum before it changes.
            __syncwarp();
            if (lane == 0) {
                normalizer[h] = normalizer[h]*alpha+sum;
                maximum[h] = m;
                if constexpr (RegisterAccum) rescale[h] = alpha;
            }
            if constexpr (!RegisterAccum) {
                for (int c = lane; c < D; c += 32) acc[h*D+c] *= alpha;
            }
        }
        __syncthreads();
        if constexpr (ReuseKV) {
            // The preceding block barrier orders every QK read of sk before
            // these writes through the aliased sv pointer. The following one
            // makes the complete V tile visible to every PV warp.
            for (int i = tid; i < TK*D; i += NTH) {
                int r = i/D, c = i%D, id = ids[r];
                if constexpr (F16KV) {
                    sv[i] = id >= 0 ? ((const half *)(v+id*v1+group*v2))[c] : __float2half(0.0f);
                } else {
                    sv[i] = __float2half(id >= 0 ? gsa_dequant(vt,v+id*v1+group*v2,c) : 0.0f);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            const int c = (j*NW+warp)*16;
            if (c >= D) continue; // uniform for this warp, including WMMA calls
            if constexpr (RegisterAccum) {
                #pragma unroll
                for (int e = 0; e < fragment_c::num_elements; ++e)
                    output_regs[j].x[e] *= rescale[output_rows[e]];
            }
            wmma::fragment<wmma::accumulator,16,16,16,float> out;
            wmma::fill_fragment(out,0.0f);
            for (int r = 0; r < TK; r += 16) {
                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
                wmma::load_matrix_sync(a,prob+r,TK);
                wmma::load_matrix_sync(b,sv+r*D+c,D);
                wmma::mma_sync(out,a,b,out);
            }
            if constexpr (RegisterAccum) {
                #pragma unroll
                for (int e = 0; e < fragment_c::num_elements; ++e) output_regs[j].x[e] += out.x[e];
            } else {
                wmma::store_matrix_sync(tmp+warp*256,out,16,wmma::mem_row_major);
                __syncwarp();
                for (int i = lane; i < 256; i += 32) acc[(i/16)*D+c+i%16] += tmp[warp*256+i];
                __syncwarp();
            }
        }
        __syncthreads();
    }
    if constexpr (RegisterAccum) {
        #pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            const int c0 = (j*NW+warp)*16;
            if (c0 >= D) continue;
            wmma::store_matrix_sync(tmp+warp*256,output_regs[j],16,wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                const int h = i/16;
                if (h0+h < H) {
                    const float value = normalizer[h] > 0.0f ? tmp[warp*256+i]/normalizer[h] : 0.0f;
                    *(float *)(dst+t*o2+(group*H+h0+h)*o1+(c0+i%16)*sizeof(float)) = value;
                }
            }
            __syncwarp();
        }
    } else {
        for (int i = tid; i < 16*D; i += NTH) {
            int h = i/D, c = i%D;
            if (h0+h < H) {
                float value = normalizer[h] > 0.0f ? acc[i]/normalizer[h] : 0.0f;
                *(float *)(dst+t*o2+(group*H+h0+h)*o1+c*sizeof(float)) = value;
            }
        }
    }

#endif
}


// Arbitrary strided views retain exact addressing. This fallback dequantizes
// once per live cell, rather than once per query, into a head-major FP16 buffer.
static __global__ void prefill_convert_strided(
        const char * src, half * dst, int d, int nk, ggml_type type,
        size_t nb1, size_t nb2) {
    const int row = blockIdx.x, head = blockIdx.y;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        dst[((size_t)head*nk+row)*d+c] = __float2half(
            gsa_dequant(type,src+row*nb1+head*nb2,c));
    }
}

static ggml_tensor prefill_bulk_kv(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src, ggml_cuda_pool_alloc<half> & storage) {
    ggml_tensor view = *src;
    if (src->type == GGML_TYPE_F16) return view;
    const int64_t d = src->ne[0], nk = src->ne[1], nh = src->ne[2];
    storage.alloc(d*nk*nh);
    auto convert = ggml_get_to_fp16_cuda(src->type);
    const size_t rb = ggml_row_size(src->type,d);
    view.data = storage.get();
    view.type = GGML_TYPE_F16;
    view.nb[0] = sizeof(half);
    if (convert && src->nb[2] == rb && src->nb[1] == rb*nh) {
        // Token-major interleaved heads: convert the active physical span once.
        convert(src->data,storage.get(),nk*nh,d,ctx.stream());
        view.nb[1] = d*nh*sizeof(half);
        view.nb[2] = d*sizeof(half);
    } else if (convert && src->nb[1] == rb) {
        // Head-major cache: convert ONLY active rows, not reserved capacity gaps.
        for (int64_t h = 0; h < nh; ++h) {
            convert((const char *)src->data+h*src->nb[2],storage.get()+h*nk*d,nk,d,ctx.stream());
        }
        view.nb[1] = d*sizeof(half);
        view.nb[2] = d*nk*sizeof(half);
    } else {
        prefill_convert_strided<<<dim3(nk,nh),128,0,ctx.stream()>>>(
            (const char *)src->data,storage.get(),d,nk,src->type,src->nb[1],src->nb[2]);
        view.nb[1] = d*sizeof(half);
        view.nb[2] = d*nk*sizeof(half);
    }
    view.nb[3] = d*nk*nh*sizeof(half);
    CUDA_CHECK(cudaGetLastError());
    return view;
}

template<int D, bool F16KV, bool ReuseKV, bool RegisterAccum>
static void enqueue_prefill_v2(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * k, const ggml_tensor * v, float scale, size_t shared) {
    const auto q = dst->src[0], pack = dst->src[3];
    auto kernel = gather_prefill_mma_v2<D,F16KV,ReuseKV,RegisterAccum>;
    CUDA_SET_SHARED_MEMORY_LIMIT(kernel, shared);
    const int H = q->ne[2]/k->ne[2];
    kernel<<<dim3(q->ne[1],k->ne[2],(H+15)/16),128,shared,ctx.stream()>>>(
        (const char *)q->data,(const char *)k->data,(const char *)v->data,
        (const int *)pack->data,(char *)dst->data,H,pack->ne[0],k->ne[1],scale,k->type,v->type,
        q->nb[1],q->nb[2],k->nb[1],k->nb[2],v->nb[1],v->nb[2],dst->nb[1],dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

template<int D, bool ReuseKV, bool RegisterAccum>
static bool launch_prefill_v2_variant(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float scale) {
    constexpr size_t shared = (16*D+(ReuseKV ? 1 : 2)*64*D+16*64)*sizeof(half) + ((RegisterAccum ? 0 : 16*D)+16*64)*sizeof(float);
    // Include compiler-reserved static shared memory as well as our arrays.
    // sm_120 currently reports 1408 bytes, not just the explicit 384 bytes.
    cudaFuncAttributes attributes;
    const bool bulk = getenv("IK_STEP5_PREFILL_INLINE_DEQUANT") == nullptr;
    const bool f16kv = bulk || (dst->src[1]->type == GGML_TYPE_F16 && dst->src[2]->type == GGML_TYPE_F16);
    if (f16kv) {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes,gather_prefill_mma_v2<D,true,ReuseKV,RegisterAccum>));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attributes,gather_prefill_mma_v2<D,false,ReuseKV,RegisterAccum>));
    }
    if (shared+attributes.sharedSizeBytes > ggml_cuda_info().devices[ctx.device].smpbo) return false;
    // Pool buffers survive through enqueue and use the same stream as attention,
    // including during CUDA graph capture/replay (same convention as dense FA).
    ggml_cuda_pool_alloc<half> k_f16(ctx.pool()), v_f16(ctx.pool());
    ggml_tensor k_view = *dst->src[1], v_view = *dst->src[2];
    if (bulk) {
        k_view = prefill_bulk_kv(ctx,dst->src[1],k_f16);
        v_view = prefill_bulk_kv(ctx,dst->src[2],v_f16);
    }
    const auto k = &k_view, v = &v_view;
    if (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) {
        enqueue_prefill_v2<D,true,ReuseKV,RegisterAccum>(ctx,dst,k,v,scale,shared);
    } else {
        enqueue_prefill_v2<D,false,ReuseKV,RegisterAccum>(ctx,dst,k,v,scale,shared);
    }
    return true;
}
template<int D, bool ReuseKV>
static bool select_prefill_accumulator(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float scale) {
    if (getenv("IK_STEP5_PREFILL_SHARED_ACCUM") != nullptr) {
        return launch_prefill_v2_variant<D,ReuseKV,false>(ctx,dst,scale);
    }
    return launch_prefill_v2_variant<D,ReuseKV,true>(ctx,dst,scale);
}

template<int D>
static bool launch_prefill_v2(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float scale) {
    if (getenv("IK_STEP5_PREFILL_SEPARATE_KV") != nullptr) {
        return select_prefill_accumulator<D,false>(ctx,dst,scale);
    }
    return select_prefill_accumulator<D,true>(ctx,dst,scale);
}
#endif

bool ggml_cuda_gather_sparse_attn_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    const auto q = dst->src[0], k = dst->src[1], v = dst->src[2], pack = dst->src[3];
    const int d = q->ne[0], H = q->ne[2]/k->ne[2];
    // Keep shared memory below 48 KiB; larger heads use existing gather fallback.
    if (d > 256 || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        q->nb[0] != sizeof(float) || ggml_cuda_info().devices[ctx.device].cc < 700) return false;
    const int dp = (d+15)/16*16;
    const size_t shared = 16*dp*(3*sizeof(half)+sizeof(float)) + 256*(sizeof(half)+sizeof(float));
    float scale; memcpy(&scale,dst->op_params,sizeof(scale));
    if (getenv("IK_STEP5_PREFILL_V1") == nullptr) {
        bool launched = false;
        switch (d) {
            case 64: launched = launch_prefill_v2<64>(ctx,dst,scale); break;
            case 128: launched = launch_prefill_v2<128>(ctx,dst,scale); break;
            case 192: launched = launch_prefill_v2<192>(ctx,dst,scale); break;
            case 256: launched = launch_prefill_v2<256>(ctx,dst,scale); break;
        }
        if (launched) return true;
    }
    gather_prefill_mma<<<dim3(q->ne[1],k->ne[2],(H+15)/16),32,shared,ctx.stream()>>>(
        (const char *)q->data,(const char *)k->data,(const char *)v->data,
        (const int *)pack->data,(char *)dst->data,d,H,pack->ne[0],k->ne[1],scale,k->type,v->type,
        q->nb[1],q->nb[2],k->nb[1],k->nb[2],v->nb[1],v->nb[2],dst->nb[1],dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    return false;
#endif
}
