#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Step-5 sparse GQA indexer: persistent proxy-key cache plumbing (Slice 1) + proxy forward &
// read-back scoring telemetry (Slice 2). Semantic selection core lands in Slice 3 env knobs.
//
// kv_self.kr_l[il] ([indexer_head_size == proxy_dim, kv_size], idx_type_k) holds one proxy
// indexer key per cached cell on FULL-attn layers only (SWA layers carry no indexer):
//   - write:  ggml_cpy per graph, rows kv_head..kv_head+n_tokens, registered in
//             lctx.dsa_cache_copies so llama_context::update_cache_copies() re-points the
//             baked view_offs when a graph is reused (GLM-DSA dsa_cache_copies pattern).
//             Without that registration a reused decode graph keeps scattering this ubatch's
//             keys into the first ubatch's slot.
//   - defrag: rows follow their cells (generic kr_l block in llm_build_context::build_defrag).
//   - shift:  rows carry NO position encoding (RoPE is applied at READ-BACK), so a K-shift
//             leaves them untouched -- "copy-through" by construction (build_k_shift()).
//
// Slice 2 write source = the REAL proxy key: k_norm(k_proj(inpL)) over proxy_dim, cached RAW.
// k_norm is position-independent so it belongs at write time; only the pe RoPE (pe/nope split
// over indexer_rope_dim) is deferred to read-back, where the cached key is rotated by its cell's
// absolute position (lctx.inp_kv_pos, host-filled from kv_self.cells in llama_set_inputs) and the
// proxy query by inp_pos. RoPE-at-read-back keeps the Slice 1 key = position-free invariant above
// intact (harness T5 asserts it) and keeps block pooling over cached keys CSA-friendly.
// Shapes/score logic ported INLINE from build_deepseek2_dsa_indexer (build_deepseek2.cpp:374;
// k_norm layernorm+bias :417, pe/nope split rope_dim 32, per-head w + sum :488) -- port logic,
// not dependencies.

// ---- Slice 3 semantic knobs (env, read ONCE per process) ----
// Fork precedent: DSA_HADAMARD_DISABLE frozen at first build (build_deepseek2.cpp:440). Frozen
// values keep cache contents self-consistent within a process (flipping IK_HADAMARD mid-process
// would mix rotated/unrotated cached rows). Future IK_* knobs (IK_SPARSE/IK_CSA_*/IK_SSMAX_*)
// land here as semantic candidates arrive.
// IK_HADAMARD (default 1; IK_HADAMARD=0 disables): Walsh-Hadamard whitening of proxy keys
// BEFORE the idx_type_k cache write, de-rotated at read-back (ggml_hadamard is orthonormal,
// H^2 == I -- hadamard.cu accumulates 1/sqrt(2) per butterfly stage). Quantizing whitened keys
// spreads each dim's energy over all quant blocks -> cache-Q8 error decorrelates from the score
// (Q8 ~= F16 KV-cache precision win). Exact-arithmetic scores identical to IK_HADAMARD=0:
// rope(H(H k_raw)) == rope(k_raw), so all plumbing invariants (T5 RAW/position-free write,
// T6b split taps) hold under either knob value.
static bool ik_hadamard_enabled() {
    static const bool on = getenv("IK_HADAMARD") == nullptr || atoi(getenv("IK_HADAMARD")) != 0;
    return on;
}
// IK_SPARSE (default 0 = dense telemetry-only; IK_SPARSE=1 activates indexer selection):
// top-M rank-penalty sparse mask -> build_std_attention additive mask on FULL attn layers (SWA
// unchanged). Frozen per process like IK_HADAMARD (selection must stay consistent with the
// cached-key set across decode steps). Dense-equivalence oracle (G1 plumbing): at ctx <= topk the
// builder's clamp makes pen(rank)==0 for every key, so the sparse mask == the causal mask BITWISE
// and IK_SPARSE=1 reproduces IK_SPARSE=0 logits EXACTLY (Tier-1: dump logits under both knobs,
// diff). Default flips to 1 only after G1-G3 gates green (user call) -- until then opt-in.
static bool ik_sparse_enabled() {
    static const bool on = getenv("IK_SPARSE") != nullptr && atoi(getenv("IK_SPARSE")) != 0;
    return on;
}
// IK_CSA (default 0 = token-grain baseline; frozen per process like the knobs above): Stage B
// CSA block-compress candidate (block size B = hparams.indexer_csa_block, HF region_block_size=8).
// IK_CSA>0 pools cached proxy keys into slot-aligned blocks of B and scores BLOCK-grain (score of
// the pooled key broadcast to every slot of its block); the IK_SPARSE selection machinery downstream
// is untouched -- broadcast scores tie within a block, argsort groups ties contiguously, and the
// rank cutoff at multiples of B therefore snaps to WHOLE blocks (the doc's "top-M blocks, M=topk/B").
// WHAT gets pooled is THE open semantic question (no public modeling code -- derivation ladder in
// AGENTS.md), hence two ranked candidates:
//   IK_CSA=1 "pooling": block key = validity-weighted mean of the cached proxy keys (doc psi).
//            Pooling runs AFTER the read-back pe RoPE (position-resolved keys pooled per block).
//   IK_CSA=2 "z proj":  block key = validity-weighted mean of z_proj(inpL) rows -- z_t is
//            per-token HISTORY, so this candidate swaps the kv_write cache source to RAW z rows
//            (csa_z_norm_type="none" literally: the z output is NOT normalized -- matched exactly
//            by caching ggml_mul_mat(indexer_z, inpL) with no k_norm) and pools those instead.
//            k_norm(k_proj) is UNUSED in this mode -- a candidate semantic ranked by the tensor
//            contract (z.shape == k.shape, z_norm_type="none") over doc silence about z.
// Partial blocks divide by |Omega_b| (doc prose rule; the /B variant stays deferred). Slot-aligned
// grouping == position-aligned in steady single-sequence decode; multi-sequence/defrag alignment
// is a documented TODO. Pooling EXCLUDES non-visible kv slots (n_kv is the PADDED span -- empty
// cells carry allocator-garbage kr_l rows; unweighted pooling poisons block means with REAL values)
// via validity weights derived from the causal mask. IK_SPARSE=0 + IK_CSA>0 = telemetry-only
// block taps (oracle/G4); G1 dense-equivalence holds under every IK_CSA mode (ctx <= topk clamp
// zeroes every rank penalty regardless of score values).
static int ik_csa_mode() {
    static const int mode = getenv("IK_CSA") ? atoi(getenv("IK_CSA")) : 0;
    GGML_ASSERT(mode >= 0 && mode <= 2 && "IK_CSA: 0=token-grain, 1=pooled-k blocks, 2=z-proj blocks");
    return mode;
}
// Selection granularity (default 1 = BLOCK grain; frozen per process like the knobs above).
// Evidence (Oct, G3 needle @ 7.2k ctx): token-grain top-512 (7% keep) loses the needle while
// dense passes and --dsa-top-k 4096 (57% keep) retrieves it EXACTLY -> HF sparse_config.topk=512
// counts BLOCKS of region_block_size=8 (512*8 = 4096 kept tokens), consistent with
// compression_method=csa_block_compress being the architecture, not an option. Block-grain
// selection = block-pooled scores broadcast to slots (existing verified machinery) + argsort
// with kept-count topk*B; argsort groups intra-block ties contiguously and topk*B is a multiple
// of B, so the rank cutoff snaps to WHOLE blocks -- the DSA scatter mask is reused unchanged.
// IK_SEL_GRAIN=0 keeps the token-grain path as the regression anchor.
static bool ik_sel_block_grain() {
    static const int mode = getenv("IK_SEL_GRAIN") ? atoi(getenv("IK_SEL_GRAIN")) : 1;
    GGML_ASSERT(mode >= 0 && mode <= 1 && "IK_SEL_GRAIN: 0=token-grain selection, 1=block-grain (default)");
    return mode != 0;
}
// IK_SSMAX (default 0 = baseline; frozen per process like the knobs above): scalable-softmax
// per-q-head attention logit scale beta_h = s_h * ln(n) (Nakanishi 2025), s from the loaded
// blk.{il}.indexer.ssmax_s F32 {n_head} tensor. Granularity settled by tensor contract: ssmax_s
// is {64} = one scale per ATTENTION q head (the indexer has 16 heads -- a {64} tensor cannot
// parametrize it), so the scale rides the MAIN attention q, applied post-rope via the existing
// build_std_attention inp_attn_scale hook (ggml_mul broadcast; FA + non-FA paths both consume
// the scaled q -- the softmax op itself is untouched, CPU+CUDA parity free). Logits are linear
// in q, so scaling q per head == scaling logits per head exactly. Applied on FULL attn layers
// only (sparse_config apply_to_layer_types=[full_attention]; SWA layers carry no indexer).
// IK_SSMAX=1 BREAKS the G1 dense-equivalence exact-match by design (it changes logits at any
// ctx) -- gates: oracle L5 q-scale replay + G2/G3 A/B; IK_SSMAX=0 stays the byte-identical
// regression anchor.
static bool ik_ssmax_enabled() {
    static const bool on = getenv("IK_SSMAX") != nullptr && atoi(getenv("IK_SSMAX")) != 0;
    return on;
}
// n source for beta = s*ln(n): 0 = ln(n_ctx) (DEFAULT -- global context regime, LogN-trick
// precedent: stable across decode so imatrix/G2/G3 see one consistent attention behaviour);
// 1 = ln(padded n_kv) (per-build constant -- pad 32/256 quantizes the drift; counts non-real
// positions at small ctx); 2 = used-span (Nakanishi-literal competition size) DEFERRED: needs
// a per-step graph input (inp_kv_pos precedent) and makes every downstream gate a moving
// target -- revisit only if G3 discriminates.
static int ik_ssmax_n_mode() {
    static const int mode = getenv("IK_SSMAX_N") ? atoi(getenv("IK_SSMAX_N")) : 0;
    GGML_ASSERT(mode >= 0 && mode <= 1 && "IK_SSMAX_N: 0=ln(n_ctx), 1=ln(padded n_kv); 2 (used-span) deferred");
    return mode;
}

void llm_build_context::build_step35_indexer_kv_write(ggml_cgraph * gf, int il, ggml_tensor * inpL) {
    const int64_t head_size = hparams.indexer_head_size;

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return;
    }
    GGML_ASSERT(head_size > 0);

    // Slice 2: real proxy key = k_norm(k_proj(inpL)) over proxy_dim, cached RAW (no RoPE).
    // Ported inline from build_deepseek2_dsa_indexer (build_deepseek2.cpp:417).
    // IK_CSA=2 "z proj" candidate swaps the cached source to RAW z_proj(inpL) (csa_z_norm_type=
    // "none": z output NOT normalized -- deliberately NO k_norm here); whitening below still
    // applies (orthonormal => plumbing-neutral precision win under either knob).
    const auto & layer = model.layers[il];
    ggml_tensor * src;
    if (ik_csa_mode() == 2) {
        GGML_ASSERT(layer.indexer_z && "STEP35 indexer z tensor missing (IK_CSA=2)");
        src = ggml_mul_mat(ctx0, layer.indexer_z, inpL); // {head_size, n_tokens}
    } else {
        GGML_ASSERT(layer.indexer_k && layer.indexer_k_norm && "STEP35 indexer k tensors missing");
        src = ggml_mul_mat(ctx0, layer.indexer_k, inpL); // {head_size, n_tokens}
        src = llm_build_norm(ctx0, src, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, cb, il);
    }
    cb(src, "step35_indexer_k_raw", il);
    GGML_ASSERT(src->ne[0] == head_size && src->ne[1] == n_tokens);

    // IK_HADAMARD whitening BEFORE the cache-type cast: the quantizer sees rotated values --
    // that is where the precision win lives. Row blocks of head_size (power of 2; ggml_hadamard
    // aborts otherwise -- IK_HADAMARD=0 for exotic toy dims).
    if (ik_hadamard_enabled()) {
        GGML_ASSERT(head_size > 1 && (head_size & ~(head_size - 1)) == head_size);
        src = ggml_hadamard(ctx0, src, (int) head_size);
        cb(src, "step35_indexer_k_rot", il);
    }

    src = ggml_cast(ctx0, ggml_cont(ctx0, src), kr_cache->type);
    cb(src, "step35_kr_plumb_src", il);

    ggml_tensor * kr_view = ggml_view_2d(ctx0, kr_cache, head_size, n_tokens,
            ggml_row_size(kr_cache->type, head_size),
            ggml_row_size(kr_cache->type, head_size) * kv_head);
    ggml_tensor * kr_cpy = ggml_cpy(ctx0, src, kr_view);
    cb(kr_cpy, "step35_kr_plumb_write", il);

    // graph-reuse fixup registration (dsa_cache_copies pattern): kr_view bakes kv_head at
    // build time; update_cache_copies() patches view_offs = kv_head*nb[1] on graph reuse.
    GGML_ASSERT((size_t) il < lctx.dsa_cache_copies.size());
    lctx.dsa_cache_copies[il].cpy  = kr_cpy;
    lctx.dsa_cache_copies[il].step = kr_cache->nb[1];

    ggml_build_forward_expand(gf, kr_cpy);
}

// Slice 2: proxy forward (q/z/w proj + q_norm rmsnorm) + indexer-key read-back scoring telemetry.
// Scores = relu(q·k)·w summed over proxy heads, seeded with the causal KQ_mask -- the fork DSA
// precedent (build_deepseek2.cpp:488); Slice 3 swaps in ssmax/block-compress/selection semantics
// behind env knobs. Skipped on SWA layers / MTP tail (write and score guards mirror each other).
// Returns the causal-seeded score {n_kv, n_tokens} (F32) for the IK_SPARSE selection skeleton,
// nullptr on layers without an indexer cache entry.
// Perf note (Slice 3): the cached span is cast to F32 for the pe RoPE; fine at gate contexts,
// revisit (quantized nope matmul + cast only the pe slice) before 1M-ctx decode work.
ggml_tensor * llm_build_context::build_step35_indexer_score(ggml_cgraph * gf, int il,
        ggml_tensor * inpL, ggml_tensor * inp_pos, ggml_tensor * KQ_mask) {
    const int64_t head_size = hparams.indexer_head_size; // proxy_dim (HF sparse_config: 256)
    const int64_t n_ihead   = hparams.indexer_n_head;    // proxy q heads (HF num_heads: 16)

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return nullptr;
    }
    GGML_ASSERT(head_size > 0 && n_ihead > 0);
    GGML_ASSERT(n_kv > 0 && n_kv <= (int64_t) kv_self.size);

    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.indexer_q && layer.indexer_z && layer.indexer_w && layer.indexer_q_norm);

    // Tier-0 oracle tap: the layer input itself (view, no copy -- avoids renaming inpL which
    // may alias an earlier cb()-named tensor). _oracle_step35.py replays the whole indexer
    // chain from this tap + mini weights (build_step35_indexer_inpL).
    GGML_ASSERT(inpL->ne[2] == 1);
    ggml_tensor * inpL_view = ggml_view_2d(ctx0, inpL, inpL->ne[0], inpL->ne[1], inpL->nb[1], 0);
    cb(inpL_view, "step35_indexer_inpL", il);
    ggml_build_forward_expand(gf, inpL_view); // dangling telemetry must ROOT in the graph

    // pe/nope split over indexer_rope_dim (HF sparse_indexer_rope_dim: 32); clamped so toy
    // harness dims (proxy_dim <= rope_dim) rope the whole proxy instead of building empty views.
    const int64_t rope_dim = std::min<int64_t>(
            hparams.indexer_rope_dim > 0 ? hparams.indexer_rope_dim : head_size, head_size);
    const int64_t nope_dim = head_size - rope_dim;

    // Cached-cell positions for the read-back pe RoPE (mirrors inp_dsa_sink: per-graph input,
    // host-filled in llama_set_inputs from kv_self.cells). RAW cached keys + this tensor are what
    // keeps the Slice 1 K-shift invariant: cached rows depend on position only at read-back.
    // The fill runs right before graph compute and ggml_rope_ext reads it DURING compute, so the
    // positions are always fresh; after decode the compute buffer is clobbered (not observable
    // post-hoc -- verified structurally in the harness, values gated in Slice 3).
    if (!lctx.inp_kv_pos) {
        lctx.inp_kv_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
        cb(lctx.inp_kv_pos, "kv_pos", -1);
        ggml_set_input(lctx.inp_kv_pos);
        ggml_build_forward_expand(gf, lctx.inp_kv_pos); // match inp_dsa_sink (build_deepseek2.cpp:1274)
    }

    // ---- proxy q: rmsnorm(q_proj(inpL)) over proxy_dim per head, pe rope at query pos ----
    ggml_tensor * q_cur = ggml_mul_mat(ctx0, layer.indexer_q, inpL); // {head_size*n_ihead, n_tokens}
    q_cur = ggml_view_3d(ctx0, q_cur, head_size, n_ihead, n_tokens,
            ggml_row_size(q_cur->type, head_size),
            ggml_row_size(q_cur->type, head_size) * n_ihead, 0);
    q_cur = llm_build_norm(ctx0, q_cur, hparams, layer.indexer_q_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(q_cur, "step35_indexer_q_normed", il);

    ggml_tensor * q_pe = ggml_view_3d(ctx0, q_cur, rope_dim, n_ihead, n_tokens,
            ggml_row_size(q_cur->type, head_size),
            ggml_row_size(q_cur->type, head_size) * n_ihead, 0);
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, rope_dim,
            rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    if (nope_dim > 0) {
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q_cur, nope_dim, n_ihead, n_tokens,
                ggml_row_size(q_cur->type, head_size),
                ggml_row_size(q_cur->type, head_size) * n_ihead,
                ggml_row_size(q_cur->type, rope_dim));
        q_cur = ggml_concat(ctx0, q_pe, q_nope, 0);
    } else {
        q_cur = q_pe;
    }
    cb(q_cur, "step35_indexer_q_cat", il);

    // ---- z/w projections (roles land in Slice 3: z ↔ csa_block_compress (z_norm_type=none),
    // w ≡ fork DSA per-head indexer weights) ----
    ggml_tensor * z_cur = ggml_mul_mat(ctx0, layer.indexer_z, inpL); // {head_size, n_tokens}
    cb(z_cur, "step35_indexer_z_raw", il);
    ggml_build_forward_expand(gf, z_cur); // z is telemetry-only: unconsumed cb() tensors never
                                          // enter the graph and cb_eval never fires (inp_dsa_sink
                                          // precedent) -- root it explicitly

    ggml_tensor * w_cur = ggml_mul_mat(ctx0, layer.indexer_w, inpL); // {n_ihead, n_tokens}
    w_cur = ggml_scale(ctx0, w_cur, 1.0f / sqrtf(float(head_size * n_ihead))); // DSA :488 port
    cb(w_cur, "step35_indexer_w", il);

    // ---- read back the cached RAW proxy keys ({head_size, n_kv}), rope pe at read-back ----
    ggml_tensor * cached_k = ggml_view_2d(ctx0, kr_cache, head_size, n_kv,
            ggml_row_size(kr_cache->type, head_size), 0);
    // IK_HADAMARD de-rotation (H^2 == I; ggml_hadamard outputs F32 so it slots in for the cast).
    // Result = the raw proxy key modulo cache quantization noise -> pe RoPE below sees the same
    // raw-domain key as IK_HADAMARD=0.
    ggml_tensor * k_f32;
    if (ik_hadamard_enabled()) {
        k_f32 = ggml_hadamard(ctx0, cached_k, (int) head_size);
        cb(k_f32, "step35_indexer_k_derot", il);
    } else {
        k_f32 = ggml_cast(ctx0, ggml_cont(ctx0, cached_k), GGML_TYPE_F32);
    }
    cb(k_f32, "step35_indexer_cached_k", il);

    ggml_tensor * k_pe = ggml_view_3d(ctx0, k_f32, rope_dim, 1, n_kv,
            ggml_row_size(k_f32->type, head_size),
            ggml_row_size(k_f32->type, head_size), 0);
    k_pe = ggml_rope_ext(ctx0, k_pe, lctx.inp_kv_pos, nullptr, rope_dim,
            rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_tensor * indexer_k_b;
    if (nope_dim > 0) {
        ggml_tensor * k_nope = ggml_view_3d(ctx0, k_f32, nope_dim, 1, n_kv,
                ggml_row_size(k_f32->type, head_size),
                ggml_row_size(k_f32->type, head_size),
                ggml_row_size(k_f32->type, rope_dim));
        indexer_k_b = ggml_concat(ctx0, k_pe, k_nope, 0);
    } else {
        indexer_k_b = k_pe;
    }
    indexer_k_b = ggml_reshape_3d(ctx0, indexer_k_b, head_size, n_kv, 1);
    cb(indexer_k_b, "step35_indexer_k_rope", il);

    // ---- scores: relu(q·k)·w summed over proxy heads (DSA :488 port) ----
    // IK_CSA=0 scores TOKEN keys (n_kv); IK_CSA>0 scores pooled BLOCK keys (n_kv/B) and
    // broadcasts per-block scores back to slots. The causal mask seed is added LAST at slot
    // granularity in both cases. Non-inplace add on purpose: the F32 mask seed aliases
    // inp_KQ_mask (-fa 0), which later softmax layers read back (build_deepseek2.cpp:540 caveat).
    ggml_tensor * mask_f32 = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
    if (mask_f32->type != GGML_TYPE_F32) {
        mask_f32 = ggml_cast(ctx0, mask_f32, GGML_TYPE_F32);
        cb(mask_f32, "step35_indexer_score_mask_f32", il);
    }

    // scoring chain reused VERBATIM for token keys or pooled block keys (n_key = n_kv / n_blocks)
    auto score_chain = [&](ggml_tensor * keys, int64_t n_key) {
        ggml_tensor * q2d = ggml_reshape_2d(ctx0, q_cur, head_size, n_ihead * n_tokens);
        ggml_tensor * kq = ggml_mul_mat(ctx0, keys, q2d); // {n_key, n_ihead*n_tokens}
        cb(kq, "step35_indexer_kq", il);
        kq = ggml_relu(ctx0, kq);
        cb(kq, "step35_indexer_kq_relu", il);
        kq = ggml_reshape_3d(ctx0, kq, n_key, n_ihead, n_tokens);
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, kq)); // {n_ihead, n_key, n_tokens}
        ggml_tensor * w3 = ggml_reshape_3d(ctx0, w_cur, n_ihead, 1, n_tokens);
        kq = ggml_mul(ctx0, kq, w3);
        cb(kq, "step35_indexer_kq_w", il);
        ggml_tensor * s = ggml_sum_rows(ctx0, kq); // {1, n_key, n_tokens}
        return ggml_reshape_2d(ctx0, s, n_key, n_tokens);
    };

    ggml_tensor * slot_score;
    const int64_t B = hparams.indexer_csa_block > 0 ? hparams.indexer_csa_block : 8;
    if (ik_csa_mode() == 0 && !ik_sel_block_grain()) {
        slot_score = score_chain(ggml_reshape_2d(ctx0, indexer_k_b, head_size, n_kv), n_kv);
    } else {
        // kv padding (32 non-FA / 256 FA) guarantees n_kv % B == 0 for B=8 -- slot-aligned reshape
        // pooling needs contiguous b*B..b*B+B-1 groups (== position-aligned blocks in steady
        // single-sequence decode; multi-sequence/defrag alignment TODO).
        GGML_ASSERT(B > 0 && n_kv % B == 0 && "CSA slot-aligned blocks need n_kv divisible by B");
        const int64_t n_blocks = n_kv / B;

        // validity weights v[slot] = 1 iff visible to SOME query of this ubatch: step(mask+0.5)
        // per (slot,query) turns {0,-inf} into {1,0}, summed over query columns and re-stepped.
        // Empty padded cells carry allocator garbage kr_l rows -- zero-weighted OUT of the pool
        // so their garbage cannot poison block means (and 0*garbage stays finite garbage-free 0).
        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, mask_f32, 1.0f, 0.5f));
        vis = ggml_sum_rows_ext(ctx0, vis, 1);                       // {n_kv, 1}
        vis = ggml_step(ctx0, ggml_scale_bias(ctx0, vis, 1.0f, -0.5f));
        vis = ggml_reshape_2d(ctx0, vis, n_kv, 1);
        cb(vis, "step35_indexer_slot_valid", il);
        ggml_tensor * v3 = ggml_reshape_3d(ctx0, vis, 1, B, n_blocks);

        // pooling = reshape+sum composition (csa_ssmax.md verdict (d): ggml_pool_1d is CPU-only
        // in this fork; ggml_sum_rows_ext(dim) has CPU+CUDA parity). slot-aligned reshape
        // {head, B, n_blocks}: slot b*B+r maps to [*, r, b]. Validity-weighted mean divides by
        // |Omega_b| (doc prose partial-block rule); den clamp min 1 keeps all-empty blocks finite
        // (zero key -> zero score -> swallowed by the causal add below).
        ggml_tensor * k3 = ggml_reshape_3d(ctx0,
                ggml_reshape_2d(ctx0, indexer_k_b, head_size, n_kv), head_size, B, n_blocks);
        ggml_tensor * num = ggml_sum_rows_ext(ctx0, ggml_mul(ctx0, k3, v3), 1); // {head, 1, n_blocks}
        num = ggml_reshape_2d(ctx0, num, head_size, n_blocks);
        ggml_tensor * den = ggml_sum_rows_ext(ctx0, v3, 1);                      // {1, 1, n_blocks}
        den = ggml_clamp(ctx0, ggml_reshape_2d(ctx0, den, 1, n_blocks), 1.0f, (float) B);
        ggml_tensor * block_k = ggml_div(ctx0, num, den); // broadcast {head,n_blocks}/{1,n_blocks}
        cb(block_k, "step35_indexer_block_k", il);

        ggml_tensor * block_score = score_chain(block_k, n_blocks); // {n_blocks, n_tokens}
        cb(block_score, "step35_indexer_block_score", il);

        // broadcast to slots: slot b*B+r takes block b's score. repeat alone does it: rep
        // {B,n_blocks,n_tokens} with rep[r,b,j] = block_score[b,j] flattens col-major to
        // linear r + B*b (+ B*n_blocks*j) -- reshaped {n_kv,n_tokens} that is exactly slot
        // i = b*B+r <- block b (the inverse of the pooling reshape's slot map). A permute
        // here TRANSPOSES the map (slot i <- block i%n_blocks -- bitwise-verified against
        // the T9 score tap via the oracle L1 broadcast check); it must NOT run.
        ggml_tensor * bs3 = ggml_reshape_3d(ctx0, block_score, 1, n_blocks, n_tokens);
        ggml_tensor * rep = ggml_repeat(ctx0, bs3,
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, B, n_blocks, n_tokens));
        slot_score = ggml_reshape_2d(ctx0, ggml_cont(ctx0, rep), n_kv, n_tokens);
    }

    ggml_tensor * indexer_score = ggml_add(ctx0, mask_f32, slot_score);
    cb(indexer_score, "step35_indexer_score", il);
    ggml_build_forward_expand(gf, indexer_score);
    return indexer_score;
}

ggml_cgraph * llm_build_context::build_step35() {
    ggml_cgraph * gf = new_graph_custom();
    ggml_tensor * cur;
    auto inp_pos     = build_inp_pos();

    // indexer-write registrations are per-graph: stale entries from a previous build must not
    // be patched against a graph that no longer contains them (build_openpangu.cpp precedent).
    std::fill(lctx.dsa_cache_copies.begin(), lctx.dsa_cache_copies.end(), llama_context::CacheCopy{});

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        GGML_ASSERT(model.mtp && hparams.nextn_predict_layers > 0);
        GGML_ASSERT(batch.token && "Step35 MTP requires token batches");

        const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
            ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;
        const int n_heads_model = (int) hparams.nextn_predict_layers;
        const int n_heads = lctx.mtp_n_heads > 0
            ? std::max(1, std::min((int) lctx.mtp_n_heads, n_heads_model)) : n_heads_model;
        const int step = std::max(0, std::min((int) lctx.mtp_step_idx, n_heads - 1));
        const int il = n_layer_base + step;

        ggml_tensor * hidden_states = build_inp_mtp_states(n_embd);

        const bool step_independent_warmup = model.arch == LLM_ARCH_STEP35 &&
            (cparams.mtp_op_type == MTP_OP_WARMUP ||
             cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) && n_heads > 1;
        if (step_independent_warmup) {
            for (int i = n_heads - 1; i >= 0; --i) {
                const int head_il = n_layer_base + i;
                const bool is_first = i == 0;
                const bool emit_logits = is_first && cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
                cur = build_step35_mtp(model.layers[head_il], hidden_states, gf, inp_pos,
                        /*reduce_output=*/is_first, emit_logits);
                ggml_build_forward_expand(gf, cur);
            }
            return gf;
        }

        const bool reduce_mtp_output = cparams.mtp_op_type != MTP_OP_NONE;
        const bool emit_mtp_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
            cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
        cur = build_step35_mtp(model.layers[il], hidden_states, gf, inp_pos,
                reduce_mtp_output, emit_mtp_logits);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();
    auto KQ_mask_swa = build_inp_KQ_mask_swa();
    //const float kq_scale = 1.0f / sqrtf(float(n_rot));
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
        ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;

    for (int il = 0; il < n_layer_base; ++il) {
        bool is_swa = hparams.swa_layers[il];
        auto & layer = const_cast<llama_layer&>(model.layers[il]);

        ggml_tensor * rope_factors = nullptr;
        const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
        if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
            rope_factors = build_rope_factors(il);
        }
        auto rope_freqs = layer.rope_freqs;
        layer.rope_freqs = nullptr;
        // Slice 1/2 plumbing: proxy-key cache write (real proxy key: k_norm(k_proj), RAW) +
        // read-back scoring telemetry (semantic selection core deferred to Slice 3 knobs).
        build_step35_indexer_kv_write(gf, il, inpL);
        ggml_tensor * idx_score = build_step35_indexer_score(gf, il, inpL, inp_pos, KQ_mask);

        // ---- Stage B IK_SPARSE selection skeleton (mask-based sparse attention FIRST) ----
        // Reuses the arch-neutral DSA members verbatim: full descending argsort over the n_kv axis
        // -> rank-penalty scatter mask (build_deepseek2_dsa_sparse_mask; the --dsa-top-k/-dsatk
        // kept-key-count characterization knob applies here too) -> FA adapter (…_dsa_fa_mask)
        // when -fa 1 needs the F16 padded contiguous mask shape. ALWAYS built under IK_SPARSE=1,
        // deliberately NO n_kv <= topk short-circuit: at ctx <= topk the clamp makes pen(rank)==0
        // for every key, so the mask == the causal mask BITWISE and logits == IK_SPARSE=0 dense
        // EXACTLY -- the G1 dense-equivalence oracle therefore exercises the whole argsort/scatter
        // path instead of bypassing it. Cost at ctx <= topk is trivial; gather/fused-indexer
        // bandwidth wins stay on the 1M-ctx decode TODO list (AGENTS.md strategy).
        ggml_tensor * attn_mask = is_swa ? KQ_mask_swa : KQ_mask;
        if (!is_swa && ik_sparse_enabled() && idx_score) {
            ggml_tensor * sorted = ggml_argsort(ctx0, idx_score, GGML_SORT_ORDER_DESC);
            cb(sorted, "step35_indexer_sorted", il);
            // Block-grain selection: topk counts BLOCKS (sparse_config topk=512 x region_block_size=8
            // = 4096 kept slots). Kept-count must be a multiple of B so the rank cutoff snaps to
            // whole blocks (intra-block broadcast scores tie contiguously in the argsort).
            const int64_t Bsel = hparams.indexer_csa_block > 0 ? hparams.indexer_csa_block : 8;
            const int64_t kept_slots = ik_sel_block_grain()
                ? (int64_t) hparams.indexer_top_k * Bsel
                : (int64_t) hparams.indexer_top_k;
            ggml_tensor * sparse = build_deepseek2_dsa_sparse_mask(sorted, KQ_mask, kept_slots);
            cb(sparse, "step35_sparse_mask", il); // rename tap for step35 G4 telemetry
            if (flash_attn) {
                sparse = build_deepseek2_dsa_fa_mask(sparse, KQ_mask);
                cb(sparse, "step35_sparse_mask_fa", il);
            }
            attn_mask = sparse;
        }
        // IK_SSMAX per-q-head logit scale beta_h = s_h * ln(n): ssmax_s {n_head} reshaped to
        // {1, n_head, 1} broadcasts along Qcur's head axis (Qcur is {hd, n_head, n_tokens} at
        // the inp_attn_scale mul site) -- scale * ln(n) folded into one ggml_scale on the
        // weight view, no graph input, no new ops. Consumed by build_std_attention's existing
        // `Qcur = ggml_mul(Qcur, inp_attn_scale)` (post-rope: rope is orthogonal per pair so a
        // per-head scalar commutes -- scaling q post-rope == scaling logits).
        ggml_tensor * ssmax_scale = nullptr;
        if (!is_swa && ik_ssmax_enabled()) {
            const auto & slayer = model.layers[il];
            GGML_ASSERT(slayer.indexer_ssmax_s && "IK_SSMAX=1 needs blk.{i}.indexer.ssmax_s");
            GGML_ASSERT(slayer.indexer_ssmax_s->ne[0] == hparams.n_head(il));
            const float n_src = ik_ssmax_n_mode() == 0 ? float(cparams.n_ctx) : float(n_kv);
            const float beta  = logf(n_src);
            ggml_tensor * s3 = ggml_reshape_3d(ctx0, slayer.indexer_ssmax_s, 1, hparams.n_head(il), 1);
            // ggml_cont: the scale chain derives from a MODEL WEIGHT (mmap/split buffer) -- without
            // the copy the tap's data pointer is not compute-buffer storage, and the harness
            // cb_eval grab segfaulted reading it (first tap ever captured outside the compute
            // buffer). The cont also gives the downstream mul a contiguous src1.
            ssmax_scale = ggml_cont(ctx0, ggml_scale(ctx0, s3, beta));
            cb(ssmax_scale, "step35_ssmax_qscale", il);
        }
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer_base - 1 && n_tokens > 1 && !cparams.mtp ? inp_out_ids : nullptr,
                rope_factors, attn_mask, nullptr, ssmax_scale, kq_scale, 0.0f, is_swa ? hparams.n_swa : 0,
                il, true, false, true);
        layer.rope_freqs = rope_freqs;

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense FFN
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            const bool  norm_w  = hparams.expert_weights_norm;
            const float w_scale = hparams.expert_weights_scale;
            const bool  scale_w = w_scale != 0.0f;
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, norm_w, scale_w, w_scale,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    //(llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    if (cparams.mtp) {
        ggml_tensor * mtp_embd = inpL->type == GGML_TYPE_F32 ? inpL : ggml_cast(ctx0, inpL, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);

        if (inp_out_ids) {
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * llm_build_context::build_step35_mtp(
        const llama_layer & mtp_layer,
        ggml_tensor * hidden_states_from_main_model,
        ggml_cgraph * gf,
        ggml_tensor * inp_pos,
        bool reduce_output,
        bool emit_logits,
        ggml_tensor ** hidden_out) {
    const int il = (int) (&mtp_layer - model.layers.data());

    GGML_ASSERT(mtp_layer.nextn.eh_proj && mtp_layer.nextn.enorm && mtp_layer.nextn.hnorm);
    GGML_ASSERT(mtp_layer.wq && mtp_layer.wk && mtp_layer.wv && mtp_layer.wo);

    ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;
    ggml_tensor * tok_embd_w = mtp_layer.nextn.embed_tokens ? mtp_layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = build_inp_embd_mtp(tok_embd_w);
    ggml_tensor * cur = build_mtp_input(mtp_layer, hidden_states_from_main_model,
            tok_embd, il, "mtp_eh_proj");

    const bool is_swa = hparams.swa_layers[il];
    ggml_tensor * rope_factors = nullptr;
    const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
    if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
        rope_factors = build_rope_factors(il);
    }
    auto KQ_mask = is_swa ? build_inp_KQ_mask_swa() : build_inp_KQ_mask();
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k(il)));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur, inp_pos, nullptr,
            rope_factors, KQ_mask, nullptr, nullptr, kq_scale, 0.0f,
            is_swa ? hparams.n_swa : 0, il, true, false, true, false, false, nullptr, il);

    if (mtp_layer.ffn_gate_inp == nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up, nullptr, nullptr,
                mtp_layer.ffn_gate, nullptr, nullptr,
                mtp_layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
    } else {
        cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_gate_inp, nullptr,
                mtp_layer.ffn_up_exps, nullptr,
                mtp_layer.ffn_gate_exps, nullptr,
                mtp_layer.ffn_down_exps, nullptr,
                mtp_layer.ffn_exp_probs_b,
                mtp_layer.ffn_up_shexp, nullptr,
                mtp_layer.ffn_gate_shexp, nullptr,
                mtp_layer.ffn_down_shexp, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_post_ffn", il);
    if (hidden_out) {
        *hidden_out = cur;
    }

    ggml_tensor * output_hidden = cur;
    if (reduce_output) {
        if (cparams.mtp_op_type != MTP_OP_NONE && n_tokens > 1) {
            output_hidden = ggml_view_2d(ctx0, cur, n_embd, 1,
                    cur->nb[1], (size_t) (n_tokens - 1) * cur->nb[1]);
        } else if (inp_out_ids) {
            output_hidden = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }
    if (reduce_output) {
        ggml_tensor * mtp_embd = output_hidden->type == GGML_TYPE_F32 ? output_hidden : ggml_cast(ctx0, output_hidden, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);
    }

    if (!emit_logits) {
        return output_hidden;
    }

    ggml_tensor * head_norm = mtp_layer.nextn.shared_head_norm
        ? mtp_layer.nextn.shared_head_norm : model.output_norm;
    ggml_tensor * head = mtp_layer.nextn.shared_head_head
        ? mtp_layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_norm && head);
    cur = llm_build_context::build_output(lctx, ctx0, output_hidden, head, head_norm, cb);
    cb(cur, "result_output", -1);
    return cur;
}
