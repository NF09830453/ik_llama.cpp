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

void llm_build_context::build_step35_indexer_kv_write(ggml_cgraph * gf, int il, ggml_tensor * inpL) {
    const int64_t head_size = hparams.indexer_head_size;

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return;
    }
    GGML_ASSERT(head_size > 0);

    // Slice 2: real proxy key = k_norm(k_proj(inpL)) over proxy_dim, cached RAW (no RoPE).
    // Ported inline from build_deepseek2_dsa_indexer (build_deepseek2.cpp:417).
    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.indexer_k && layer.indexer_k_norm && "STEP35 indexer k tensors missing");
    ggml_tensor * src = ggml_mul_mat(ctx0, layer.indexer_k, inpL); // {head_size, n_tokens}
    src = llm_build_norm(ctx0, src, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, cb, il);
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
// Perf note (Slice 3): the cached span is cast to F32 for the pe RoPE; fine at gate contexts,
// revisit (quantized nope matmul + cast only the pe slice) before 1M-ctx decode work.
void llm_build_context::build_step35_indexer_score(ggml_cgraph * gf, int il,
        ggml_tensor * inpL, ggml_tensor * inp_pos, ggml_tensor * KQ_mask) {
    const int64_t head_size = hparams.indexer_head_size; // proxy_dim (HF sparse_config: 256)
    const int64_t n_ihead   = hparams.indexer_n_head;    // proxy q heads (HF num_heads: 16)

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return;
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

    // ---- scores: relu(q·k)·w summed over proxy heads, seeded with the causal mask ----
    // Non-inplace add on purpose: the F32 mask seeds aliases inp_KQ_mask (-fa 0), which later
    // softmax layers read back (build_deepseek2.cpp:540 caveat).
    ggml_tensor * indexer_score = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
    if (indexer_score->type != GGML_TYPE_F32) {
        indexer_score = ggml_cast(ctx0, indexer_score, GGML_TYPE_F32);
        cb(indexer_score, "step35_indexer_score_mask_f32", il);
    }

    ggml_tensor * q2d = ggml_reshape_2d(ctx0, q_cur, head_size, n_ihead * n_tokens);
    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k_b, q2d); // {n_kv, n_ihead*n_tokens}
    cb(indexer_kq, "step35_indexer_kq", il);
    indexer_kq = ggml_relu(ctx0, indexer_kq);
    cb(indexer_kq, "step35_indexer_kq_relu", il);
    indexer_kq = ggml_reshape_3d(ctx0, indexer_kq, n_kv, n_ihead, n_tokens);
    indexer_kq = ggml_cont(ctx0, ggml_transpose(ctx0, indexer_kq)); // {n_ihead, n_kv, n_tokens}
    ggml_tensor * w3 = ggml_reshape_3d(ctx0, w_cur, n_ihead, 1, n_tokens);
    indexer_kq = ggml_mul(ctx0, indexer_kq, w3);
    cb(indexer_kq, "step35_indexer_kq_w", il);
    ggml_tensor * score = ggml_sum_rows(ctx0, indexer_kq); // {1, n_kv, n_tokens}
    score = ggml_reshape_2d(ctx0, score, n_kv, n_tokens);
    indexer_score = ggml_add(ctx0, indexer_score, score);
    cb(indexer_score, "step35_indexer_score", il);
    ggml_build_forward_expand(gf, indexer_score);
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
        build_step35_indexer_score(gf, il, inpL, inp_pos, KQ_mask);
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer_base - 1 && n_tokens > 1 && !cparams.mtp ? inp_out_ids : nullptr,
                rope_factors, is_swa ? KQ_mask_swa : KQ_mask, nullptr, nullptr, kq_scale, 0.0f, is_swa ? hparams.n_swa : 0,
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
