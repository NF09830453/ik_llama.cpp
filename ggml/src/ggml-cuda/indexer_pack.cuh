//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "common.cuh"

void ggml_cuda_op_indexer_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gather_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Supported K/V cache types for the gather_sparse_attn kernel. One list, consumed three
// ways: the CUDA dispatch (gather_sparse_attn.cu expands the full K x V cross product),
// the CUDA supports_op gate, and the builder-side fallback guard. Adding a KV type =
// one accessor struct in gather_sparse_attn.cu + one line here.
#define GSA_TYPES(X)                                                         \
    X(GGML_TYPE_F16,  gsa_f16)                                               \
    X(GGML_TYPE_F32,  gsa_f32)                                               \
    X(GGML_TYPE_Q8_0, gsa_q8_0)                                              \
    X(GGML_TYPE_Q6_0, gsa_q6_0)                                              \
    X(GGML_TYPE_Q5_0, gsa_q5_0)                                              \
    X(GGML_TYPE_Q5_1, gsa_q5_1)                                              \
    X(GGML_TYPE_Q4_0, gsa_q4_0)                                              \
    X(GGML_TYPE_Q4_1, gsa_q4_1)

inline bool ggml_gsa_kv_supported(ggml_type kt, ggml_type vt) {
    // must stay in sync with GSA_TYPES (the dispatch cross product in gather_sparse_attn.cu)
    auto v_ok = [](ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_F32 || t == GGML_TYPE_Q8_0 ||
               t == GGML_TYPE_Q6_0 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 ||
               t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1;
    };
    return v_ok(kt) && v_ok(vt);
}
