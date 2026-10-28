//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "common.cuh"

void ggml_cuda_op_indexer_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gather_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// ggml_gsa_kv_supported moved to ggml.h/ggml.c (builders need it too).
