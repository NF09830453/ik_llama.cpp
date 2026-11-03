#pragma once
#include "common.cuh"

// Returns false without launching when the existing gather fallback is required.
bool ggml_cuda_gather_sparse_attn_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
