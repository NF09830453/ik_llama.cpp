//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "common.cuh"

#include <cstdio>
#include <cstdlib>

void ggml_cuda_op_indexer_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gather_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GSA_TIMING=1: per-op wall-clock accumulator (true GPU time under CUDA_LAUNCH_BLOCKING=1).
// Prints every 45 calls (one decode step across the full-attn layers). No-op when unset.
static inline void gsa_time_log(const char * name, int64_t t0_ms, int n_kv) {
    static const int kPeriod = 45;
    static double acc[8] = {};
    static int    cnt[8] = {};
    static const char * names[8] = {};
    static int    nn = 0;
    if (getenv("GSA_TIMING") == nullptr) return;
    double ms = (double)(ggml_time_us() - t0_ms) / 1000.0;
    int slot = -1;
    for (int i = 0; i < nn; ++i) if (names[i] == name) { slot = i; break; }
    if (slot < 0 && nn < 8) { slot = nn++; names[slot] = name; }
    if (slot < 0) return;
    acc[slot] += ms;
    if (cnt[slot] < 2) {
        fprintf(stderr, "[GSA_TIME] %-28s n_kv=%6d  first-calls %.3f ms (call #%d)\n", name, n_kv, ms, cnt[slot] + 1);
    }
    if (++cnt[slot] == kPeriod) {
        fprintf(stderr, "[GSA_TIME] %-28s n_kv=%6d  %8.3f ms/call\n", name, n_kv, acc[slot] / cnt[slot]);
        acc[slot] = 0; cnt[slot] = 0;
    }
}

// ggml_gsa_kv_supported moved to ggml.h/ggml.c (builders need it too).
