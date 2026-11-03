#pragma once
#include "common.cuh"

static __device__ inline float gsa_dequant(ggml_type type, const char * row, int i) {
    switch (type) {
        case GGML_TYPE_F16:
            return __half2float(*(const half *)(row + i * sizeof(half)));
        case GGML_TYPE_F32:
            return *(const float *)(row + i * sizeof(float));
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * b = (const block_q8_0 *)(row + (i / QK8_0) * sizeof(block_q8_0));
            return __half2float(b->d) * (float)b->qs[i % QK8_0];
        }
        case GGML_TYPE_Q6_0: {
            const block_q6_0 * b = (const block_q6_0 *)(row + (i / QK6_0) * sizeof(block_q6_0));
            const int j = i % QK6_0;
            if (j < QK6_0 / 2) {
                const uint8_t h = b->qh[j % (QK6_0 / 4)] >> (4 * (j / (QK6_0 / 4)));
                return __half2float(b->d) * (float)(((b->qs[j] & 0x0F) | ((h << 4) & 0x30)) - 32);
            }
            const int j2 = j - QK6_0 / 2;
            const uint8_t h = b->qh[j2 % (QK6_0 / 4)] >> (4 * (j2 / (QK6_0 / 4)));
            return __half2float(b->d) * (float)(((b->qs[j2] >> 4) | ((h << 2) & 0x30)) - 32);
        }
        case GGML_TYPE_Q5_0: {
            const block_q5_0 * b = (const block_q5_0 *)(row + (i / QK5_0) * sizeof(block_q5_0));
            const int j = i % QK5_0;
            const int xh = ((b->qh[j >> 3] >> (j & 7)) & 1) << 4;
            const int xl = (j < QK5_0 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK5_0 / 2] >> 4);
            return __half2float(b->d) * (float)((xl | xh) - 16);
        }
        case GGML_TYPE_Q5_1: {
            const block_q5_1 * b = (const block_q5_1 *)(row + (i / QK5_1) * sizeof(block_q5_1));
            const int j = i % QK5_1;
            const int xh = ((b->qh[j >> 3] >> (j & 7)) & 1) << 4;
            const int xl = (j < QK5_1 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK5_1 / 2] >> 4);
            return __half2float(b->data.d) * (float)(xl | xh) + __half2float(b->data.m);
        }
        case GGML_TYPE_Q4_0: {
            const block_q4_0 * b = (const block_q4_0 *)(row + (i / QK4_0) * sizeof(block_q4_0));
            const int j = i % QK4_0;
            const int q = (j < QK4_0 / 2 ? (b->qs[j] & 0x0F) : (b->qs[j - QK4_0 / 2] >> 4)) - 8;
            return __half2float(b->d) * (float)q;
        }
        case GGML_TYPE_Q4_1: {
            const block_q4_1 * b = (const block_q4_1 *)(row + (i / QK4_1) * sizeof(block_q4_1));
            const int j = i % QK4_1;
            const int q = (j < QK4_1 / 2) ? (b->qs[j] & 0x0F) : (b->qs[j - QK4_1 / 2] >> 4);
            return __half2float(b->data.d) * (float)q + __half2float(b->data.m);
        }
        default:
            return 0.0f;
    }
}

