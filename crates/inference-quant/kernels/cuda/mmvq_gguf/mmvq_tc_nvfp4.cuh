// NVFP4's tensor-core MoE matvec (mmvq_tc_moe.cuh over nvfp4_tiled.rs's tiles) and its untiling for gathers.
#pragma once
#include "mmvq_tc_moe.cuh"

MMVQ_TC_MOE_LAUNCHER(nvfp4, TcNvfp4)

// Rows `ids` of a tiled NVFP4 stack (row id = expert * rows + row, rows a multiple of 16) back as GGUF's row-major
// 36-byte blocks: a thread per output block.
static __global__ void nvfp4_untile_kernel(const uint8_t *__restrict__ src, const uint32_t *__restrict__ ids,
                                           uint8_t *__restrict__ dst, const int64_t n, const int nb) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * nb) {
    return;
  }
  const int64_t row = idx / nb;
  const int kb = (int)(idx % nb);
  const uint32_t id = ids[row];
  const int r = id % MMVQ_TC_MOE_ROWS;
  const int e = r % 8;
  const int upper = r / 8;
  const uint8_t *tile = src + ((int64_t)(id / MMVQ_TC_MOE_ROWS) * nb + kb) * TcNvfp4::TILE_BYTES;
  uint8_t *out = dst + (row * nb + kb) * (TcNvfp4::TILE_BYTES / MMVQ_TC_MOE_ROWS);
#pragma unroll
  for (int s = 0; s < TcNvfp4::SUBS; ++s) {
    out[s] = tile[e * 8 + upper * 4 + s];
  }
  // value v of sub-block s sits in lane (e, v / 4)'s word s, byte v % 4; GGUF keeps it at byte 8 s + v % 8
  auto nibble = [&](const int s, const int v) {
    const uint8_t byte = tile[64 + (4 * e + v / 4) * 16 + 4 * s + v % 4];
    return upper ? byte >> 4 : byte & 0x0F;
  };
#pragma unroll
  for (int s = 0; s < TcNvfp4::SUBS; ++s) {
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      out[TcNvfp4::SUBS + 8 * s + b] = (uint8_t)(nibble(s, b) | nibble(s, b + 8) << 4);
    }
  }
}

extern "C" void launch_nvfp4_untile_rows(const void *src, const void *ids, void *dst, int64_t n, int64_t ncols,
                                         void *stream) {
  constexpr int threads = 256;
  const int nb = (int)(ncols / QK_NVFP4);
  const int64_t total = n * nb;
  const int blocks = (int)((total + threads - 1) / threads);
  nvfp4_untile_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      (const uint8_t *)src, (const uint32_t *)ids, (uint8_t *)dst, n, nb);
}
