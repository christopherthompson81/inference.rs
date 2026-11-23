#include "mmvq_tc_k.cuh"
#include "mmvq_tc_moe.cuh"

MMVQ_PLAIN_BATCH_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                     vec_dot_q6_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)
MMVQ_LAUNCHER_PLAIN(q6_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q6_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q6_k, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q6_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q6_k)
MMVQ_TC_K_LAUNCHERS(q6_k, TckQ6K)
// Rows `ids` of a tiled Q6_K stack back as GGUF's row-major 210-byte blocks: a thread per output block, each value's
// six bits moved from its lane's word and top-bit byte to `ql` and `qh`.
static __global__ void q6_k_untile_kernel(const uint8_t *__restrict__ src, const uint32_t *__restrict__ ids,
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
  const uint8_t *tile = src + ((int64_t)(id / MMVQ_TC_MOE_ROWS) * nb + kb) * TcQ6K::TILE_BYTES;
  uint8_t *out = dst + (row * nb + kb) * (TcQ6K::TILE_BYTES / MMVQ_TC_MOE_ROWS);
  for (int b = 0; b < 192; ++b) {
    out[b] = 0;
  }
  for (int v = 0; v < QK_K; ++v) {
    const int h = v / 128, i = v % 128 / 16, t = v % 16 / 4, j = v % 4;
    const uint8_t *lane = tile + 288 + h * 1536 + (4 * e + t) * 48;
    const int low = (lane[4 * i + j] >> (4 * upper)) & 0x0F;
    const int high = (lane[32 + 2 * i + upper] >> (2 * j)) & 3;
    const int n2 = v / 128, q = v % 128 / 32, l = v % 32;
    out[64 * n2 + l + 32 * (q % 2)] |= (uint8_t)(low << (q < 2 ? 0 : 4));
    out[128 + 32 * n2 + l] |= (uint8_t)(high << (2 * q));
  }
  for (int h = 0; h < 2; ++h)
    for (int i = 0; i < 8; ++i) {
      out[192 + 8 * h + i] = tile[32 + e * 32 + h * 16 + upper * 8 + i];
    }
  out[208] = tile[e * 4 + upper * 2];
  out[209] = tile[e * 4 + upper * 2 + 1];
}

MMVQ_TC_MOE_LAUNCHER(q6_k, TcQ6K)
MMVQ_TC_UNTILE_LAUNCHER(q6_k, q6_k_untile_kernel, QK_K)
