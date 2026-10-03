// Copies whole rows of a byte matrix by index: embedding lookups on GGUF weights kept as raw blocks.
// An id past the last row yields a zeroed row rather than an out-of-bounds read.
#include <stdint.h>

#define GATHER_THREADS 256

static __global__ void gather_rows_u8(const uint8_t *__restrict__ src, const uint32_t *__restrict__ ids,
                                      uint8_t *__restrict__ dst, int64_t row_bytes, int64_t rows) {
  const int64_t id = ids[blockIdx.x];
  uint8_t *out = dst + (int64_t)blockIdx.x * row_bytes;
  const uint8_t *row = src + id * row_bytes;
  for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
    out[i] = id < rows ? row[i] : 0;
  }
}

extern "C" void launch_gather_rows_u8(const void *src, const void *ids, void *dst, int64_t row_bytes, int64_t rows,
                                      int64_t n, void *stream) {
  gather_rows_u8<<<(unsigned int)n, GATHER_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
      (const uint8_t *)src, (const uint32_t *)ids, (uint8_t *)dst, row_bytes, rows);
}
