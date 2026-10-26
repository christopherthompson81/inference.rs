// The doorbell an offloaded MoE layer's decode step rings for the CPU: the routes and activations go to mapped host
// memory and a sequence number tells a spinning host thread they are there; the stream later spins on the thread's
// reply. Both run in a captured graph, where a host function would cost a launch per layer.

#include <cuda_runtime.h>
#include <stdint.h>

#define DOORBELL_THREADS 256
// a reply this late means the host thread is gone: the step adds nothing rather than spinning forever
#define DOORBELL_TIMEOUT_NS 2000000000ull

__device__ __forceinline__ uint64_t doorbell_clock() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Publishes the next sequence number once the step's picks are in the mapped exchange (written by the preparation
// kernel before it on the stream)
__global__ void moe_doorbell_ring_kernel(uint32_t *counter, volatile uint32_t *bell, int tokens) {
  __threadfence_system();
  const uint32_t seq = *counter + 1;
  *counter = seq;
  // the token count, then the sequence the thread waits for
  bell[1] = static_cast<uint32_t>(tokens);
  __threadfence_system();
  bell[0] = seq;
}

// Spins until the host thread has answered this layer's latest ring, then copies its rows to the device
__global__ void moe_doorbell_wait_kernel(const uint32_t *__restrict__ counter, volatile const uint32_t *done,
                                         const float *host_ys, float *__restrict__ ys, int values,
                                         uint32_t *timed_out) {
  __shared__ int ok;
  if (threadIdx.x == 0) {
    const uint32_t seq = *counter;
    const uint64_t start = doorbell_clock();
    ok = 1;
    while (*done != seq) {
      if (doorbell_clock() - start > DOORBELL_TIMEOUT_NS) {
        ok = 0;
        *timed_out = 1;
        break;
      }
#if __CUDA_ARCH__ >= 700
      __nanosleep(200);
#endif
    }
    __threadfence_system();
  }
  __syncthreads();
  const volatile float *src = host_ys;
  for (int i = threadIdx.x; i < values; i += blockDim.x) {
    ys[i] = ok ? src[i] : 0.0f;
  }
}

extern "C" int moe_doorbell_ring(void *counter, void *bell, int tokens, int64_t stream) {
  moe_doorbell_ring_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<uint32_t *>(counter), static_cast<volatile uint32_t *>(bell), tokens);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int moe_doorbell_wait(const void *counter, const void *done, const void *host_ys, void *ys, int values,
                                 void *timed_out, int64_t stream) {
  moe_doorbell_wait_kernel<<<1, DOORBELL_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t *>(counter), static_cast<volatile const uint32_t *>(done),
      static_cast<const float *>(host_ys), static_cast<float *>(ys), values, static_cast<uint32_t *>(timed_out));
  return static_cast<int>(cudaGetLastError());
}
