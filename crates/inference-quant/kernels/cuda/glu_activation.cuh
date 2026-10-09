// The GLU activations the matvec kernels fuse, numbered as GluActivationType on the Rust side.
#pragma once

enum GluActivation {
  GLU_SILU = 0,
  GLU_GELU = 1,
  GLU_RELU = 2,
  GLU_GELU_ERF = 3,
  GLU_SIGMOID = 4
};

static __device__ __forceinline__ float glu_silu(float x) {
  return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float glu_gelu(float x) {
  const float kSqrt2OverPi = 0.7978845608f;
  const float kCoeff = 0.044715f;
  const float x3 = x * x * x;
  const float inner = kSqrt2OverPi * (x + kCoeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

static __device__ __forceinline__ float glu_relu(float x) {
  return fmaxf(x, 0.0f);
}

static __device__ __forceinline__ float glu_gelu_erf(float x) {
  return x * normcdff(x);
}

static __device__ __forceinline__ float apply_glu_activation(float x, int act) {
  switch (act) {
  case GLU_SILU:
    return glu_silu(x);
  case GLU_GELU:
    return glu_gelu(x);
  case GLU_RELU:
    return glu_relu(x);
  case GLU_GELU_ERF:
    return glu_gelu_erf(x);
  case GLU_SIGMOID:
    return 1.0f / (1.0f + expf(-x));
  default:
    return glu_silu(x);
  }
}
