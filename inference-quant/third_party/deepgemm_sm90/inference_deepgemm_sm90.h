// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum InferenceDeepGemmStatus {
  INFERENCE_RS_DEEPGEMM_SUCCESS = 0,
  INFERENCE_RS_DEEPGEMM_UNAVAILABLE = 1,
  INFERENCE_RS_DEEPGEMM_INVALID_ARGUMENT = 2,
  INFERENCE_RS_DEEPGEMM_NOT_PREPARED = 3,
  INFERENCE_RS_DEEPGEMM_WORKSPACE_TOO_SMALL = 4,
  INFERENCE_RS_DEEPGEMM_CAPTURE_ACTIVE = 5,
  INFERENCE_RS_DEEPGEMM_COMPILE_ERROR = 6,
  INFERENCE_RS_DEEPGEMM_CUDA_ERROR = 7,
  INFERENCE_RS_DEEPGEMM_INTERNAL_ERROR = 8,
} InferenceDeepGemmStatus;

typedef enum InferenceDeepGemmPlanFlags {
  INFERENCE_RS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB = 1U,
#ifdef INFERENCE_RS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
  INFERENCE_RS_DEEPGEMM_PLAN_SWAP_AB = INFERENCE_RS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB,
#endif
  INFERENCE_RS_DEEPGEMM_PLAN_OFFICIAL_1D2D = 2U,
  INFERENCE_RS_DEEPGEMM_PLAN_MULTICAST_ON_A = 4U,
  INFERENCE_RS_DEEPGEMM_PLAN_PDL = 8U,
} InferenceDeepGemmPlanFlags;

enum {
  INFERENCE_RS_DEEPGEMM_BLOCK_SIZE = 128U,
  INFERENCE_RS_DEEPGEMM_ACTIVATION_SCALE_M_ALIGNMENT = 4U,
};

typedef struct InferenceDeepGemmPlan {
  uint32_t abi_version;
  uint32_t flags;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t block_m;
  uint32_t block_n;
  uint32_t block_k;
  uint32_t num_stages;
  uint32_t num_tma_multicast;
  uint32_t sm_count;
  uint32_t smem_bytes;
  uint32_t device_ordinal;
  uint32_t reserved;
  size_t workspace_bytes;
  uint64_t cache_key;
} InferenceDeepGemmPlan;

typedef struct InferenceDeepGemmPrepared {
  InferenceDeepGemmPlan plan;
  uintptr_t function;
} InferenceDeepGemmPrepared;

const char* inference_deepgemm_sm90_error_string(int32_t status);

const char* inference_deepgemm_sm90_last_error();

int32_t inference_deepgemm_sm90_plan(uint32_t m, uint32_t n, uint32_t k,
                                     InferenceDeepGemmPlan* plan);

#ifdef INFERENCE_RS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
int32_t inference_deepgemm_sm90_plan_legacy_for_test(
    uint32_t m, uint32_t n, uint32_t k, InferenceDeepGemmPlan* plan);
#endif

int32_t inference_deepgemm_sm90_prepare(const InferenceDeepGemmPlan* plan,
                                        const char* include_dir,
                                        cudaStream_t stream,
                                        InferenceDeepGemmPrepared* prepared);

int32_t inference_deepgemm_sm90_gemm(const InferenceDeepGemmPrepared* prepared,
                                     uint32_t m,
                                     const void* activation_bf16,
                                     const void* weight_e4m3,
                                     const float* weight_scales,
                                     void* output_bf16, void* workspace,
                                     size_t workspace_bytes, cudaStream_t stream);

// activation_scales is contiguous [K / 128, activation_scale_stride_m].
int32_t inference_deepgemm_sm90_gemm_prequantized(
    const InferenceDeepGemmPrepared* prepared, uint32_t m,
    const void* activation_e4m3, const float* activation_scales,
    uint32_t activation_scale_stride_m, const void* weight_e4m3,
    const float* weight_scales, void* output_bf16, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
