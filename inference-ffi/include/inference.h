/*
 * inference.h - C ABI for inference.rs (libinference_ffi).
 *
 * Contract
 *   - Only opaque handles cross the boundary. No Rust panic ever unwinds into the caller: every entry point catches it
 *     and returns INFERENCE_ERR_INTERNAL.
 *   - Every call that can fail returns an inference_status. On failure, inference_last_error() describes it; that string
 *     is thread-local, never NULL, and valid until the next inference_* call on the same thread.
 *   - Inputs (strings, pixel buffers, config structs) are copied during the call; the caller may free them afterwards.
 *   - Output pointers (const char*, arrays) are BORROWED from the handle that produced them and stay valid until that
 *     handle is freed. Label strings are static and stay valid for the life of the process.
 *   - Freeing NULL is a no-op. Results do not reference their model, so handles may be freed in any order.
 *   - A model handle may be used from several threads at once. Result handles are immutable.
 *   - Optional out-parameters may be NULL.
 *
 * Versioning
 *   inference_abi_version() returns (major << 16) | (minor << 8) | patch. A different major version must not be used.
 *   Minor versions only add entry points (callers may require a minimum); patch versions change behaviour only.
 *
 * Symbol surface
 *   The shared library exports exactly the inference_* functions declared here.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(INFERENCE_BUILD_DLL)
#    define INFERENCE_API __declspec(dllexport)
#  elif defined(INFERENCE_USE_DLL)
#    define INFERENCE_API __declspec(dllimport)
#  else
#    define INFERENCE_API
#  endif
#elif defined(__GNUC__)
#  define INFERENCE_API __attribute__((visibility("default")))
#else
#  define INFERENCE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_ABI_VERSION_MAJOR 0
#define INFERENCE_ABI_VERSION_MINOR 1
#define INFERENCE_ABI_VERSION_PATCH 0

typedef enum inference_status {
    INFERENCE_OK = 0,
    /* A NULL/out-of-range argument, malformed image description, or unknown backend name. */
    INFERENCE_ERR_INVALID_ARGUMENT = 1,
    /* The model directory could not be read or its weights/config are unusable. */
    INFERENCE_ERR_LOAD_FAILED = 2,
    /* Inference failed (device error, out of device memory, ...). */
    INFERENCE_ERR_RUNTIME = 3,
    /* An index passed to an accessor is past the end. */
    INFERENCE_ERR_OUT_OF_RANGE = 4,
    /* The requested backend (e.g. "cuda") is not compiled into this build or the device is missing. */
    INFERENCE_ERR_NOT_AVAILABLE = 5,
    /* A bug: an internal panic was caught at the boundary. */
    INFERENCE_ERR_INTERNAL = 6
} inference_status;

/* (major << 16) | (minor << 8) | patch of the ABI this library implements. */
INFERENCE_API uint32_t inference_abi_version(void);
/* Human-readable build identification, e.g. "inference.rs 0.9.3 (cuda)". Static. */
INFERENCE_API const char *inference_build_version(void);
/* Detail for the last failed call on this thread; "" if none. Never NULL. */
INFERENCE_API const char *inference_last_error(void);
/* Static name of a status code, e.g. "INFERENCE_ERR_LOAD_FAILED". */
INFERENCE_API const char *inference_status_string(inference_status status);

/* Where a model runs. Every field may be zero/NULL; the struct pointer itself may be NULL for all defaults. */
typedef struct inference_backend_config {
    /* "cpu" (default when NULL), "cuda", or "metal". */
    const char *backend;
    /* Device ordinal for "cuda"/"metal". */
    int32_t device;
    /* CPU worker threads; <= 0 uses one per physical core (hyperthreads slow the conv kernels). */
    int32_t threads;
} inference_backend_config;

typedef enum inference_pixel_format {
    INFERENCE_PIXEL_RGB8 = 0,
    INFERENCE_PIXEL_BGR8 = 1,
    /* Alpha is ignored (not composited), matching a plain RGB conversion. */
    INFERENCE_PIXEL_RGBA8 = 2,
    INFERENCE_PIXEL_BGRA8 = 3,
    INFERENCE_PIXEL_GRAY8 = 4
} inference_pixel_format;

/* An 8-bit image in caller memory; copied during the call. */
typedef struct inference_image {
    const uint8_t *pixels;
    uint32_t width;
    uint32_t height;
    /* Bytes per row; 0 means tightly packed (width * bytes per pixel). */
    uint32_t stride;
    /* An inference_pixel_format value. */
    int32_t format;
} inference_image;

/* ---- Document layout detection (PP-DocLayoutV3) ---------------------------------------------------------------- */

typedef struct inference_layout_model inference_layout_model;
typedef struct inference_layout_result inference_layout_result;

/* Pass as `threshold` to use the model's default score threshold (0.5). */
#define INFERENCE_LAYOUT_DEFAULT_THRESHOLD (-1.0f)

/* Loads an HF-format PP-DocLayoutV3 directory (config.json, preprocessor_config.json, model.safetensors). */
INFERENCE_API inference_status inference_layout_model_load(const char *model_dir,
                                                          const inference_backend_config *backend,
                                                          inference_layout_model **out_model);
INFERENCE_API void inference_layout_model_free(inference_layout_model *model);

/* The model's class labels, indexed by class id. */
INFERENCE_API size_t inference_layout_model_label_count(const inference_layout_model *model);
INFERENCE_API inference_status inference_layout_model_label(const inference_layout_model *model, size_t index,
                                                           const char **out_label);

/* Detects layout regions; `threshold` in [0, 1], or INFERENCE_LAYOUT_DEFAULT_THRESHOLD. */
INFERENCE_API inference_status inference_layout_detect(const inference_layout_model *model,
                                                      const inference_image *image, float threshold,
                                                      inference_layout_result **out_result);
/* Detects on `count` images in one batched forward; fills out_results[0..count). On failure every entry is NULL. */
INFERENCE_API inference_status inference_layout_detect_batch(const inference_layout_model *model,
                                                            const inference_image *images, size_t count,
                                                            float threshold,
                                                            inference_layout_result **out_results);
INFERENCE_API void inference_layout_result_free(inference_layout_result *result);

/* Detections are ordered by predicted reading order (index == reading order). */
INFERENCE_API size_t inference_layout_result_count(const inference_layout_result *result);
/* out_bbox receives 4 floats: x1, y1, x2, y2 in source-image pixels. */
INFERENCE_API inference_status inference_layout_result_detection(const inference_layout_result *result, size_t index,
                                                                int32_t *out_class_id, const char **out_label,
                                                                float *out_score, float *out_bbox);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
