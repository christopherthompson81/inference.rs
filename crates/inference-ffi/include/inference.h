/*
 * inference.h - C ABI for inference.rs (libinference_ffi).
 *
 * Contract
 *   - Only opaque handles cross the boundary. No Rust panic ever unwinds into the caller: every entry point catches it
 *     and returns INFERENCE_ERR_INTERNAL.
 *   - Every call that can fail returns an inference_status. inference_last_error() holds the detail of the most recent
 *     such call on the calling thread ("" if it succeeded); it is thread-local, never NULL, and valid until the next
 *     inference_* call on that thread.
 *   - Inputs (strings, pixel buffers, config structs) are copied during the call; the caller may free them afterwards.
 *   - Output pointers (const char*, arrays) are BORROWED from the handle that produced them and stay valid until that
 *     handle is freed. A result's label strings stay valid until the result is freed, even after its model is freed.
 *   - Freeing NULL is a no-op. Results do not reference their model, so handles may be freed in any order.
 *   - A model handle may be used from several threads at once. Result handles are immutable.
 *   - Optional out-parameters may be NULL. Count functions return 0 for a NULL handle.
 *
 * Versioning
 *   inference_abi_version() returns (major << 16) | (minor << 8) | patch. A different major version must not be used.
 *   Minor versions only add entry points and status codes (callers may require a minimum; treat an unknown status as
 *   an error); patch versions change behaviour only.
 *
 * Symbol surface
 *   The shared library exports exactly the inference_* functions declared here (tests/export_surface.py checks it).
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
#define INFERENCE_ABI_VERSION_MINOR 2
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
    INFERENCE_ERR_INTERNAL = 6,
    /* The engine rejected a request (malformed JSON, unknown model, bad parameters). inference_last_error() holds the
     * OpenAI error JSON: {"error": {"message", "type", "param", "code"}}. */
    INFERENCE_ERR_INVALID_REQUEST = 7,
    /* The engine is overloaded or unavailable; retrying later may succeed. inference_last_error() holds the error
     * JSON. */
    INFERENCE_ERR_UNAVAILABLE = 8
} inference_status;

/* (major << 16) | (minor << 8) | patch of the ABI this library implements. */
INFERENCE_API uint32_t inference_abi_version(void);
/* Human-readable build identification, e.g. "inference.rs 0.9.3 (cuda)". Static. */
INFERENCE_API const char *inference_build_version(void);
/* Detail of the most recent failing inference_* call on this thread; "" after a success. Never NULL. */
INFERENCE_API const char *inference_last_error(void);
/* Static name of a status code, e.g. "INFERENCE_ERR_LOAD_FAILED". */
INFERENCE_API const char *inference_status_string(inference_status status);

/* Where a model runs. Every field may be zero/NULL; the struct pointer itself may be NULL for all defaults. */
typedef struct inference_backend_config {
    /* "cpu" (default when NULL), "cuda", or "metal". */
    const char *backend;
    /* Device ordinal for "cuda"/"metal". */
    int32_t device;
    /* CPU worker threads, at most 1024. <= 0 is the process default: one per physical core (hyperthreads slow the conv
     * kernels), or rayon's global pool when RAYON_NUM_THREADS is set or the CPU lacks AVX2+FMA. */
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

/* An 8-bit image in caller memory; copied during the call. At most 2^28 pixels. */
typedef struct inference_image {
    const uint8_t *pixels;
    uint32_t width;
    uint32_t height;
    /* Bytes per row; 0 means tightly packed (width * bytes per pixel). */
    uint32_t stride;
    /* An inference_pixel_format value. */
    int32_t format;
} inference_image;

/* Document layout detection (PP-DocLayoutV3). */

typedef struct inference_layout_model inference_layout_model;
typedef struct inference_layout_result inference_layout_result;

/* Pass as `threshold` to use the model's default score threshold (0.5). */
#define INFERENCE_LAYOUT_DEFAULT_THRESHOLD (-1.0f)

/* Loads an HF-format PP-DocLayoutV3 directory (config.json, preprocessor_config.json, model.safetensors). */
INFERENCE_API inference_status inference_layout_model_load(const char *model_dir,
                                                          const inference_backend_config *backend,
                                                          inference_layout_model **out_model);
INFERENCE_API void inference_layout_model_free(inference_layout_model *model);

/* The model's class labels, indexed by class id (from its config; the Paddle names for the stock 25-class model). */
INFERENCE_API size_t inference_layout_model_label_count(const inference_layout_model *model);
INFERENCE_API inference_status inference_layout_model_label(const inference_layout_model *model, size_t index,
                                                           const char **out_label);

/* Detects layout regions; `threshold` in [0, 1], or exactly INFERENCE_LAYOUT_DEFAULT_THRESHOLD (anything else is
 * INFERENCE_ERR_INVALID_ARGUMENT, including NaN). */
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

/* Engine: a loaded model serving OpenAI-style requests. Requests and responses are the JSON the HTTP server accepts and
 * returns (e.g. POST /v1/chat/completions bodies). Engine calls block; they must not be made from inside a tokio
 * runtime thread. An engine handle may be used from several threads at once; a stream handle from one at a time.
 * Failing engine calls, including INFERENCE_ERR_RUNTIME ones, leave the OpenAI error JSON in inference_last_error().
 * Freeing the last handle of an engine (the engine or one of its streams) waits up to 10 s for the engine to stop.
 * Not yet on this surface: agent tool approvals (agent_permission "ask" is rejected) and uploaded skills. */

typedef struct inference_engine inference_engine;
typedef struct inference_stream inference_stream;
/* An owned, NUL-terminated UTF-8 JSON string. */
typedef struct inference_string inference_string;

/* Loads an engine from a JSON spec: {"model": <model selection>, "model_id"?, "runtime"?: {"device": "auto" | "cpu" |
 * "cuda:N" | "metal:N", "seed", "max_seqs", "prefix_cache_n", "no_kv_cache", "chat_template", "jinja_explicit",
 * "max_model_len", "isq", "paged_attn", "token_source"}, "agentic"?: {"max_tool_rounds", "tool_dispatch_url",
 * "agent_permission"}}. The model selection is the ModelSelected JSON, e.g. {"Plain": {"model_id": "org/model"}}.
 * A malformed spec is INFERENCE_ERR_INVALID_ARGUMENT, a device this build or machine lacks is
 * INFERENCE_ERR_NOT_AVAILABLE, and a model that fails to load is INFERENCE_ERR_LOAD_FAILED. */
INFERENCE_API inference_status inference_engine_load(const char *spec, size_t spec_len,
                                                    inference_engine **out_engine);
INFERENCE_API void inference_engine_free(inference_engine *engine);

/* Runs a chat completion to its end; out_response receives the chat.completion JSON. "stream" in the request is
 * ignored. */
INFERENCE_API inference_status inference_chat(const inference_engine *engine, const char *request,
                                             size_t request_len, inference_string **out_response);

/* Starts a streaming chat completion. Poll it with inference_stream_next; freeing it abandons the request. */
INFERENCE_API inference_status inference_chat_stream_open(const inference_engine *engine, const char *request,
                                                         size_t request_len, inference_stream **out_stream);
/* Waits up to timeout_ms (< 0 waits indefinitely, 0 polls) for the next event. On an event, out_event receives
 * {"event": "chunk" | "agentic_tool_call_progress" | "agentic_tool_approval_required" | "file_produced" | "error",
 * "data": ...}; a chunk's data is a chat.completion.chunk and an error's is the OpenAI error JSON. On a timeout
 * out_event is NULL and out_done 0. Once the stream has ended, out_event is NULL and out_done 1; an error event is
 * always the last event. out_event and out_done are required. */
INFERENCE_API inference_status inference_stream_next(inference_stream *stream, int64_t timeout_ms,
                                                    inference_string **out_event, int32_t *out_done);
INFERENCE_API void inference_stream_free(inference_stream *stream);

/* The string's bytes, NUL-terminated; valid until the string is freed. "" for NULL. */
INFERENCE_API const char *inference_string_data(const inference_string *string);
/* Length in bytes, excluding the terminating NUL. 0 for NULL. */
INFERENCE_API size_t inference_string_len(const inference_string *string);
INFERENCE_API void inference_string_free(inference_string *string);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
