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
 *   inference_abi_version() returns (major << 16) | (minor << 8) | patch. The ABI is unstable while it is 0.0.x: every
 *   release that changes it bumps the patch number and may add, change or remove entry points, so require an exact
 *   match. Compatibility rules (minor adds, patch fixes) start at 0.1.0.
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
#define INFERENCE_ABI_VERSION_MINOR 0
#define INFERENCE_ABI_VERSION_PATCH 26

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
    /* The engine rejected a request (malformed JSON, bad parameters). inference_last_error() holds the
     * OpenAI error JSON: {"error": {"message", "type", "param", "code"}}. */
    INFERENCE_ERR_INVALID_REQUEST = 7,
    /* The engine is overloaded or unavailable; retrying later may succeed. inference_last_error() holds the error
     * JSON. */
    INFERENCE_ERR_UNAVAILABLE = 8,
    /* The request names something that does not exist: a model, adapter or response id. inference_last_error() holds
     * the error JSON. */
    INFERENCE_ERR_NOT_FOUND = 9
} inference_status;

/* (major << 16) | (minor << 8) | patch of the ABI this library implements. */
INFERENCE_API uint32_t inference_abi_version(void);
/* Human-readable build identification, e.g. "inference.rs 0.0.26 (cuda)". Static. */
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

/* Loads PP-DocLayoutV3 from an HF-format directory (config.json, preprocessor_config.json, model.safetensors) or
 * from a GGUF file of it (`examples/pp_doclayout_v3_gguf.rs` in inference-layout writes one). */
INFERENCE_API inference_status inference_layout_model_load(const char *path,
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
/* The region's outline from its mask: *out_count (four or more) vertices as x, y float pairs in source-image pixels, the
 * box's corners when the mask gives no polygon. *out_points is borrowed until inference_layout_result_free(result). */
INFERENCE_API inference_status inference_layout_result_polygon(const inference_layout_result *result, size_t index,
                                                              const float **out_points, size_t *out_count);

/* Engine: a loaded model serving OpenAI-style requests. Requests and responses are the JSON the HTTP server accepts and
 * returns (e.g. POST /v1/chat/completions bodies). Engine calls block; they must not be made from inside a tokio
 * runtime thread. An engine handle may be used from several threads at once; a stream handle from one at a time, except that
 * inference_stream_cancel may be called from any thread.
 * Failing engine calls, including INFERENCE_ERR_RUNTIME ones, leave the error JSON in inference_last_error(): the
 * OpenAI envelope.
 * Freeing the last handle of an engine (the engine or one of its streams) waits up to 10 s for the engine to stop.
 */

typedef struct inference_engine inference_engine;
typedef struct inference_stream inference_stream;
/* An owned, NUL-terminated UTF-8 JSON string. */
typedef struct inference_string inference_string;
/* Owned bytes (audio, a file's content) and their MIME type. */
typedef struct inference_blob inference_blob;
/* A held CUDA device (see inference_device_lease_open); freeing it releases the device. */
typedef struct inference_device_lease inference_device_lease;
typedef struct inference_transcription_session inference_transcription_session;

/* Loads an engine from a JSON spec, the EngineSpec schema in docs/openapi.json: {"model": <model selection>,
 * "model_id"?, "runtime"?, "agentic"?, "adapters"?, "skills"?, "anymoe"?}, or several models as "models": [{"model",
 * "model_id"?, per-model overrides}] with "default_model_id"? in place of "model" and "model_id". "runtime" holds the
 * device ("auto" | "cpu" | "cuda:N" | "metal:N"), batching, quantization, paged-attention cache sizing and MTP;
 * "agentic" the tool loop's limits and permission, search reranking, and the MCP client, code execution and shell
 * tools. The model selection is the ModelSelected JSON, e.g. {"Plain": {"model_id": "org/model"}}. A malformed spec is
 * INFERENCE_ERR_INVALID_ARGUMENT, a device or feature this build or machine lacks is INFERENCE_ERR_NOT_AVAILABLE, and
 * a model that fails to load is INFERENCE_ERR_LOAD_FAILED. */
INFERENCE_API inference_status inference_engine_load(const char *spec, size_t spec_len,
                                                    inference_engine **out_engine);
INFERENCE_API void inference_engine_free(inference_engine *engine);
/* A second handle to the same loaded engine that acts for owner (non-empty UTF-8): the sessions, files,
 * skills and approvals it creates are owner's, and it reaches no one else's. A handle from
 * inference_engine_load is unscoped and likewise reaches only unscoped data. Free each handle with
 * inference_engine_free; the engine stays loaded until every handle is freed. Serving several users, make one each. */
INFERENCE_API inference_status inference_engine_for_owner(const inference_engine *engine, const char *owner,
                                                         size_t owner_len, inference_engine **out_engine);

/* Host callbacks: C functions the agent loop calls. A callback answers through the library-owned result it is given,
 * with inference_callback_result_set (text, copied; NULL with len 0 is empty, invalid UTF-8 is replaced) or
 * inference_callback_result_fail; the last call wins, and returning without either is a failure. The result is valid
 * only during the call, on the calling thread. Callbacks run on engine worker threads, possibly several at once, may
 * block, must not throw or longjmp, and must not call inference_* engine functions. user_data is passed back untouched.
 * Requests still finishing may call a callback shortly after inference_engine_free returns, so user_data must stay
 * valid beyond it, or the callback must recognise a stale user_data (e.g. an id whose handler is gone) and fail. */
typedef struct inference_callback_result inference_callback_result;
INFERENCE_API void inference_callback_result_set(inference_callback_result *result, const char *data, size_t len);
/* message may be NULL. */
INFERENCE_API void inference_callback_result_fail(inference_callback_result *result, const char *message);

/* Runs a host tool. arguments is the JSON the model passed; context is {"session_id", "round"} (the agent loop's
 * round). The result text is what the model sees; a failure reaches the model as a failed tool call. */
typedef void (*inference_tool_callback)(void *user_data, const char *tool_name, const char *arguments,
                                        size_t arguments_len, const char *context, size_t context_len,
                                        inference_callback_result *result);
/* Answers a web search with a JSON array of {"title", "description", "url", "content"}. A failure, or JSON that is not
 * that array, is logged and the model sees no results. */
typedef void (*inference_search_callback)(void *user_data, const char *query, size_t query_len,
                                          inference_callback_result *result);

/* definition is the OpenAI function tool JSON ({"type": "function", "function": {"name", "description",
 * "parameters"}}). Every chat request offers the host tools to the model (and so runs the agent loop); a request that
 * declares its own tool of the same name is refused. Two host tools may not share a name, and a built-in tool (MCP,
 * code execution, shell) of the same name replaces a host tool. */
typedef struct inference_host_tool {
    const char *definition;
    size_t definition_len;
    inference_tool_callback callback;
    void *user_data;
} inference_host_tool;

/* Receives model-load progress as JSON {"stage", "done", "total", "unit"}: "stage" names the step ("Loading repeating
 * layers", "Loading model-00001-of-00002.safetensors", "Downloading config.json"), "done" of "total" counts "unit"s
 * ("items" or "bytes"). Each stage reports at its start (done 0), at each whole percent and when its last unit is done;
 * stages may run at once (a model's shards load in parallel), and a name may recur (a vision tower, then the text
 * stack). It hears every model load in the process while the engine lives: its own load, models it adds or reloads
 * (including on demand), and other engines' loads. It runs on the loading threads under the host callback rules above;
 * event is valid only during the call. */
typedef void (*inference_progress_callback)(void *user_data, const char *event, size_t event_len);

/* tools may be NULL when tool_count is 0; search may be NULL to keep the built-in search; progress may be NULL. */
typedef struct inference_host_callbacks {
    const inference_host_tool *tools;
    size_t tool_count;
    inference_search_callback search;
    void *search_user_data;
    inference_progress_callback progress;
    void *progress_user_data;
} inference_host_callbacks;

/* inference_engine_load with host callbacks; callbacks may be NULL. A malformed tool is
 * INFERENCE_ERR_INVALID_ARGUMENT. */
INFERENCE_API inference_status inference_engine_load_with_callbacks(const char *spec, size_t spec_len,
                                                                   const inference_host_callbacks *callbacks,
                                                                   inference_engine **out_engine);

/* Registers a host tool after load, under its definition's function name. Unlike the tools passed at load, which every
 * chat request offers, a registered tool is offered only by a chat request that
 * names it in its "host_tools" array. Every handle of the engine shares it. A name already registered is
 * INFERENCE_ERR_INVALID_REQUEST with code "host_tool_conflict"; a request naming an unregistered one is
 * INFERENCE_ERR_INVALID_REQUEST with param "host_tools", and one naming a tool that clashes with another of its tools
 * (one it declares, a tool given at load, an MCP or built-in tool) is INFERENCE_ERR_INVALID_REQUEST. Registering a
 * name a tool given at load already has succeeds, but every request naming it is refused that way. tool_choice sees
 * only a request's declared tools. The callback runs under the host callback rules above; the definition is copied. */
INFERENCE_API inference_status inference_engine_register_tool(const inference_engine *engine,
                                                             const inference_host_tool *tool);
/* Requests already running keep the tool, so user_data must stay valid until they finish. An unknown name is
 * INFERENCE_ERR_NOT_FOUND. */
INFERENCE_API inference_status inference_engine_unregister_tool(const inference_engine *engine, const char *name,
                                                               size_t name_len);

/* Edits one decoding step's logits in place: vocab_size floats, after the sampling penalties and before the token is
 * picked. context holds every token so far, the prompt's included; both are valid only during the call. A request with
 * a grammar may call it twice in a step, when the first pick breaks the grammar. Return 0 to go on; any other value
 * fails the request it runs in (as an internal error). Runs on an engine worker thread under the same rules as the host
 * callbacks above. */
typedef int32_t (*inference_logits_processor_callback)(void *user_data, float *logits, size_t vocab_size,
                                                       const uint32_t *context, size_t context_len);
/* Makes callback selectable by name (non-empty UTF-8) in a request's "logits_processors" array (chat and
 * completions), applied in the order listed. Every handle of the engine shares it. A name already
 * registered is INFERENCE_ERR_INVALID_REQUEST with code "logits_processor_conflict"; a request naming an unregistered
 * one is INFERENCE_ERR_INVALID_REQUEST with param "logits_processors". */
INFERENCE_API inference_status inference_engine_register_logits_processor(
    const inference_engine *engine, const char *name, size_t name_len, inference_logits_processor_callback callback,
    void *user_data);
/* Requests already running keep the processor, so user_data must stay valid until they finish. An unknown name is
 * INFERENCE_ERR_NOT_FOUND. */
INFERENCE_API inference_status inference_engine_unregister_logits_processor(const inference_engine *engine,
                                                                          const char *name, size_t name_len);

/* Runs a chat completion to its end; out_response receives the chat.completion JSON. "stream" in the request is
 * ignored. */
INFERENCE_API inference_status inference_chat(const inference_engine *engine, const char *request,
                                             size_t request_len, inference_string **out_response);

/* A buffer passed with a request (image, audio or video bytes, copied during the call). The request names it by
 * position: an image_url / audio_url / video_url of "media://0" is the first entry. data must not be NULL; mime_type
 * may be. */
typedef struct inference_media {
    const uint8_t *data;
    size_t len;
    const char *mime_type;
} inference_media;

/* inference_chat with media attachments. media may be NULL when media_count is 0. */
INFERENCE_API inference_status inference_chat_with_media(const inference_engine *engine, const char *request,
                                                        size_t request_len, const inference_media *media,
                                                        size_t media_count, inference_string **out_response);

/* Starts a streaming chat completion. Poll it with inference_stream_next; freeing it abandons the request. */
INFERENCE_API inference_status inference_chat_stream_open(const inference_engine *engine, const char *request,
                                                         size_t request_len, inference_stream **out_stream);
/* inference_chat_stream_open with media attachments. */
INFERENCE_API inference_status inference_chat_stream_open_with_media(const inference_engine *engine,
                                                                    const char *request, size_t request_len,
                                                                    const inference_media *media, size_t media_count,
                                                                    inference_stream **out_stream);
/* Waits up to timeout_ms (< 0 waits indefinitely, 0 polls) for the next event of a stream. On an event, out_event
 * receives {"event": <name>, "data": ...}. Chat streams emit "chunk" (data: a chat.completion.chunk),
 * "agentic_tool_call_progress", "agentic_tool_approval_required", "file_produced" and "error"; completion streams emit
 * "chunk" (data: a text_completion chunk) and "error". An error's data is the OpenAI error JSON, and an error event is always the last event. On a timeout out_event is NULL and
 * out_done 0. Once the stream has ended, out_event is NULL and out_done 1. out_event and out_done are required. */
INFERENCE_API inference_status inference_stream_next(inference_stream *stream, int64_t timeout_ms,
                                                    inference_string **out_event, int32_t *out_done);
/* Asks a stream's request to stop on its next sampled token. Keep polling: the stream still ends with its final
 * event (a chunk with finish_reason "canceled", message_stop, or response.cancelled) carrying usage. Unlike the other
 * stream calls it may run on any thread, including while another thread waits in inference_stream_next; it must not
 * race inference_stream_free. Cancelling a finished stream does nothing. */
INFERENCE_API inference_status inference_stream_cancel(const inference_stream *stream);
INFERENCE_API void inference_stream_free(inference_stream *stream);

/* Runs a text completion (the POST /v1/completions body) to its end; out_response receives the text_completion JSON.
 * "stream" in the request is ignored. */
INFERENCE_API inference_status inference_completion(const inference_engine *engine, const char *request,
                                                   size_t request_len, inference_string **out_response);
/* Starts a streaming completion; poll it with inference_stream_next. Its events are "chunk" (a completion chunk) and
 * "error". */
INFERENCE_API inference_status inference_completion_stream_open(const inference_engine *engine, const char *request,
                                                               size_t request_len, inference_stream **out_stream);
/* Embeds every input of an embeddings request (the POST /v1/embeddings body); out_response receives the list JSON. */
INFERENCE_API inference_status inference_embeddings(const inference_engine *engine, const char *request,
                                                   size_t request_len, inference_string **out_response);
/* Scores each document against the query with a cross-encoder (the POST /v1/rerank body); out_response receives the
 * results JSON, most relevant first. */
INFERENCE_API inference_status inference_rerank(const inference_engine *engine, const char *request,
                                               size_t request_len, inference_string **out_response);
/* Answers typed questions about a state with a decision model (the POST /v1/decisions body); out_response receives
 * each question's answer JSON. */
INFERENCE_API inference_status inference_decide(const inference_engine *engine, const char *request,
                                               size_t request_len, inference_string **out_response);

/* The served models (the GET /v1/models body): the "default" alias, each model with its status, and each loaded LoRA
 * adapter as a model of its own. The model the alias goes to has "default": true. */
INFERENCE_API inference_status inference_models_list(const inference_engine *engine, inference_string **out_response);
/* Whether a request naming a model would be routed: the request is {"model_id"} and out_response receives {"model_id",
 * "served"}, true for a served model, the "default" alias or a LoRA adapter (as "parent::alias", or the alias alone
 * when only one model has it). */
INFERENCE_API inference_status inference_model_served(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
/* The tools the engine's MCP servers give the default model, built-in ones aside: {"object": "list", "data":
 * [{"name", "description"}]}, description null when the server gave none. */
INFERENCE_API inference_status inference_mcp_tools_list(const inference_engine *engine,
                                                       inference_string **out_response);
/* Each loaded model's cache counters (the GET /v1/models/cache_stats body): {"object": "list", "data": [{"model_id",
 * "prefix_cache_hits", "prefix_cache_sequences", "encoder_cache": {"hits", "misses"}}]}, sorted by model_id. They
 * count since the model loaded (the prefix ones since its engine last started); encoder_cache is absent for a model
 * without one. Diff two readings to see what the requests between them used. */
INFERENCE_API inference_status inference_models_cache_stats(const inference_engine *engine,
                                                          inference_string **out_response);
/* Each loaded model's speculative decoding counters (the GET /v1/models/speculative_stats body): {"object": "list",
 * "data": [{"model_id", "drafts", "draft_tokens_proposed", "draft_tokens_accepted", "accepted_per_position"}]}, sorted
 * by model_id. They count since the model loaded and stay zero without a speculative proposer. */
INFERENCE_API inference_status inference_models_speculative_stats(const inference_engine *engine,
                                                                inference_string **out_response);
/* Unloads, reloads or reports a model; the request is {"model_id"} and out_response receives {"model_id", "status":
 * "loaded" | "unloaded" | "reloading"}. Unloading an unloaded model or reloading a loaded one succeeds. An unknown
 * model is INFERENCE_ERR_NOT_FOUND. */
INFERENCE_API inference_status inference_model_unload(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_model_reload(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_model_status(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
/* Adds a model to a running engine; the request is one entry of the spec's "models" ({"model", "model_id"?, ...})
 * and it loads with the engine's runtime settings, out_response receiving {"model_id", "status": "loaded"}. A bad spec,
 * a failed load or a model_id already served (code "model_conflict") is INFERENCE_ERR_INVALID_REQUEST; a
 * tensor-parallel engine refuses with INFERENCE_ERR_UNAVAILABLE. Blocks while the model loads. */
INFERENCE_API inference_status inference_model_add(const inference_engine *engine, const char *request,
                                                  size_t request_len, inference_string **out_response);
/* Removes a served model and frees it ({"model_id"} in, {"model_id"} out); when it was the default, another model
 * becomes the default. An unknown model is INFERENCE_ERR_NOT_FOUND; removing the last one, or an unloaded one, is
 * INFERENCE_ERR_INVALID_REQUEST with code "model_conflict". */
INFERENCE_API inference_status inference_model_remove(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
/* Makes a served model the one requests without "model", or naming "default", go to ({"model_id"} in and out). */
INFERENCE_API inference_status inference_model_set_default(const inference_engine *engine, const char *request,
                                                          size_t request_len, inference_string **out_response);
/* Lets requests name a served model by another id ({"alias", "model_id"} in and out). An alias that already names a
 * model, an adapter or "default" is INFERENCE_ERR_INVALID_REQUEST with code "model_conflict". */
INFERENCE_API inference_status inference_model_alias(const inference_engine *engine, const char *request,
                                                    size_t request_len, inference_string **out_response);

/* LoRA adapters. Listing takes {"model"?} and returns the GET /v1/lora_adapters body. Loading takes {"lora_name",
 * "lora_path", "load_inplace"?, "expected_generation"?, "model"?} and unloading {"lora_name", "expected_generation"?,
 * "model"?}; both return the adapter object and need "adapters": {"runtime_updates": true} in the engine spec (and,
 * with "root", a lora_path under it). One load runs at a time; a second is INFERENCE_ERR_UNAVAILABLE. Disabled updates
 * ("lora_updates_disabled"), a path outside the root ("adapter_path_forbidden") and a model mid-reload or an adapter
 * runtime at capacity (the error's "code" says which) are INFERENCE_ERR_INVALID_REQUEST. */
INFERENCE_API inference_status inference_lora_adapters_list(const inference_engine *engine, const char *request,
                                                           size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_lora_adapter_load(const inference_engine *engine, const char *request,
                                                          size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_lora_adapter_unload(const inference_engine *engine, const char *request,
                                                            size_t request_len, inference_string **out_response);

/* Generates images with a diffusion model (the POST /v1/images/generations body); out_response receives the image
 * list JSON. A "url" image is a PNG in the file store: its url is /v1/files/<id>/content, read with
 * inference_file_content. */
INFERENCE_API inference_status inference_image_generation(const inference_engine *engine, const char *request,
                                                         size_t request_len, inference_string **out_response);
/* Edits an image, or places subject references in a new scene (the POST /v1/images/edits JSON body), with a model
 * that takes reference images; each "images" entry's image_url may be "media://N" for media[N]. out_response
 * receives the image list JSON, as inference_image_generation's. media may be NULL when media_count is 0. */
INFERENCE_API inference_status inference_image_edit(const inference_engine *engine, const char *request,
                                                   size_t request_len, const inference_media *media,
                                                   size_t media_count, inference_string **out_response);
/* Scores a prompt in one forward pass. The request is {"model"?, "prompt": text | [token ids], "output"?:
 * "logprobs" | "logits"}; text is tokenized with the model's special tokens. out_response receives {"tokens",
 * "vocab_size", "token_logprobs"}, token_logprobs[i] being log p(tokens[i] | tokens[..i]) and null for the first token.
 * With "output": "logits", out_blob (then required) receives tokens * vocab_size little-endian f32 values, row-major,
 * MIME "application/x-f32le", and a NULL out_blob is INFERENCE_ERR_INVALID_ARGUMENT; otherwise *out_blob is set NULL
 * when out_blob is given. The prompt must fit the model's context; a model that isn't a text or multimodal one, a
 * one-token prompt or one past the context is INFERENCE_ERR_INVALID_REQUEST. */
INFERENCE_API inference_status inference_prompt_logits(const inference_engine *engine, const char *request,
                                                      size_t request_len, inference_string **out_response,
                                                      inference_blob **out_blob);
/* Speaks text with a speech model (the POST /v1/audio/speech body; "response_format" is "wav" or "pcm"). out_blob
 * receives the encoded audio; its MIME type carries the sample rate and channel count, e.g.
 * "audio/pcm; codecs=1; format=s16le; rate=44100; channels=1". */
INFERENCE_API inference_status inference_speech_generation(const inference_engine *engine, const char *request,
                                                          size_t request_len, inference_blob **out_blob);
/* Transcribes audio with a speech recognition model. request holds the POST /v1/audio/transcriptions form fields as
 * JSON ({"model"?, "response_format"?: "json" | "text" | "srt" | "verbose_json" | "vtt", "timestamp_granularities"?:
 * ["word", "segment"], "language"?, "beam_size"?}; "language" conditions Nemotron-3.5 and is ignored by Parakeet, as
 * are "prompt" and "temperature"; "beam_size" (1 to 16) beam-searches an RNN-T or TDT head); audio is audio_len bytes
 * of an encoded file (WAV, MP3, FLAC, OGG, ..., at most 64 MiB). out_blob receives the transcript in that format, its
 * MIME type naming it, e.g. "application/json". Audio that does not decode, a language the model does not take, a
 * beam out of range or on a CTC head, or a recording past 24 minutes on a full-attention model loaded without a VAD
 * ("vad_model_id" in its spec), is INFERENCE_ERR_INVALID_REQUEST. */
INFERENCE_API inference_status inference_transcription(const inference_engine *engine, const char *request,
                                                      size_t request_len, const uint8_t *audio, size_t audio_len,
                                                      inference_blob **out_blob);
/* Opens a streamed transcription on a cache-aware streaming recogniser (Nemotron streaming ASR): audio is pushed as it
 * arrives and each push returns the text it completes, the model deciding as it would over the whole recording.
 * request is {"model"?, "language"?, "sample_rate"? (of the pushed audio, default 16000; resampled to the model's),
 * "response_format"?, "timestamp_granularities"?}, the last two shaping the finished transcript as for
 * inference_transcription. A sample_rate outside 8000 to 192000, a model that only transcribes whole recordings, a
 * language it does not take, or a model already holding 64 open sessions (from any owner) is
 * INFERENCE_ERR_INVALID_REQUEST. The session stays on the model it opened on, even if "model" was an alias or the
 * default. A session is used by one thread at a time; it may move between threads. A session that gets no push for
 * 10 minutes is released, so a live client pushes its silence too; pushes wait behind any whole-recording upload the
 * same model is transcribing. */
INFERENCE_API inference_status inference_transcription_session_open(const inference_engine *engine,
                                                                   const char *request, size_t request_len,
                                                                   inference_transcription_session **out_session);
/* Pushes audio_len bytes of little-endian 16-bit mono PCM at the session's rate (an even count). out_response
 * receives {"delta"}, the text the audio so far completes; the model emits it a chunk (about a second) at a time, so
 * a short push often adds none. A push that fails ends the session; later calls report it not open. */
INFERENCE_API inference_status inference_transcription_session_push(inference_transcription_session *session,
                                                                   const uint8_t *audio, size_t audio_len,
                                                                   inference_string **out_response);
/* Ends the session: out_delta receives {"delta"} with the text its last audio adds, out_blob the whole transcript in
 * the session's response_format, its MIME type naming it. Audio too short to transcribe is
 * INFERENCE_ERR_INVALID_REQUEST. The session takes no pushes after this; free it. */
INFERENCE_API inference_status inference_transcription_session_finish(inference_transcription_session *session,
                                                                     inference_string **out_delta,
                                                                     inference_blob **out_blob);
/* Frees a session, closing it if unfinished. */
INFERENCE_API void inference_transcription_session_free(inference_transcription_session *session);
/* Finds the speech in audio with a voice activity model (Silero VAD). request holds the POST /v1/audio/vad form
 * fields as JSON ({"model"?, "threshold"?, "neg_threshold"?, "min_speech_duration_ms"?, "max_speech_duration_s"?,
 * "min_silence_duration_ms"?, "speech_pad_ms"?, "return_probabilities"?}, unset fields keeping the reference
 * get_speech_timestamps defaults); audio is audio_len bytes of an encoded file (at most 64 MiB). out_response receives
 * {"duration", "segments": [{"start", "end"}], "chunk_seconds", "probabilities"?}, times in seconds. Audio that does
 * not decode, a negative duration or pad, or a model that is not a voice activity one is
 * INFERENCE_ERR_INVALID_REQUEST. */
INFERENCE_API inference_status inference_voice_activity(const inference_engine *engine, const char *request,
                                                       size_t request_len, const uint8_t *audio, size_t audio_len,
                                                       inference_string **out_response);
/* Who speaks when in audio, from a speaker diarization model (Nemotron-3 or Sortformer). request holds the POST
 * /v1/audio/diarization form fields as JSON ({"model"?, "threshold"? (0.5), "response_format"?: "json" | "rttm",
 * "return_probabilities"?}); audio is audio_len bytes of an encoded file (at most 64 MiB). out_blob receives, as JSON,
 * {"duration", "num_speakers", "segments": [{"speaker", "start", "end"}], "frame_seconds", "probabilities"?: one row of
 * per-speaker probabilities per frame}, or RTTM SPEAKER lines; its MIME type names which. Audio that does not decode,
 * a threshold outside 0 to 1, or a model that is not a diarization one, is INFERENCE_ERR_INVALID_REQUEST. */
INFERENCE_API inference_status inference_diarization(const inference_engine *engine, const char *request,
                                                    size_t request_len, const uint8_t *audio, size_t audio_len,
                                                    inference_blob **out_blob);
/* Answers the approval an "agentic_tool_approval_required" stream event named (its "approval_id"); the request is
 * {"decision": "approve" | "deny", "remember_for_session"?, "message"?} and out_response receives {"status":
 * "resolved" | "queued"}. An unknown approval is INFERENCE_ERR_NOT_FOUND. Approvals only arise on streamed requests
 * with agent_permission "ask" (per request or in the spec, which makes blocking chat calls refuse); unanswered
 * ones, including those of a freed stream, are denied after 5 minutes and hold their sequence slot until then. */
INFERENCE_API inference_status inference_approval_resolve(const inference_engine *engine, const char *approval_id,
                                                         size_t approval_id_len, const char *request,
                                                         size_t request_len, inference_string **out_response);

/* The engine's file store, shared by its models: uploads that requests name by id, files agentic tools produce, and
 * "url" images. Uploading copies len bytes (at most 64 MiB; data must not be NULL) under filename and purpose (e.g.
 * "user_data"); mime_type may be NULL. Metadata calls return the /v1/files JSON; content returns the bytes, and a
 * body the store elided is INFERENCE_ERR_NOT_FOUND with code "file_content_unavailable".
 */
INFERENCE_API inference_status inference_file_upload(const inference_engine *engine, const uint8_t *data, size_t len,
                                                    const char *filename, const char *mime_type, const char *purpose,
                                                    inference_string **out_response);
/* Every stored file, for the host application. The HTTP server only lists them when started with --allow-file-listing,
 * since its store is shared by every client; a host relaying this to several users should scope it itself. */
INFERENCE_API inference_status inference_files_list(const inference_engine *engine, inference_string **out_response);
INFERENCE_API inference_status inference_file_get(const inference_engine *engine, const char *file_id,
                                                 size_t file_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_file_delete(const inference_engine *engine, const char *file_id,
                                                    size_t file_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_file_content(const inference_engine *engine, const char *file_id,
                                                     size_t file_id_len, inference_blob **out_blob);

/* The engine's CUDA devices, for GPU work of the caller's own. The requests name devices by CUDA ordinal ({"devices":
 * [0, 1]}; every visible one when "devices" is absent); without one, the calls are INFERENCE_ERR_UNAVAILABLE with
 * code "device_unavailable". Snapshot returns {"data": [{"device", "free", "total", "held_by", "models": [{"id",
 * "bytes", "pinned", "quiet", "last_used"}]}]}: free and total bytes, the holder of a lease on it, and its resident
 * models. Place takes {"devices", "need": "device" | {"bytes": n}, "displace": "busy" | "idle" | "nothing"} and returns
 * where the work would go now, {"status": "run", "device", "evict": [model ids]}, or why it cannot yet,
 * {"status": "wait", "reason"}, or ever, {"status": "never", "reason"}; "displace" says which models it may unload
 * (any, idle ones only, none). Lease open takes {"device", "holder", "unload": [model ids], "drain_timeout_secs"} and
 * holds the device for holder until inference_device_lease_free: nothing loads there meanwhile, and each model there
 * (or each of "unload") unloads once it goes quiet. It blocks while they drain, and fails
 * (INFERENCE_ERR_INVALID_REQUEST, code "device_busy") if the device is already held, a pinned model would have to go,
 * or a model stays busy past the timeout (120 s by default). */
INFERENCE_API inference_status inference_devices_snapshot(const inference_engine *engine, const char *request,
                                                         size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_device_place(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_device_lease_open(const inference_engine *engine, const char *request,
                                                          size_t request_len, inference_device_lease **out_lease);
INFERENCE_API void inference_device_lease_free(inference_device_lease *lease);

/* Jobs: world generation and reconstruction, queued and run one at a time. A job holds its CUDA device to itself while
 * it runs, unloading the idle models there (a pinned one makes it fail) and refusing loads there until it ends.
 * Configure takes and returns the JobsConfig JSON ({"dir", "worldstereo", "wan", "moge", "worldmirror", "lpips",
 * "device_ordinal", "drain_timeout_secs"}); submit takes {"request": JobRequest, "inputs": [paths]} and returns the
 * queued JobRecord. Poll inference_job_get to follow one: its stages, previews and outputs name files in the job's
 * folder, which inference_job_file resolves ({"id", "file"} to {"path"}). Cancel stops a job at its next step;
 * delete removes a finished one with its folder. A job of another owner is INFERENCE_ERR_NOT_FOUND. */
INFERENCE_API inference_status inference_jobs_configure(const inference_engine *engine, const char *request,
                                                       size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_job_submit(const inference_engine *engine, const char *request,
                                                   size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_jobs_list(const inference_engine *engine, inference_string **out_response);
INFERENCE_API inference_status inference_job_get(const inference_engine *engine, const char *job_id, size_t job_id_len,
                                                inference_string **out_response);
INFERENCE_API inference_status inference_job_cancel(const inference_engine *engine, const char *job_id,
                                                   size_t job_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_job_delete(const inference_engine *engine, const char *job_id,
                                                   size_t job_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_job_file(const inference_engine *engine, const char *request,
                                                 size_t request_len, inference_string **out_response);
/* The files a container (a code-running session, its "container_id") produced, as the
 * /v1/containers/{id}/files JSON; a file of another container is INFERENCE_ERR_NOT_FOUND. */
INFERENCE_API inference_status inference_container_files_list(const inference_engine *engine,
                                                             const char *container_id, size_t container_id_len,
                                                             inference_string **out_response);
INFERENCE_API inference_status inference_container_file_get(const inference_engine *engine, const char *container_id,
                                                           size_t container_id_len, const char *file_id,
                                                           size_t file_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_container_file_content(const inference_engine *engine,
                                                               const char *container_id, size_t container_id_len,
                                                               const char *file_id, size_t file_id_len,
                                                               inference_blob **out_blob);

/* One file of a skill upload: path within the skill (e.g. "SKILL.md", "scripts/run.py") and its bytes. */
typedef struct inference_skill_file {
    const char *path;
    const uint8_t *data;
    size_t len;
} inference_skill_file;

/* Skills: a SKILL.md with name and description frontmatter, and the files it references, stored under the spec's
 * skills.root for requests to mount in the shell tool. Uploads return the skill (or version) JSON; lists return
 * {"object": "list", "data": [...]}. */
INFERENCE_API inference_status inference_skill_upload(const inference_engine *engine, const inference_skill_file *files,
                                                     size_t file_count, inference_string **out_response);
INFERENCE_API inference_status inference_skill_version_upload(const inference_engine *engine, const char *skill_id,
                                                             size_t skill_id_len, const inference_skill_file *files,
                                                             size_t file_count, inference_string **out_response);
INFERENCE_API inference_status inference_skills_list(const inference_engine *engine, inference_string **out_response);
INFERENCE_API inference_status inference_skill_versions_list(const inference_engine *engine, const char *skill_id,
                                                            size_t skill_id_len, inference_string **out_response);

/* Requantizes a model, which must have loaded with ISQ, to {"ggml_type", "model"?} (an ISQ type such as "Q4K";
 * numeric shorthands resolve as on the CPU; the default model when "model" is absent); out_response echoes it once the
 * model is requantized. A model not loaded with ISQ is INFERENCE_ERR_INVALID_REQUEST (code "requantize_failed"). */
INFERENCE_API inference_status inference_re_isq(const inference_engine *engine, const char *request, size_t request_len,
                                               inference_string **out_response);
/* Online calibration: start collecting activation statistics from live traffic, report per-layer progress, or apply
 * them, requantizing from the source weights and hot-swapping each layer. Start and status take {"model"?}; apply
 * takes {"save_cimatrix"?, "model"?}, save_cimatrix a path to also save the importance matrix to, unrestricted and
 * relative to the process's working directory. Each acts on the default model when "model" is absent and returns the
 * calibration status JSON, as it stood before an apply; a model without ISQ, or an apply with nothing collected, is
 * INFERENCE_ERR_INVALID_REQUEST with the reason. */
INFERENCE_API inference_status inference_calibration_start(const inference_engine *engine, const char *request,
                                                          size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_calibration_status(const inference_engine *engine, const char *request,
                                                           size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_calibration_apply(const inference_engine *engine, const char *request,
                                                          size_t request_len, inference_string **out_response);

/* Agentic sessions (a chat request's "session_id"): list their ids as {"data"}, export one (the GET
 * /v1/sessions/{id} body; an unknown id is INFERENCE_ERR_NOT_FOUND), import one under an id, replacing any session
 * there ({"id"}), or delete one ({"id", "deleted"}, false when there was none). */
INFERENCE_API inference_status inference_sessions_list(const inference_engine *engine, inference_string **out_response);
INFERENCE_API inference_status inference_session_get(const inference_engine *engine, const char *session_id,
                                                    size_t session_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_session_put(const inference_engine *engine, const char *session_id,
                                                    size_t session_id_len, const char *session, size_t session_len,
                                                    inference_string **out_response);
INFERENCE_API inference_status inference_session_delete(const inference_engine *engine, const char *session_id,
                                                       size_t session_id_len, inference_string **out_response);
/* Branches a session: the request is {"num_turns"} and out_response receives {"id"} of a new session, named by the
 * engine, holding the first num_turns turns (a turn ends at an assistant message without tool calls); 0, or more turns
 * than the session has, copies all of it. An unknown session is INFERENCE_ERR_INVALID_REQUEST. */
INFERENCE_API inference_status inference_session_fork(const inference_engine *engine, const char *session_id,
                                                     size_t session_id_len, const char *request, size_t request_len,
                                                     inference_string **out_response);

/* Tokenizes {"text", "add_special_tokens"?, "model"?} to {"tokens"}, and detokenizes {"tokens",
 * "skip_special_tokens"?, "model"?} to {"text"}; both flags default to true. */
INFERENCE_API inference_status inference_tokenize(const inference_engine *engine, const char *request,
                                                 size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_detokenize(const inference_engine *engine, const char *request,
                                                   size_t request_len, inference_string **out_response);
/* Tokenizes a chat completion request as the model's chat template renders it, with its tools and reasoning controls
 * and the generation prompt, to {"tokens"}. Media given as URLs or data URLs are fetched as a chat would fetch them;
 * "media://N" sources cannot be tokenized, since the call takes no buffers. */
INFERENCE_API inference_status inference_tokenize_chat(const inference_engine *engine, const char *request,
                                                      size_t request_len, inference_string **out_response);

/* Host, device and build information, and environment diagnostics (the /v1/system/info and /v1/system/doctor
 * JSON). They need no engine. */
INFERENCE_API inference_status inference_system_info(inference_string **out_response);
INFERENCE_API inference_status inference_system_doctor(inference_string **out_response);
/* Picks the quantization and settings that fit a model on this machine without loading it (the POST /v1/models/tune
 * body: {"model_id", "profile"?, ...}); out_response receives the candidates and the recommended one. */
INFERENCE_API inference_status inference_model_tune(const char *request, size_t request_len,
                                                   inference_string **out_response);

/* The models on this machine, read without loading them; none needs an engine. Each model is {"path" (what a spec
 * loads), "source" ("local" | "hf_cache"), "repo_id", "format" ("gguf" | "safetensors" | "nemo" | "other"), "kind"
 * ("text" | "multimodal" | "embedding" | "image_generation" | "speech" | "transcription" | "voice_activity" |
 * "diarization" | "projector", null when it cannot load), "loader", "architecture", "name", "size_label",
 * "quantization", "context_length", "base_model", "size_bytes", "modified" (Unix seconds), "files" (a GGUF's shards),
 * "projectors" (for a text GGUF, the mmproj GGUFs beside it made for the same base model), "error"}.
 * inference_model_inspect reads one ({"path"}: a model directory, a GGUF by any shard, or a .nemo) and fails with
 * INFERENCE_ERR_INVALID_REQUEST for anything else. inference_models_scan takes {"folders": [...], "hf_cache": bool,
 * "hf_cache_path"?} and returns {"models": [...], "hf_cache": {"path", "repos": [{"repo_id", "size_bytes", "modified"}]} | null,
 * "errors": [{"path", "error"}]}. inference_hf_cache_delete takes {"repo_id": "org/name", "hf_cache_path"?}, removes
 * every revision of it from the Hugging Face cache, and returns {"repo_id", "freed_bytes"}. "hf_cache_path" names
 * the cache to use instead of the environment's (HF_HUB_CACHE, else HF_HOME/hub). */
INFERENCE_API inference_status inference_model_inspect(const char *request, size_t request_len,
                                                      inference_string **out_response);
INFERENCE_API inference_status inference_models_scan(const char *request, size_t request_len,
                                                    inference_string **out_response);
INFERENCE_API inference_status inference_hf_cache_delete(const char *request, size_t request_len,
                                                        inference_string **out_response);
/* Starts downloading a Hugging Face repository into the cache: {"repo_id", "revision"? ("main"), "files"? (these
 * files), "quant"? (a GGUF quantization the repository publishes, "Q4_K_M" or "4": its files and the projector
 * beside it), "token_source"? ("cache" | "none" | "literal:<token>" | "env:<variable>"), "hf_cache_path"?}; with neither "files" nor
 * "quant" it downloads the whole repository. Poll the stream with inference_stream_next: "files" (data: the file
 * names), "progress" (data: {"file", "done", "total"} in bytes, about once a percent; a file already cached reports
 * once, complete), then a last "done" (data: {"repo_id", "revision", "files"}, the local paths), "canceled" or
 * "error" (data: the error envelope). inference_stream_cancel or inference_stream_free stops the download; a
 * partly downloaded file resumes on the next attempt. */
INFERENCE_API inference_status inference_model_download_open(const char *request, size_t request_len,
                                                            inference_stream **out_stream);

/* The blob's bytes; valid until the blob is freed. NULL for NULL. */
INFERENCE_API const uint8_t *inference_blob_data(const inference_blob *blob);
/* Length in bytes. 0 for NULL. */
INFERENCE_API size_t inference_blob_len(const inference_blob *blob);
/* The MIME type, NUL-terminated; valid until the blob is freed. "" for NULL. */
INFERENCE_API const char *inference_blob_mime_type(const inference_blob *blob);
INFERENCE_API void inference_blob_free(inference_blob *blob);

/* The string's bytes, NUL-terminated; valid until the string is freed. "" for NULL. */
INFERENCE_API const char *inference_string_data(const inference_string *string);
/* Length in bytes, excluding the terminating NUL. 0 for NULL. */
INFERENCE_API size_t inference_string_len(const inference_string *string);
INFERENCE_API void inference_string_free(inference_string *string);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
