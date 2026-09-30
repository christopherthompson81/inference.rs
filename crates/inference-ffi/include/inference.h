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
#define INFERENCE_ABI_VERSION_PATCH 11

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
 * Failing engine calls, including INFERENCE_ERR_RUNTIME ones, leave the error JSON in inference_last_error(): the
 * OpenAI envelope, or the Anthropic one for the inference_anthropic_* calls.
 * Freeing the last handle of an engine (the engine or one of its streams) waits up to 10 s for the engine to stop.
 */

typedef struct inference_engine inference_engine;
typedef struct inference_stream inference_stream;
/* An owned, NUL-terminated UTF-8 JSON string. */
typedef struct inference_string inference_string;
/* Owned bytes (audio, a file's content) and their MIME type. */
typedef struct inference_blob inference_blob;

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

/* tools may be NULL when tool_count is 0; search may be NULL to keep the built-in search. */
typedef struct inference_host_callbacks {
    const inference_host_tool *tools;
    size_t tool_count;
    inference_search_callback search;
    void *search_user_data;
} inference_host_callbacks;

/* inference_engine_load with host callbacks; callbacks may be NULL. A malformed tool is
 * INFERENCE_ERR_INVALID_ARGUMENT. */
INFERENCE_API inference_status inference_engine_load_with_callbacks(const char *spec, size_t spec_len,
                                                                   const inference_host_callbacks *callbacks,
                                                                   inference_engine **out_engine);

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
 * "chunk" (data: a text_completion chunk) and "error"; Anthropic and Responses streams emit their protocol's events
 * (see inference_anthropic_messages_stream_open and inference_responses_stream_open). An error's data is the error
 * JSON of the stream's protocol, and an error event is always the last event. On a timeout out_event is NULL and
 * out_done 0. Once the stream has ended, out_event is NULL and out_done 1. out_event and out_done are required. */
INFERENCE_API inference_status inference_stream_next(inference_stream *stream, int64_t timeout_ms,
                                                    inference_string **out_event, int32_t *out_done);
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

/* Runs an Anthropic Messages request (the POST /v1/messages body) to its end; out_response receives the message JSON.
 * Failures leave the Anthropic error JSON ({"type": "error", "error": {"type", "message"}}) in inference_last_error().
 * "stream" in the request is ignored. */
INFERENCE_API inference_status inference_anthropic_messages(const inference_engine *engine, const char *request,
                                                           size_t request_len, inference_string **out_response);
/* Starts a streaming Messages request; poll it with inference_stream_next. Its events are the Anthropic stream events
 * (message_start, content_block_start/delta/stop, message_delta, message_stop, error), agentic_tool_call_progress and
 * file_produced. */
INFERENCE_API inference_status inference_anthropic_messages_stream_open(const inference_engine *engine,
                                                                       const char *request, size_t request_len,
                                                                       inference_stream **out_stream);

/* Runs a Responses request (the POST /v1/responses body); out_response receives the response resource JSON. A request
 * with "background": true returns at once with status "queued"; follow it with inference_responses_get. "stream" in
 * the request is ignored. Responses, streamed or not, are stored (unless "store" is false) for inference_responses_get
 * and "previous_response_id"; the store is shared by every engine in the process and lives until the process exits
 * or the response is deleted. */
INFERENCE_API inference_status inference_responses_create(const inference_engine *engine, const char *request,
                                                         size_t request_len, inference_string **out_response);
/* Starts a streaming Responses request; poll it with inference_stream_next. Its events are the OpenResponses stream
 * events named by their "type" (response.created, response.in_progress, response.output_item.added/done,
 * response.content_part.added/done, response.output_text.delta, response.reasoning_text.delta/done,
 * response.function_call_arguments.delta/done, response.completed, response.failed, error), plus
 * agentic_tool_call_progress and file_produced. "background" cannot be streamed. */
INFERENCE_API inference_status inference_responses_stream_open(const inference_engine *engine, const char *request,
                                                              size_t request_len, inference_stream **out_stream);
/* A background response in its current state, or a stored one; an unknown id is INFERENCE_ERR_NOT_FOUND. */
INFERENCE_API inference_status inference_responses_get(const inference_engine *engine, const char *response_id,
                                                      size_t response_id_len, inference_string **out_response);
/* Forgets a response; out_response receives {"id", "object": "response.deleted", "deleted": true}. */
INFERENCE_API inference_status inference_responses_delete(const inference_engine *engine, const char *response_id,
                                                         size_t response_id_len, inference_string **out_response);
/* Cancels a queued or running background response and returns it; a finished one comes back unchanged. */
INFERENCE_API inference_status inference_responses_cancel(const inference_engine *engine, const char *response_id,
                                                         size_t response_id_len, inference_string **out_response);

/* The served models (the GET /v1/models body): the "default" alias, each model with its status, and each loaded LoRA
 * adapter as a model of its own. */
INFERENCE_API inference_status inference_models_list(const inference_engine *engine, inference_string **out_response);
/* Unloads, reloads or reports a model; the request is {"model_id"} and out_response receives {"model_id", "status":
 * "loaded" | "unloaded" | "reloading"}. Unloading an unloaded model or reloading a loaded one succeeds. An unknown
 * model is INFERENCE_ERR_NOT_FOUND. */
INFERENCE_API inference_status inference_model_unload(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_model_reload(const inference_engine *engine, const char *request,
                                                     size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_model_status(const inference_engine *engine, const char *request,
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
/* Speaks text with a speech model (the POST /v1/audio/speech body; "response_format" is "wav" or "pcm"). out_blob
 * receives the encoded audio; its MIME type carries the sample rate and channel count, e.g.
 * "audio/pcm; codecs=1; format=s16le; rate=44100; channels=1". */
INFERENCE_API inference_status inference_speech_generation(const inference_engine *engine, const char *request,
                                                          size_t request_len, inference_blob **out_blob);
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
INFERENCE_API inference_status inference_files_list(const inference_engine *engine, inference_string **out_response);
INFERENCE_API inference_status inference_file_get(const inference_engine *engine, const char *file_id,
                                                 size_t file_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_file_delete(const inference_engine *engine, const char *file_id,
                                                    size_t file_id_len, inference_string **out_response);
INFERENCE_API inference_status inference_file_content(const inference_engine *engine, const char *file_id,
                                                     size_t file_id_len, inference_blob **out_blob);

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

/* Requantizes the loaded model, which must have loaded with ISQ, to {"ggml_type"} (an ISQ type such as "Q4K";
 * numeric shorthands resolve as on the CPU); out_response echoes it once the engine has queued the requantization
 * behind the running requests. */
INFERENCE_API inference_status inference_re_isq(const inference_engine *engine, const char *request, size_t request_len,
                                               inference_string **out_response);
/* Online calibration: start collecting activation statistics from live traffic, report per-layer progress, or apply
 * them, requantizing from the source weights and hot-swapping each layer. Apply takes {"save_cimatrix"?}, a path to
 * also save the importance matrix to, unrestricted and relative to the process's working directory. Each returns the
 * calibration status JSON, as it stood before an apply; a model without ISQ, or an apply with nothing collected, is
 * INFERENCE_ERR_INVALID_REQUEST with the reason. */
INFERENCE_API inference_status inference_calibration_start(const inference_engine *engine,
                                                          inference_string **out_response);
INFERENCE_API inference_status inference_calibration_status(const inference_engine *engine,
                                                           inference_string **out_response);
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

/* Tokenizes {"text", "add_special_tokens"?, "model"?} to {"tokens"}, and detokenizes {"tokens",
 * "skip_special_tokens"?, "model"?} to {"text"}; both flags default to true. */
INFERENCE_API inference_status inference_tokenize(const inference_engine *engine, const char *request,
                                                 size_t request_len, inference_string **out_response);
INFERENCE_API inference_status inference_detokenize(const inference_engine *engine, const char *request,
                                                   size_t request_len, inference_string **out_response);

/* Host, device and build information, and environment diagnostics (the /v1/system/info and /v1/system/doctor
 * JSON). They need no engine. */
INFERENCE_API inference_status inference_system_info(inference_string **out_response);
INFERENCE_API inference_status inference_system_doctor(inference_string **out_response);

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
