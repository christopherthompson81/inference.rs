//! The engine surface: load an engine from a JSON spec and run OpenAI-style chat completions on it.

use std::{
    ffi::{CString, c_char},
    time::Duration,
};

use inference_api::{
    Engine, EngineLoadError,
    anthropic::anthropic_error_body,
    api_error::{ApiError, ApiErrorKind},
    blocking::{BlockingEngine, BlockingStream, StreamPoll},
    engine_chat::RequestCancellation,
    files::{FileUpload, MAX_FILE_UPLOAD_BYTES, file_too_large},
    media_source::{MediaAttachment, MediaAttachments},
    skill_store::{SkillFiles, skill_api_error},
    system::{system_doctor_json, system_info_json, tune_model_json},
};

use crate::{
    Failure, FfiResult,
    callbacks::{engine_callbacks, inference_host_callbacks},
    guard, guard_value, inference_status,
    inference_status::{
        INFERENCE_ERR_INVALID_REQUEST, INFERENCE_ERR_LOAD_FAILED, INFERENCE_ERR_NOT_AVAILABLE,
        INFERENCE_ERR_NOT_FOUND, INFERENCE_ERR_RUNTIME, INFERENCE_ERR_UNAVAILABLE,
    },
};

/// Mirrors `inference_skill_file`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct inference_skill_file {
    pub path: *const c_char,
    pub data: *const u8,
    pub len: usize,
}

/// Mirrors `inference_media`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct inference_media {
    pub data: *const u8,
    pub len: usize,
    pub mime_type: *const c_char,
}

/// Opaque; mirrors `inference_engine`.
#[allow(non_camel_case_types)]
pub struct inference_engine {
    engine: BlockingEngine,
}

/// Opaque; mirrors `inference_stream`.
#[allow(non_camel_case_types)]
pub struct inference_stream {
    stream: BlockingStream,
    done: bool,
    // Apart from `stream` so a cancel on another thread never borrows what a blocked poll holds.
    cancellation: RequestCancellation,
}

/// Opaque; mirrors `inference_string`.
#[allow(non_camel_case_types)]
pub struct inference_string {
    text: CString,
}

/// Opaque; mirrors `inference_blob`.
#[allow(non_camel_case_types)]
pub struct inference_blob {
    bytes: Vec<u8>,
    mime_type: CString,
}

// The header promises engines may be shared across threads and streams moved between them.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    const fn assert_send<T: Send>() {}
    assert_send_sync::<BlockingEngine>();
    assert_send::<BlockingStream>();
};

fn api_status(error: &ApiError) -> inference_status {
    match error.kind {
        ApiErrorKind::Internal => INFERENCE_ERR_RUNTIME,
        ApiErrorKind::NotFound | ApiErrorKind::Gone => INFERENCE_ERR_NOT_FOUND,
        ApiErrorKind::RateLimited | ApiErrorKind::Unavailable | ApiErrorKind::Overloaded => {
            INFERENCE_ERR_UNAVAILABLE
        }
        _ => INFERENCE_ERR_INVALID_REQUEST,
    }
}

fn api_failure(error: ApiError) -> Failure {
    Failure::new(api_status(&error), error.to_openai_body().to_string())
}

// Same statuses as `api_failure`, with the Anthropic error envelope, for calls made in the Anthropic protocol.
fn anthropic_failure(error: ApiError) -> Failure {
    Failure::new(api_status(&error), anthropic_error_body(&error).to_string())
}

fn load_failure(error: EngineLoadError) -> Failure {
    match error {
        EngineLoadError::InvalidSpec(_) => Failure::invalid(error.to_string()),
        EngineLoadError::Unavailable(_) => {
            Failure::new(INFERENCE_ERR_NOT_AVAILABLE, error.to_string())
        }
        EngineLoadError::Load(_) => Failure::new(INFERENCE_ERR_LOAD_FAILED, error.to_string()),
    }
}

/// Safety: `data` is NULL (rejected) or valid for `len` bytes.
pub(crate) unsafe fn arg_bytes<'a>(
    data: *const c_char,
    len: usize,
    name: &str,
) -> FfiResult<&'a [u8]> {
    unsafe {
        if data.is_null() {
            return Err(Failure::invalid(format!("{name} is NULL")));
        }
        Ok(std::slice::from_raw_parts(data.cast::<u8>(), len))
    }
}

unsafe fn out_arg<T>(out: *mut *mut T, name: &str) -> FfiResult<()> {
    unsafe {
        if out.is_null() {
            return Err(Failure::invalid(format!("{name} is NULL")));
        }
        out.write(std::ptr::null_mut());
        Ok(())
    }
}

/// Safety: `media` is NULL (rejected unless `count` is 0) or valid for `count` entries, each `data` valid for `len`
/// bytes and each `mime_type` NULL or a C string.
unsafe fn arg_media(media: *const inference_media, count: usize) -> FfiResult<MediaAttachments> {
    unsafe {
        if count == 0 {
            return Ok(MediaAttachments::default());
        }
        if media.is_null() {
            return Err(Failure::invalid("media is NULL but media_count is not 0"));
        }
        let attachments = std::slice::from_raw_parts(media, count)
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let name = format!("media[{index}].data");
                let bytes = arg_bytes(item.data.cast::<c_char>(), item.len, &name)?.to_vec();
                let mime_type = (!item.mime_type.is_null())
                    .then(|| {
                        crate::arg_str(item.mime_type, &format!("media[{index}].mime_type"))
                            .map(str::to_string)
                    })
                    .transpose()?;
                Ok(MediaAttachment { bytes, mime_type })
            })
            .collect::<FfiResult<Vec<_>>>()?;
        Ok(MediaAttachments::new(attachments))
    }
}

fn blob_handle(bytes: Vec<u8>, mime_type: String) -> *mut inference_blob {
    // MIME types never contain NUL; dropping any keeps the C string intact rather than failing the call
    let mime_type = CString::new(mime_type.replace('\0', "")).unwrap_or_default();
    Box::into_raw(Box::new(inference_blob { bytes, mime_type }))
}

fn string_handle(text: String) -> *mut inference_string {
    // JSON text never contains NUL; dropping any keeps the C string intact rather than failing the call
    let text = CString::new(text.replace('\0', "")).unwrap_or_default();
    Box::into_raw(Box::new(inference_string { text }))
}

/// Safety: `spec` is valid for `spec_len` bytes and `out_engine` for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_engine_load(
    spec: *const c_char,
    spec_len: usize,
    out_engine: *mut *mut inference_engine,
) -> inference_status {
    unsafe { inference_engine_load_with_callbacks(spec, spec_len, std::ptr::null(), out_engine) }
}

/// Safety: as for `inference_engine_load`, with `callbacks` NULL or valid (see `engine_callbacks`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_engine_load_with_callbacks(
    spec: *const c_char,
    spec_len: usize,
    callbacks: *const inference_host_callbacks,
    out_engine: *mut *mut inference_engine,
) -> inference_status {
    unsafe {
        guard(|| {
            out_arg(out_engine, "out_engine")?;
            let spec = arg_bytes(spec, spec_len, "spec")?;
            let callbacks = engine_callbacks(callbacks)?;
            let engine = BlockingEngine::load_json(spec, callbacks).map_err(load_failure)?;
            out_engine.write(Box::into_raw(Box::new(inference_engine { engine })));
            Ok(())
        })
    }
}

/// Safety: `engine` is NULL or a handle from `inference_engine_load*` or `inference_engine_for_owner`, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_engine_free(engine: *mut inference_engine) {
    unsafe {
        if !engine.is_null() {
            guard_value((), || drop(Box::from_raw(engine)));
        }
    }
}

/// Safety: `engine` is a live handle, `owner` valid for `owner_len` bytes, `out_engine` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_engine_for_owner(
    engine: *const inference_engine,
    owner: *const c_char,
    owner_len: usize,
    out_engine: *mut *mut inference_engine,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (owner, owner_len, "owner"),
            (out_engine, "out_engine"),
            |engine, owner| {
                if owner.is_empty() {
                    return Err(Failure::invalid("owner is empty"));
                }
                let engine = engine.for_owner(owner);
                Ok(Box::into_raw(Box::new(inference_engine { engine })))
            },
        )
    }
}

/// Safety: `engine` is a live handle, `request` valid for `request_len` bytes, `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_chat(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        inference_chat_with_media(
            engine,
            request,
            request_len,
            std::ptr::null(),
            0,
            out_response,
        )
    }
}

/// Safety: as for `inference_chat`, with `media` valid for `media_count` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_chat_with_media(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    media: *const inference_media,
    media_count: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                let media = arg_media(media, media_count)?;
                engine.chat_json(request, media).map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_chat`, with `out_stream` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_chat_stream_open(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    unsafe {
        inference_chat_stream_open_with_media(
            engine,
            request,
            request_len,
            std::ptr::null(),
            0,
            out_stream,
        )
    }
}

/// Safety: as for `inference_chat_stream_open`, with `media` valid for `media_count` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_chat_stream_open_with_media(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    media: *const inference_media,
    media_count: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    unsafe {
        stream_call(
            engine,
            request,
            request_len,
            out_stream,
            |engine, request| {
                let media = arg_media(media, media_count)?;
                engine.chat_stream_json(request, media).map_err(api_failure)
            },
        )
    }
}

// Every stream opener has the same shape: engine and request in, an owned stream handle out.
unsafe fn stream_call(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
    open: impl FnOnce(&BlockingEngine, &[u8]) -> FfiResult<BlockingStream>,
) -> inference_status {
    unsafe {
        engine_call(
            engine,
            (request, request_len, "request"),
            (out_stream, "out_stream"),
            |engine, request| {
                let stream = open(engine, request)?;
                Ok(Box::into_raw(Box::new(inference_stream {
                    cancellation: stream.cancellation(),
                    stream,
                    done: false,
                })))
            },
        )
    }
}

// Every blocking operation has the same shape: engine and request in, an owned handle out.
unsafe fn engine_call<H>(
    engine: *const inference_engine,
    input: (*const c_char, usize, &str),
    out: (*mut *mut H, &str),
    call: impl FnOnce(&BlockingEngine, &[u8]) -> FfiResult<*mut H>,
) -> inference_status {
    unsafe {
        let ((input, input_len, input_name), (out, out_name)) = (input, out);
        guard(|| {
            out_arg(out, out_name)?;
            let engine = engine
                .as_ref()
                .ok_or_else(|| Failure::invalid("engine is NULL"))?;
            let request = arg_bytes(input, input_len, input_name)?;
            out.write(call(&engine.engine, request)?);
            Ok(())
        })
    }
}

unsafe fn json_call(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
    call: impl FnOnce(&BlockingEngine, &[u8]) -> FfiResult<String>,
) -> inference_status {
    unsafe {
        engine_call(
            engine,
            (request, request_len, "request"),
            (out_response, "out_response"),
            |engine, request| call(engine, request).map(string_handle),
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_completion(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.completion_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat_stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_completion_stream_open(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    unsafe {
        stream_call(
            engine,
            request,
            request_len,
            out_stream,
            |engine, request| engine.completion_stream_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_embeddings(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.embeddings_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_anthropic_messages(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                engine
                    .anthropic_messages_json(request)
                    .map_err(anthropic_failure)
            },
        )
    }
}

/// Safety: as for `inference_chat_stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_anthropic_messages_stream_open(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    unsafe {
        stream_call(
            engine,
            request,
            request_len,
            out_stream,
            |engine, request| {
                engine
                    .anthropic_messages_stream_json(request)
                    .map_err(anthropic_failure)
            },
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_responses_create(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.responses_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat_stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_responses_stream_open(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    unsafe {
        stream_call(
            engine,
            request,
            request_len,
            out_stream,
            |engine, request| engine.responses_stream_json(request).map_err(api_failure),
        )
    }
}

// Engine and a UTF-8 id in, an owned handle out.
unsafe fn id_call<H>(
    engine: *const inference_engine,
    id: (*const c_char, usize, &str),
    out: (*mut *mut H, &str),
    call: impl FnOnce(&BlockingEngine, &str) -> FfiResult<*mut H>,
) -> inference_status {
    unsafe {
        let name = id.2;
        engine_call(engine, id, out, |engine, id| call(engine, utf8(id, name)?))
    }
}

unsafe fn response_id_call(
    engine: *const inference_engine,
    response_id: *const c_char,
    response_id_len: usize,
    out_response: *mut *mut inference_string,
    call: impl FnOnce(&Engine, &str) -> Result<String, ApiError>,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (response_id, response_id_len, "response_id"),
            (out_response, "out_response"),
            |engine, id| {
                call(engine.engine(), id)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: `engine` is a live handle, `response_id` valid for `response_id_len` bytes, `out_response` valid for a
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_responses_get(
    engine: *const inference_engine,
    response_id: *const c_char,
    response_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        response_id_call(
            engine,
            response_id,
            response_id_len,
            out_response,
            Engine::response_json,
        )
    }
}

/// Safety: as for `inference_responses_get`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_responses_delete(
    engine: *const inference_engine,
    response_id: *const c_char,
    response_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        response_id_call(
            engine,
            response_id,
            response_id_len,
            out_response,
            Engine::delete_response_json,
        )
    }
}

/// Safety: as for `inference_responses_get`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_responses_cancel(
    engine: *const inference_engine,
    response_id: *const c_char,
    response_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        response_id_call(
            engine,
            response_id,
            response_id_len,
            out_response,
            Engine::cancel_response_json,
        )
    }
}

/// Safety: `engine` is a live handle and `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_models_list(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe { query_call(engine, out_response, |engine| engine.engine().models_json()) }
}

// An engine call with no request body.
unsafe fn query_call(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
    call: impl FnOnce(&BlockingEngine) -> Result<String, ApiError>,
) -> inference_status {
    unsafe {
        guard(|| {
            out_arg(out_response, "out_response")?;
            let engine = engine
                .as_ref()
                .ok_or_else(|| Failure::invalid("engine is NULL"))?;
            let response = call(&engine.engine).map_err(api_failure)?;
            out_response.write(string_handle(response));
            Ok(())
        })
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_model_unload(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                engine
                    .engine()
                    .unload_model_json(request)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_model_reload(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.reload_model_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_model_status(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                engine
                    .engine()
                    .model_status_json(request)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_lora_adapters_list(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.lora_adapters_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_lora_adapter_load(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.load_lora_adapter_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_lora_adapter_unload(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                engine
                    .unload_lora_adapter_json(request)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_image_generation(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.image_generation_json(request).map_err(api_failure),
        )
    }
}

/// Safety: `engine` is a live handle, `request` valid for `request_len` bytes, `out_blob` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_speech_generation(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_blob: *mut *mut inference_blob,
) -> inference_status {
    unsafe {
        engine_call(
            engine,
            (request, request_len, "request"),
            (out_blob, "out_blob"),
            |engine, request| {
                let audio = engine
                    .speech_generation_json(request)
                    .map_err(api_failure)?;
                Ok(blob_handle(audio.bytes, audio.content_type))
            },
        )
    }
}

/// Safety: `engine` is a live handle, `approval_id` valid for `approval_id_len` bytes, `request` for `request_len`
/// bytes and `out_response` for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_approval_resolve(
    engine: *const inference_engine,
    approval_id: *const c_char,
    approval_id_len: usize,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                let id = arg_bytes(approval_id, approval_id_len, "approval_id")?;
                let id = std::str::from_utf8(id)
                    .map_err(|_| Failure::invalid("approval_id is not UTF-8"))?;
                engine
                    .engine()
                    .resolve_approval_json(id, request)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: `engine` is a live handle, `data` valid for `len` bytes, `filename` and `purpose` C strings, `mime_type`
/// NULL or a C string, `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_file_upload(
    engine: *const inference_engine,
    data: *const u8,
    len: usize,
    filename: *const c_char,
    mime_type: *const c_char,
    purpose: *const c_char,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        engine_call(
            engine,
            (data.cast::<c_char>(), len, "data"),
            (out_response, "out_response"),
            |engine, bytes| {
                // Checked before the copy, so an oversized buffer is refused without being duplicated.
                if bytes.len() > MAX_FILE_UPLOAD_BYTES {
                    return Err(api_failure(file_too_large()));
                }
                let upload = FileUpload {
                    filename: crate::arg_str(filename, "filename")?.to_string(),
                    mime_type: (!mime_type.is_null())
                        .then(|| crate::arg_str(mime_type, "mime_type").map(str::to_string))
                        .transpose()?,
                    purpose: crate::arg_str(purpose, "purpose")?.to_string(),
                    bytes: bytes.to_vec(),
                };
                engine
                    .engine()
                    .upload_file_json(upload)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: `engine` is a live handle and `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_files_list(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe { query_call(engine, out_response, |engine| engine.engine().files_json()) }
}

unsafe fn file_id_call(
    engine: *const inference_engine,
    file_id: *const c_char,
    file_id_len: usize,
    out_response: *mut *mut inference_string,
    call: impl FnOnce(&Engine, &str) -> Result<String, ApiError>,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (file_id, file_id_len, "file_id"),
            (out_response, "out_response"),
            |engine, id| {
                call(engine.engine(), id)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: `engine` is a live handle, `file_id` valid for `file_id_len` bytes, `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_file_get(
    engine: *const inference_engine,
    file_id: *const c_char,
    file_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        file_id_call(
            engine,
            file_id,
            file_id_len,
            out_response,
            Engine::file_json,
        )
    }
}

/// Safety: as for `inference_file_get`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_file_delete(
    engine: *const inference_engine,
    file_id: *const c_char,
    file_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        file_id_call(
            engine,
            file_id,
            file_id_len,
            out_response,
            Engine::delete_file_json,
        )
    }
}

/// Safety: as for `inference_file_get`, with `out_blob` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_file_content(
    engine: *const inference_engine,
    file_id: *const c_char,
    file_id_len: usize,
    out_blob: *mut *mut inference_blob,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (file_id, file_id_len, "file_id"),
            (out_blob, "out_blob"),
            |engine, id| {
                let body = engine.engine().file_content(id).map_err(api_failure)?;
                Ok(blob_handle(body.bytes, body.mime_type))
            },
        )
    }
}

/// Safety: `files` is NULL (rejected unless `count` is 0) or valid for `count` entries, each `path` a C string and
/// `data` valid for `len` bytes.
unsafe fn arg_skill_files(
    files: *const inference_skill_file,
    count: usize,
) -> FfiResult<SkillFiles> {
    unsafe {
        if files.is_null() && count != 0 {
            return Err(Failure::invalid("files is NULL but file_count is not 0"));
        }
        let mut skill_files = SkillFiles::default();
        for (index, file) in (0..count).map(|index| (index, &*files.add(index))) {
            let path = crate::arg_str(file.path, &format!("files[{index}].path"))?.to_string();
            let bytes = arg_bytes(
                file.data.cast::<c_char>(),
                file.len,
                &format!("files[{index}].data"),
            )?;
            skill_files
                .push(path, bytes.to_vec())
                .map_err(|error| api_failure(skill_api_error(error)))?;
        }
        Ok(skill_files)
    }
}

/// Safety: `engine` is a live handle, `files` as for `arg_skill_files`, `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_skill_upload(
    engine: *const inference_engine,
    files: *const inference_skill_file,
    file_count: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        guard(|| {
            out_arg(out_response, "out_response")?;
            let engine = engine
                .as_ref()
                .ok_or_else(|| Failure::invalid("engine is NULL"))?;
            let files = arg_skill_files(files, file_count)?;
            let response = engine
                .engine
                .engine()
                .upload_skill_json(files)
                .map_err(api_failure)?;
            out_response.write(string_handle(response));
            Ok(())
        })
    }
}

/// Safety: as for `inference_skill_upload`, with `skill_id` valid for `skill_id_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_skill_version_upload(
    engine: *const inference_engine,
    skill_id: *const c_char,
    skill_id_len: usize,
    files: *const inference_skill_file,
    file_count: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (skill_id, skill_id_len, "skill_id"),
            (out_response, "out_response"),
            |engine, id| {
                let files = arg_skill_files(files, file_count)?;
                engine
                    .engine()
                    .upload_skill_version_json(id, files)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: `engine` is a live handle and `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_skills_list(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        guard(|| {
            out_arg(out_response, "out_response")?;
            let engine = engine
                .as_ref()
                .ok_or_else(|| Failure::invalid("engine is NULL"))?;
            let response = engine.engine.engine().skills_json().map_err(api_failure)?;
            out_response.write(string_handle(response));
            Ok(())
        })
    }
}

/// Safety: `engine` is a live handle, `skill_id` valid for `skill_id_len` bytes, `out_response` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_skill_versions_list(
    engine: *const inference_engine,
    skill_id: *const c_char,
    skill_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (skill_id, skill_id_len, "skill_id"),
            (out_response, "out_response"),
            |engine, id| {
                engine
                    .engine()
                    .skill_versions_json(id)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

unsafe fn report(
    out_response: *mut *mut inference_string,
    report: fn() -> Result<String, ApiError>,
) -> inference_status {
    unsafe {
        guard(|| {
            out_arg(out_response, "out_response")?;
            out_response.write(string_handle(report().map_err(api_failure)?));
            Ok(())
        })
    }
}

/// Safety: `out_response` is valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_system_info(
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe { report(out_response, system_info_json) }
}

/// Safety: `out_response` is valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_system_doctor(
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe { report(out_response, system_doctor_json) }
}

/// Safety: `blob` is NULL or a live blob handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_blob_data(blob: *const inference_blob) -> *const u8 {
    unsafe {
        guard_value(std::ptr::null(), || {
            blob.as_ref()
                .map_or(std::ptr::null(), |blob| blob.bytes.as_ptr())
        })
    }
}

/// Safety: `blob` is NULL or a live blob handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_blob_len(blob: *const inference_blob) -> usize {
    unsafe { guard_value(0, || blob.as_ref().map_or(0, |blob| blob.bytes.len())) }
}

/// Safety: `blob` is NULL or a live blob handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_blob_mime_type(blob: *const inference_blob) -> *const c_char {
    unsafe {
        guard_value(c"".as_ptr(), || {
            blob.as_ref()
                .map_or(c"".as_ptr(), |blob| blob.mime_type.as_ptr())
        })
    }
}

/// Safety: `blob` is NULL or a blob handle that is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_blob_free(blob: *mut inference_blob) {
    unsafe {
        if !blob.is_null() {
            guard_value((), || drop(Box::from_raw(blob)));
        }
    }
}

/// Safety: `stream` is a live handle not used concurrently; `out_event` and `out_done` are valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_stream_next(
    stream: *mut inference_stream,
    timeout_ms: i64,
    out_event: *mut *mut inference_string,
    out_done: *mut i32,
) -> inference_status {
    unsafe {
        guard(|| {
            if out_done.is_null() {
                return Err(Failure::invalid("out_done is NULL"));
            }
            out_done.write(0);
            out_arg(out_event, "out_event")?;
            if stream.is_null() {
                return Err(Failure::invalid("stream is NULL"));
            }
            // Borrows only the fields a poll uses, so `inference_stream_cancel` may read `cancellation` meanwhile.
            let (poller, done) = (&mut (*stream).stream, &mut (*stream).done);
            if *done {
                out_done.write(1);
                return Ok(());
            }
            let timeout = u64::try_from(timeout_ms).ok().map(Duration::from_millis);
            match poller.next(timeout) {
                StreamPoll::Event(event) => out_event.write(string_handle(event)),
                StreamPoll::Timeout => {}
                StreamPoll::Done => {
                    *done = true;
                    out_done.write(1);
                }
            }
            Ok(())
        })
    }
}

/// Safety: `stream` is a live handle; it may be called while another thread is in `inference_stream_next` on it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_stream_cancel(
    stream: *const inference_stream,
) -> inference_status {
    unsafe {
        guard(|| {
            if stream.is_null() {
                return Err(Failure::invalid("stream is NULL"));
            }
            (*stream).cancellation.cancel();
            Ok(())
        })
    }
}

/// Safety: `stream` is NULL or a handle from `inference_chat_stream_open` that is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_stream_free(stream: *mut inference_stream) {
    unsafe {
        if !stream.is_null() {
            guard_value((), || drop(Box::from_raw(stream)));
        }
    }
}

/// Safety: `string` is NULL or a live string handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_string_data(string: *const inference_string) -> *const c_char {
    unsafe {
        guard_value(c"".as_ptr(), || {
            string
                .as_ref()
                .map_or(c"".as_ptr(), |string| string.text.as_ptr())
        })
    }
}

/// Safety: `string` is NULL or a live string handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_string_len(string: *const inference_string) -> usize {
    unsafe {
        guard_value(0, || {
            string
                .as_ref()
                .map_or(0, |string| string.text.as_bytes().len())
        })
    }
}

/// Safety: `string` is NULL or a string handle that is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_string_free(string: *mut inference_string) {
    unsafe {
        if !string.is_null() {
            guard_value((), || drop(Box::from_raw(string)));
        }
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_re_isq(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.re_isq_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_models_list`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_calibration_start(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe { query_call(engine, out_response, BlockingEngine::calibration_start_json) }
}

/// Safety: as for `inference_models_list`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_models_cache_stats(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe { query_call(engine, out_response, BlockingEngine::cache_stats_json) }
}

/// Safety: as for `inference_models_list`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_calibration_status(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        query_call(
            engine,
            out_response,
            BlockingEngine::calibration_status_json,
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_calibration_apply(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.calibration_apply_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_models_list`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_sessions_list(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        query_call(engine, out_response, |engine| {
            engine.engine().sessions_json()
        })
    }
}

unsafe fn session_id_call(
    engine: *const inference_engine,
    session_id: *const c_char,
    session_id_len: usize,
    out_response: *mut *mut inference_string,
    call: impl FnOnce(&Engine, &str) -> Result<String, ApiError>,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (session_id, session_id_len, "session_id"),
            (out_response, "out_response"),
            |engine, id| {
                call(engine.engine(), id)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: `engine` is a live handle, `session_id` valid for `session_id_len` bytes, `out_response` valid for a
/// write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_session_get(
    engine: *const inference_engine,
    session_id: *const c_char,
    session_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        session_id_call(
            engine,
            session_id,
            session_id_len,
            out_response,
            Engine::session_json,
        )
    }
}

/// Safety: as for `inference_approval_resolve`, with `session_id` in place of `approval_id`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_session_put(
    engine: *const inference_engine,
    session_id: *const c_char,
    session_id_len: usize,
    session: *const c_char,
    session_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            session,
            session_len,
            out_response,
            |engine, session| {
                let id = arg_bytes(session_id, session_id_len, "session_id")?;
                let id = std::str::from_utf8(id)
                    .map_err(|_| Failure::invalid("session_id is not UTF-8"))?;
                engine
                    .engine()
                    .put_session_json(id, session)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_session_get`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_session_delete(
    engine: *const inference_engine,
    session_id: *const c_char,
    session_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        session_id_call(
            engine,
            session_id,
            session_id_len,
            out_response,
            Engine::delete_session_json,
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_tokenize(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.tokenize_json(request).map_err(api_failure),
        )
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_detokenize(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.detokenize_json(request).map_err(api_failure),
        )
    }
}

fn utf8<'a>(bytes: &'a [u8], name: &str) -> FfiResult<&'a str> {
    std::str::from_utf8(bytes).map_err(|_| Failure::invalid(format!("{name} is not UTF-8")))
}

// Safety: `data` is valid for `len` bytes.
unsafe fn arg_utf8<'a>(data: *const c_char, len: usize, name: &str) -> FfiResult<&'a str> {
    unsafe { utf8(arg_bytes(data, len, name)?, name) }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_anthropic_count_tokens(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| engine.count_tokens_json(request).map_err(anthropic_failure),
        )
    }
}

/// Safety: as for `inference_file_get`, with `container_id` in place of `file_id`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_container_files_list(
    engine: *const inference_engine,
    container_id: *const c_char,
    container_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (container_id, container_id_len, "container_id"),
            (out_response, "out_response"),
            |engine, id| {
                engine
                    .engine()
                    .container_files_json(id)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_container_files_list`, and `file_id` valid for `file_id_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_container_file_get(
    engine: *const inference_engine,
    container_id: *const c_char,
    container_id_len: usize,
    file_id: *const c_char,
    file_id_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (container_id, container_id_len, "container_id"),
            (out_response, "out_response"),
            |engine, container_id| {
                let file_id = arg_utf8(file_id, file_id_len, "file_id")?;
                engine
                    .engine()
                    .container_file_json(container_id, file_id)
                    .map(string_handle)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_container_file_get`, with `out_blob` valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_container_file_content(
    engine: *const inference_engine,
    container_id: *const c_char,
    container_id_len: usize,
    file_id: *const c_char,
    file_id_len: usize,
    out_blob: *mut *mut inference_blob,
) -> inference_status {
    unsafe {
        id_call(
            engine,
            (container_id, container_id_len, "container_id"),
            (out_blob, "out_blob"),
            |engine, container_id| {
                let file_id = arg_utf8(file_id, file_id_len, "file_id")?;
                let body = engine
                    .engine()
                    .container_file_content(container_id, file_id)
                    .map_err(api_failure)?;
                Ok(blob_handle(body.bytes, body.mime_type))
            },
        )
    }
}

/// Safety: as for `inference_session_put`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_session_fork(
    engine: *const inference_engine,
    session_id: *const c_char,
    session_id_len: usize,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                let id = arg_utf8(session_id, session_id_len, "session_id")?;
                engine
                    .engine()
                    .fork_session_json(id, request)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: as for `inference_models_list`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_mcp_tools_list(
    engine: *const inference_engine,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        query_call(engine, out_response, |engine| {
            engine.engine().mcp_tools_json()
        })
    }
}

/// Safety: as for `inference_chat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_model_served(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        json_call(
            engine,
            request,
            request_len,
            out_response,
            |engine, request| {
                engine
                    .engine()
                    .model_served_json(request)
                    .map_err(api_failure)
            },
        )
    }
}

/// Safety: `request` is valid for `request_len` bytes and `out_response` for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_model_tune(
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    unsafe {
        guard(|| {
            out_arg(out_response, "out_response")?;
            let request = arg_bytes(request, request_len, "request")?;
            out_response.write(string_handle(
                tune_model_json(request).map_err(api_failure)?,
            ));
            Ok(())
        })
    }
}
