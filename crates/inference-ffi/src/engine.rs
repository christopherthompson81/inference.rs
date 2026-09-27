//! The engine surface: load an engine from a JSON spec and run OpenAI-style chat completions on it.

use std::{
    ffi::{c_char, CString},
    time::Duration,
};

use inference_api::{
    anthropic::anthropic_error_body,
    api_error::{ApiError, ApiErrorKind},
    blocking::{BlockingEngine, BlockingStream, StreamPoll},
    media_source::{MediaAttachment, MediaAttachments},
    EngineLoadError,
};

use crate::{
    guard, guard_value, inference_status,
    inference_status::{
        INFERENCE_ERR_INVALID_REQUEST, INFERENCE_ERR_LOAD_FAILED, INFERENCE_ERR_NOT_AVAILABLE,
        INFERENCE_ERR_RUNTIME, INFERENCE_ERR_UNAVAILABLE,
    },
    Failure, FfiResult,
};

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
}

/// Opaque; mirrors `inference_string`.
#[allow(non_camel_case_types)]
pub struct inference_string {
    text: CString,
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
        EngineLoadError::DeviceUnavailable(_) => {
            Failure::new(INFERENCE_ERR_NOT_AVAILABLE, error.to_string())
        }
        EngineLoadError::Load(_) => Failure::new(INFERENCE_ERR_LOAD_FAILED, error.to_string()),
    }
}

/// Safety: `data` is NULL (rejected) or valid for `len` bytes.
unsafe fn arg_bytes<'a>(data: *const c_char, len: usize, name: &str) -> FfiResult<&'a [u8]> {
    if data.is_null() {
        return Err(Failure::invalid(format!("{name} is NULL")));
    }
    Ok(std::slice::from_raw_parts(data.cast::<u8>(), len))
}

unsafe fn out_arg<T>(out: *mut *mut T, name: &str) -> FfiResult<()> {
    if out.is_null() {
        return Err(Failure::invalid(format!("{name} is NULL")));
    }
    out.write(std::ptr::null_mut());
    Ok(())
}

/// Safety: `media` is NULL (rejected unless `count` is 0) or valid for `count` entries, each `data` valid for `len`
/// bytes and each `mime_type` NULL or a C string.
unsafe fn arg_media(media: *const inference_media, count: usize) -> FfiResult<MediaAttachments> {
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

fn string_handle(text: String) -> *mut inference_string {
    // JSON text never contains NUL; dropping any keeps the C string intact rather than failing the call
    let text = CString::new(text.replace('\0', "")).unwrap_or_default();
    Box::into_raw(Box::new(inference_string { text }))
}

/// Safety: `spec` is valid for `spec_len` bytes and `out_engine` for a write.
#[no_mangle]
pub unsafe extern "C" fn inference_engine_load(
    spec: *const c_char,
    spec_len: usize,
    out_engine: *mut *mut inference_engine,
) -> inference_status {
    guard(|| {
        out_arg(out_engine, "out_engine")?;
        let spec = arg_bytes(spec, spec_len, "spec")?;
        let engine = BlockingEngine::load_json(spec).map_err(load_failure)?;
        out_engine.write(Box::into_raw(Box::new(inference_engine { engine })));
        Ok(())
    })
}

/// Safety: `engine` is NULL or a handle from `inference_engine_load` that is not used again.
#[no_mangle]
pub unsafe extern "C" fn inference_engine_free(engine: *mut inference_engine) {
    if !engine.is_null() {
        guard_value((), || drop(Box::from_raw(engine)));
    }
}

/// Safety: `engine` is a live handle, `request` valid for `request_len` bytes, `out_response` valid for a write.
#[no_mangle]
pub unsafe extern "C" fn inference_chat(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    inference_chat_with_media(
        engine,
        request,
        request_len,
        std::ptr::null(),
        0,
        out_response,
    )
}

/// Safety: as for `inference_chat`, with `media` valid for `media_count` entries.
#[no_mangle]
pub unsafe extern "C" fn inference_chat_with_media(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    media: *const inference_media,
    media_count: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
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

/// Safety: as for `inference_chat`, with `out_stream` valid for a write.
#[no_mangle]
pub unsafe extern "C" fn inference_chat_stream_open(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    inference_chat_stream_open_with_media(
        engine,
        request,
        request_len,
        std::ptr::null(),
        0,
        out_stream,
    )
}

/// Safety: as for `inference_chat_stream_open`, with `media` valid for `media_count` entries.
#[no_mangle]
pub unsafe extern "C" fn inference_chat_stream_open_with_media(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    media: *const inference_media,
    media_count: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    guard(|| {
        out_arg(out_stream, "out_stream")?;
        let engine = engine
            .as_ref()
            .ok_or_else(|| Failure::invalid("engine is NULL"))?;
        let request = arg_bytes(request, request_len, "request")?;
        let media = arg_media(media, media_count)?;
        let stream = engine
            .engine
            .chat_stream_json(request, media)
            .map_err(api_failure)?;
        out_stream.write(Box::into_raw(Box::new(inference_stream {
            stream,
            done: false,
        })));
        Ok(())
    })
}

// Every blocking JSON operation has the same shape: engine and request in, an owned response string out.
unsafe fn json_call(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
    call: impl FnOnce(&BlockingEngine, &[u8]) -> FfiResult<String>,
) -> inference_status {
    guard(|| {
        out_arg(out_response, "out_response")?;
        let engine = engine
            .as_ref()
            .ok_or_else(|| Failure::invalid("engine is NULL"))?;
        let request = arg_bytes(request, request_len, "request")?;
        let response = call(&engine.engine, request)?;
        out_response.write(string_handle(response));
        Ok(())
    })
}

/// Safety: as for `inference_chat`.
#[no_mangle]
pub unsafe extern "C" fn inference_completion(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    json_call(
        engine,
        request,
        request_len,
        out_response,
        |engine, request| engine.completion_json(request).map_err(api_failure),
    )
}

/// Safety: as for `inference_chat_stream_open`.
#[no_mangle]
pub unsafe extern "C" fn inference_completion_stream_open(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    guard(|| {
        out_arg(out_stream, "out_stream")?;
        let engine = engine
            .as_ref()
            .ok_or_else(|| Failure::invalid("engine is NULL"))?;
        let request = arg_bytes(request, request_len, "request")?;
        let stream = engine
            .engine
            .completion_stream_json(request)
            .map_err(api_failure)?;
        out_stream.write(Box::into_raw(Box::new(inference_stream {
            stream,
            done: false,
        })));
        Ok(())
    })
}

/// Safety: as for `inference_chat`.
#[no_mangle]
pub unsafe extern "C" fn inference_embeddings(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
    json_call(
        engine,
        request,
        request_len,
        out_response,
        |engine, request| engine.embeddings_json(request).map_err(api_failure),
    )
}

/// Safety: as for `inference_chat`.
#[no_mangle]
pub unsafe extern "C" fn inference_anthropic_messages(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_response: *mut *mut inference_string,
) -> inference_status {
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

/// Safety: as for `inference_chat_stream_open`.
#[no_mangle]
pub unsafe extern "C" fn inference_anthropic_messages_stream_open(
    engine: *const inference_engine,
    request: *const c_char,
    request_len: usize,
    out_stream: *mut *mut inference_stream,
) -> inference_status {
    guard(|| {
        out_arg(out_stream, "out_stream")?;
        let engine = engine
            .as_ref()
            .ok_or_else(|| Failure::invalid("engine is NULL"))?;
        let request = arg_bytes(request, request_len, "request")?;
        let stream = engine
            .engine
            .anthropic_messages_stream_json(request)
            .map_err(anthropic_failure)?;
        out_stream.write(Box::into_raw(Box::new(inference_stream {
            stream,
            done: false,
        })));
        Ok(())
    })
}

/// Safety: `stream` is a live handle not used concurrently; `out_event` and `out_done` are valid for writes.
#[no_mangle]
pub unsafe extern "C" fn inference_stream_next(
    stream: *mut inference_stream,
    timeout_ms: i64,
    out_event: *mut *mut inference_string,
    out_done: *mut i32,
) -> inference_status {
    guard(|| {
        if out_done.is_null() {
            return Err(Failure::invalid("out_done is NULL"));
        }
        out_done.write(0);
        out_arg(out_event, "out_event")?;
        let stream = stream
            .as_mut()
            .ok_or_else(|| Failure::invalid("stream is NULL"))?;
        if stream.done {
            out_done.write(1);
            return Ok(());
        }
        let timeout = u64::try_from(timeout_ms).ok().map(Duration::from_millis);
        match stream.stream.next(timeout) {
            StreamPoll::Event(event) => out_event.write(string_handle(event)),
            StreamPoll::Timeout => {}
            StreamPoll::Done => {
                stream.done = true;
                out_done.write(1);
            }
        }
        Ok(())
    })
}

/// Safety: `stream` is NULL or a handle from `inference_chat_stream_open` that is not used again.
#[no_mangle]
pub unsafe extern "C" fn inference_stream_free(stream: *mut inference_stream) {
    if !stream.is_null() {
        guard_value((), || drop(Box::from_raw(stream)));
    }
}

/// Safety: `string` is NULL or a live string handle.
#[no_mangle]
pub unsafe extern "C" fn inference_string_data(string: *const inference_string) -> *const c_char {
    guard_value(c"".as_ptr(), || {
        string
            .as_ref()
            .map_or(c"".as_ptr(), |string| string.text.as_ptr())
    })
}

/// Safety: `string` is NULL or a live string handle.
#[no_mangle]
pub unsafe extern "C" fn inference_string_len(string: *const inference_string) -> usize {
    guard_value(0, || {
        string
            .as_ref()
            .map_or(0, |string| string.text.as_bytes().len())
    })
}

/// Safety: `string` is NULL or a string handle that is not used again.
#[no_mangle]
pub unsafe extern "C" fn inference_string_free(string: *mut inference_string) {
    if !string.is_null() {
        guard_value((), || drop(Box::from_raw(string)));
    }
}
