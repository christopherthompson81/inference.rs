//! The engine C ABI end to end on a tiny random-weight PaddleOCR-VL built at test time.

use std::ffi::{c_char, CStr};
use std::ptr::{null, null_mut};

use base64::Engine as _;
use inference_ffi::engine::*;
use inference_ffi::inference_status::{self, *};
use inference_ffi::*;
use serde_json::{json, Value};

#[path = "../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

const MAX_TOKENS: usize = 6;
const PROMPT: &str = "Reply with the single word: ok";
// Long enough that the first poll never has to wait on a slow CI machine, short enough to fail a hang quickly.
const POLL_TIMEOUT_MS: i64 = 60_000;

fn last_error() -> String {
    unsafe { CStr::from_ptr(inference_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn spec(dir: &std::path::Path) -> String {
    json!({
        "model": {"MultimodalPlain": {"model_id": dir.to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    })
    .to_string()
}

fn load(spec: &str) -> (inference_status, *mut inference_engine) {
    let mut engine = null_mut();
    let status =
        unsafe { inference_engine_load(spec.as_ptr().cast::<c_char>(), spec.len(), &mut engine) };
    (status, engine)
}

fn chat_request(stream: bool) -> String {
    json!({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "stream": stream,
    })
    .to_string()
}

fn take_string(string: *mut inference_string) -> String {
    let text = unsafe {
        let len = inference_string_len(string);
        let data = inference_string_data(string);
        assert_eq!(CStr::from_ptr(data).to_bytes().len(), len);
        CStr::from_ptr(data).to_str().unwrap().to_owned()
    };
    unsafe { inference_string_free(string) };
    text
}

fn chat(engine: *const inference_engine, request: &str) -> (inference_status, Option<String>) {
    let mut response = null_mut();
    let status = unsafe {
        inference_chat(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut response,
        )
    };
    (status, (!response.is_null()).then(|| take_string(response)))
}

#[test]
fn chat_and_stream_agree_and_errors_carry_openai_bodies() {
    let dir = support::tiny_checkpoint().unwrap();
    let (status, engine) = load(&spec(dir.path()));
    assert_eq!(status, INFERENCE_OK, "{}", last_error());

    let (status, response) = chat(engine, &chat_request(false));
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    assert_eq!(last_error(), "");
    let response: Value = serde_json::from_str(&response.unwrap()).unwrap();
    assert_eq!(response["object"], "chat.completion");
    assert!(response["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    let text = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    let request = chat_request(true);
    let mut stream = null_mut();
    let status = unsafe {
        inference_chat_stream_open(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut stream,
        )
    };
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let mut streamed = String::new();
    loop {
        let (mut event, mut done) = (null_mut(), 0);
        let status =
            unsafe { inference_stream_next(stream, POLL_TIMEOUT_MS, &mut event, &mut done) };
        assert_eq!(status, INFERENCE_OK, "{}", last_error());
        if done == 1 {
            assert!(event.is_null());
            break;
        }
        assert!(!event.is_null(), "a {POLL_TIMEOUT_MS} ms poll timed out");
        let event: Value = serde_json::from_str(&take_string(event)).unwrap();
        assert_eq!(event["event"], "chunk", "{event}");
        if let Some(delta) = event["data"]["choices"][0]["delta"]["content"].as_str() {
            streamed.push_str(delta);
        }
    }
    // a finished stream stays finished
    let (mut event, mut done) = (null_mut(), 0);
    assert_eq!(
        unsafe { inference_stream_next(stream, 0, &mut event, &mut done) },
        INFERENCE_OK
    );
    assert_eq!((event.is_null(), done), (true, 1));
    unsafe { inference_stream_free(stream) };
    assert_eq!(streamed, text);

    let (status, response) = chat(engine, "{not json");
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST);
    assert!(response.is_none());
    let error: Value = serde_json::from_str(&last_error()).unwrap();
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "invalid_request_body");

    let unknown_model =
        json!({"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]});
    let (status, _) = chat(engine, &unknown_model.to_string());
    assert_eq!(status, INFERENCE_ERR_NOT_FOUND, "{}", last_error());
    let error: Value = serde_json::from_str(&last_error()).unwrap();
    assert_eq!(error["error"]["code"], "model_not_found", "{error}");
    assert_eq!(error["error"]["param"], "model", "{error}");

    // a request cannot ask for tool approvals this surface cannot answer
    let ask = json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "agent_permission": "ask",
    });
    let (status, _) = chat(engine, &ask.to_string());
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST);
    let error: Value = serde_json::from_str(&last_error()).unwrap();
    assert_eq!(error["error"]["param"], "agent_permission", "{error}");

    let mut event = null_mut();
    let mut stream = null_mut();
    let request = chat_request(true);
    assert_eq!(
        unsafe {
            inference_chat_stream_open(
                engine,
                request.as_ptr().cast::<c_char>(),
                request.len(),
                &mut stream,
            )
        },
        INFERENCE_OK
    );
    assert_eq!(
        unsafe { inference_stream_next(stream, 0, &mut event, null_mut()) },
        INFERENCE_ERR_INVALID_ARGUMENT
    );
    unsafe { inference_stream_free(stream) };

    unsafe { inference_engine_free(engine) };
}

#[test]
fn load_failures_are_classified() {
    let (status, engine) = load("{\"model\": 1}");
    assert_eq!(status, INFERENCE_ERR_INVALID_ARGUMENT);
    assert!(engine.is_null());
    assert!(
        last_error().contains("invalid engine spec"),
        "{}",
        last_error()
    );

    let ask = json!({
        "model": {"Plain": {"model_id": "org/model"}},
        "agentic": {"agent_permission": "ask"},
    });
    assert_eq!(load(&ask.to_string()).0, INFERENCE_ERR_INVALID_ARGUMENT);

    let bad_device =
        json!({"model": {"Plain": {"model_id": "org/model"}}, "runtime": {"device": "tpu:0"}});
    assert_eq!(
        load(&bad_device.to_string()).0,
        INFERENCE_ERR_INVALID_ARGUMENT
    );

    let missing = tempfile::tempdir().unwrap();
    let (status, engine) = load(&spec(&missing.path().join("absent")));
    assert_eq!(status, INFERENCE_ERR_LOAD_FAILED, "{}", last_error());
    assert!(engine.is_null());
}

#[test]
fn null_arguments_are_rejected_without_crashing() {
    let mut engine = null_mut();
    assert_eq!(
        unsafe { inference_engine_load(null(), 0, &mut engine) },
        INFERENCE_ERR_INVALID_ARGUMENT
    );
    let mut response = null_mut();
    assert_eq!(
        unsafe { inference_chat(null(), c"{}".as_ptr(), 2, &mut response) },
        INFERENCE_ERR_INVALID_ARGUMENT
    );
    assert!(response.is_null());
    unsafe {
        inference_engine_free(null_mut());
        inference_stream_free(null_mut());
        inference_string_free(null_mut());
        assert_eq!(inference_string_len(null()), 0);
        assert_eq!(
            CStr::from_ptr(inference_string_data(null())).to_bytes(),
            b""
        );
    }
    let name = unsafe {
        CStr::from_ptr(inference_status_string(
            inference_status::INFERENCE_ERR_NOT_FOUND as i32,
        ))
    };
    assert_eq!(name.to_str().unwrap(), "INFERENCE_ERR_NOT_FOUND");
}

fn image_request(url: &str) -> String {
    json!({
        "model": "default",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "OCR:"},
        ]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "logprobs": true,
        "top_logprobs": 1,
    })
    .to_string()
}

fn chat_with_media(
    engine: *const inference_engine,
    request: &str,
    media: &[inference_media],
) -> (inference_status, Option<Value>) {
    let mut response = null_mut();
    let status = unsafe {
        inference_chat_with_media(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            media.as_ptr(),
            media.len(),
            &mut response,
        )
    };
    let response =
        (!response.is_null()).then(|| serde_json::from_str(&take_string(response)).unwrap());
    (status, response)
}

fn decoded_tokens(response: &Value) -> Vec<Value> {
    response["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|token| token["top_logprobs"][0]["token"].clone())
        .collect()
}

#[test]
fn attached_media_decodes_like_the_same_image_as_a_data_url() {
    let dir = support::tiny_checkpoint().unwrap();
    let (status, engine) = load(&spec(dir.path()));
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let png = std::fs::read(std::path::Path::new(support::FIXTURES).join("page_00.png")).unwrap();

    let data_url = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    );
    let (status, by_url) = chat_with_media(engine, &image_request(&data_url), &[]);
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let by_url = by_url.unwrap();
    assert!(!decoded_tokens(&by_url).is_empty());

    let media = [inference_media {
        data: png.as_ptr(),
        len: png.len(),
        mime_type: c"image/png".as_ptr(),
    }];
    let (status, by_attachment) = chat_with_media(engine, &image_request("media://0"), &media);
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    assert_eq!(
        decoded_tokens(&by_url),
        decoded_tokens(&by_attachment.unwrap())
    );

    let (status, _) = chat_with_media(engine, &image_request("media://1"), &media);
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST);
    assert!(last_error().contains("media://1"), "{}", last_error());

    let mut response = null_mut();
    let request = image_request("media://0");
    let status = unsafe {
        inference_chat_with_media(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            null(),
            1,
            &mut response,
        )
    };
    assert_eq!(status, INFERENCE_ERR_INVALID_ARGUMENT);
    assert!(response.is_null());

    unsafe { inference_engine_free(engine) };
}

fn completion_request(stream: bool) -> String {
    json!({
        "model": "default",
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "stream": stream,
    })
    .to_string()
}

#[test]
fn completion_and_stream_agree_and_embeddings_need_an_embedding_model() {
    let dir = support::tiny_checkpoint().unwrap();
    let (status, engine) = load(&spec(dir.path()));
    assert_eq!(status, INFERENCE_OK, "{}", last_error());

    let request = completion_request(false);
    let mut response = null_mut();
    let status = unsafe {
        inference_completion(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut response,
        )
    };
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let response: Value = serde_json::from_str(&take_string(response)).unwrap();
    assert_eq!(response["object"], "text_completion");
    let text = response["choices"][0]["text"].as_str().unwrap().to_string();

    let request = completion_request(true);
    let mut stream = null_mut();
    let status = unsafe {
        inference_completion_stream_open(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut stream,
        )
    };
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let mut streamed = String::new();
    loop {
        let (mut event, mut done) = (null_mut(), 0);
        let status =
            unsafe { inference_stream_next(stream, POLL_TIMEOUT_MS, &mut event, &mut done) };
        assert_eq!(status, INFERENCE_OK, "{}", last_error());
        if done == 1 {
            break;
        }
        assert!(!event.is_null(), "a {POLL_TIMEOUT_MS} ms poll timed out");
        let event: Value = serde_json::from_str(&take_string(event)).unwrap();
        assert_eq!(event["event"], "chunk", "{event}");
        streamed.push_str(
            event["data"]["choices"][0]["text"]
                .as_str()
                .unwrap_or_default(),
        );
    }
    unsafe { inference_stream_free(stream) };
    assert_eq!(streamed, text);

    let request = json!({"model": "default", "input": "hello"}).to_string();
    let mut response = null_mut();
    let status = unsafe {
        inference_embeddings(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut response,
        )
    };
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST, "{}", last_error());
    assert!(response.is_null());
    let error: Value = serde_json::from_str(&last_error()).unwrap();
    assert_eq!(error["error"]["type"], "invalid_request_error", "{error}");

    unsafe { inference_engine_free(engine) };
}

fn anthropic_request(stream: bool) -> String {
    json!({
        "model": "default",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "top_k": 1,
        "stream": stream,
    })
    .to_string()
}

#[test]
fn anthropic_messages_and_stream_agree_and_errors_use_the_anthropic_shape() {
    let dir = support::tiny_checkpoint().unwrap();
    let (status, engine) = load(&spec(dir.path()));
    assert_eq!(status, INFERENCE_OK, "{}", last_error());

    let request = anthropic_request(false);
    let mut response = null_mut();
    let status = unsafe {
        inference_anthropic_messages(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut response,
        )
    };
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let response: Value = serde_json::from_str(&take_string(response)).unwrap();
    assert_eq!(response["type"], "message", "{response}");
    let text: String = response["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str())
        .collect();

    let request = anthropic_request(true);
    let mut stream = null_mut();
    let status = unsafe {
        inference_anthropic_messages_stream_open(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut stream,
        )
    };
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let (mut names, mut streamed) = (Vec::new(), String::new());
    loop {
        let (mut event, mut done) = (null_mut(), 0);
        let status =
            unsafe { inference_stream_next(stream, POLL_TIMEOUT_MS, &mut event, &mut done) };
        assert_eq!(status, INFERENCE_OK, "{}", last_error());
        if done == 1 {
            break;
        }
        assert!(!event.is_null(), "a {POLL_TIMEOUT_MS} ms poll timed out");
        let event: Value = serde_json::from_str(&take_string(event)).unwrap();
        let name = event["event"].as_str().unwrap().to_string();
        if name == "content_block_delta" && event["data"]["delta"]["type"] == "text_delta" {
            streamed.push_str(event["data"]["delta"]["text"].as_str().unwrap());
        }
        names.push(name);
    }
    unsafe { inference_stream_free(stream) };
    assert_eq!(
        names.first().map(String::as_str),
        Some("message_start"),
        "{names:?}"
    );
    assert_eq!(
        names.last().map(String::as_str),
        Some("message_stop"),
        "{names:?}"
    );
    assert_eq!(streamed, text);

    let request =
        json!({"model": "default", "messages": [{"role": "user", "content": "hi"}]}).to_string();
    let mut response = null_mut();
    let status = unsafe {
        inference_anthropic_messages(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut response,
        )
    };
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST, "{}", last_error());
    let error: Value = serde_json::from_str(&last_error()).unwrap();
    assert_eq!(error["type"], "error", "{error}");
    assert_eq!(error["error"]["type"], "invalid_request_error", "{error}");

    // a streamed request cannot ask for tool approvals this surface cannot answer
    let ask = json!({
        "model": "default",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "agent_permission": "ask",
    })
    .to_string();
    let mut stream = null_mut();
    let status = unsafe {
        inference_anthropic_messages_stream_open(
            engine,
            ask.as_ptr().cast::<c_char>(),
            ask.len(),
            &mut stream,
        )
    };
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST, "{}", last_error());
    assert!(stream.is_null());

    unsafe { inference_engine_free(engine) };
}

// How long a background response may take on the tiny checkpoint before the test calls it hung.
const BACKGROUND_DEADLINE: std::time::Duration = std::time::Duration::from_secs(60);
const BACKGROUND_POLL: std::time::Duration = std::time::Duration::from_millis(50);

fn responses_request(extra: Value) -> String {
    let mut request = json!({
        "model": "default",
        "input": PROMPT,
        "max_output_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
    });
    request
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    request.to_string()
}

fn create_response(engine: *const inference_engine, request: &str) -> (inference_status, Value) {
    let mut response = null_mut();
    let status = unsafe {
        inference_responses_create(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut response,
        )
    };
    (status, stored_body(status, response))
}

fn stored_body(status: inference_status, response: *mut inference_string) -> Value {
    if status == INFERENCE_OK {
        serde_json::from_str(&take_string(response)).unwrap()
    } else {
        assert!(response.is_null());
        serde_json::from_str(&last_error()).unwrap()
    }
}

type ResponseIdCall = unsafe extern "C" fn(
    *const inference_engine,
    *const c_char,
    usize,
    *mut *mut inference_string,
) -> inference_status;

fn by_id(
    call: ResponseIdCall,
    engine: *const inference_engine,
    id: &str,
) -> (inference_status, Value) {
    let mut response = null_mut();
    let status = unsafe {
        call(
            engine,
            id.as_ptr().cast::<c_char>(),
            id.len(),
            &mut response,
        )
    };
    (status, stored_body(status, response))
}

fn drain(stream: *mut inference_stream) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        let (mut event, mut done) = (null_mut(), 0);
        let status =
            unsafe { inference_stream_next(stream, POLL_TIMEOUT_MS, &mut event, &mut done) };
        assert_eq!(status, INFERENCE_OK, "{}", last_error());
        if done == 1 {
            break;
        }
        assert!(!event.is_null(), "a {POLL_TIMEOUT_MS} ms poll timed out");
        events.push(serde_json::from_str(&take_string(event)).unwrap());
    }
    unsafe { inference_stream_free(stream) };
    events
}

#[test]
fn responses_stream_store_continue_and_run_in_the_background() {
    let dir = support::tiny_checkpoint().unwrap();
    let (status, engine) = load(&spec(dir.path()));
    assert_eq!(status, INFERENCE_OK, "{}", last_error());

    let (status, response) = create_response(engine, &responses_request(json!({})));
    assert_eq!(status, INFERENCE_OK, "{response}");
    assert_eq!(response["object"], "response", "{response}");
    assert_eq!(response["status"], "completed", "{response}");
    let id = response["id"].as_str().unwrap().to_string();
    let text = response["output_text"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    let request = responses_request(json!({"stream": true}));
    let mut stream = null_mut();
    let status = unsafe {
        inference_responses_stream_open(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut stream,
        )
    };
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    let events = drain(stream);
    let names: Vec<&str> = events
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    assert_eq!(names.first(), Some(&"response.created"), "{names:?}");
    assert_eq!(names.last(), Some(&"response.completed"), "{names:?}");
    let streamed: String = events
        .iter()
        .filter(|e| e["event"] == "response.output_text.delta")
        .map(|e| e["data"]["delta"].as_str().unwrap())
        .collect();
    assert_eq!(streamed, text);
    // a streamed response is stored as soon as it completes, so it can be continued at once
    let streamed_id = events.last().unwrap()["data"]["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let (status, stored) = by_id(inference_responses_get, engine, &streamed_id);
    assert_eq!(status, INFERENCE_OK, "{stored}");
    assert_eq!(stored["status"], "completed", "{stored}");
    let follow_up = responses_request(json!({"previous_response_id": streamed_id}));
    let (status, continued) = create_response(engine, &follow_up);
    assert_eq!(status, INFERENCE_OK, "{continued}");
    let input_tokens = |response: &Value| response["usage"]["input_tokens"].as_u64().unwrap();
    assert!(
        input_tokens(&continued) > input_tokens(&response),
        "the follow-up carries the stored conversation: {continued}"
    );

    let (status, stored) = by_id(inference_responses_get, engine, &id);
    assert_eq!(status, INFERENCE_OK, "{stored}");
    assert_eq!(
        (stored["id"].as_str(), stored["status"].as_str()),
        (Some(id.as_str()), Some("completed"))
    );
    let (status, cancelled) = by_id(inference_responses_cancel, engine, &id);
    assert_eq!(status, INFERENCE_OK, "{cancelled}");
    assert_eq!(
        cancelled["status"], "completed",
        "a finished response stays finished"
    );
    let (status, deleted) = by_id(inference_responses_delete, engine, &id);
    assert_eq!(status, INFERENCE_OK, "{deleted}");
    assert_eq!(
        deleted,
        json!({"id": id, "object": "response.deleted", "deleted": true})
    );
    for call in [inference_responses_get, inference_responses_delete] {
        let (status, error) = by_id(call, engine, &id);
        assert_eq!(status, INFERENCE_ERR_NOT_FOUND, "{error}");
        assert_eq!(error["error"]["code"], "response_not_found", "{error}");
    }
    let (status, error) = create_response(
        engine,
        &responses_request(json!({"previous_response_id": id})),
    );
    assert_eq!(status, INFERENCE_ERR_NOT_FOUND, "{error}");
    assert_eq!(error["error"]["param"], "previous_response_id", "{error}");

    let (status, queued) = create_response(engine, &responses_request(json!({"background": true})));
    assert_eq!(status, INFERENCE_OK, "{queued}");
    assert_eq!(queued["status"], "queued", "{queued}");
    let background_id = queued["id"].as_str().unwrap();
    let deadline = std::time::Instant::now() + BACKGROUND_DEADLINE;
    let finished = loop {
        let (status, response) = by_id(inference_responses_get, engine, background_id);
        assert_eq!(status, INFERENCE_OK, "{response}");
        if !matches!(response["status"].as_str(), Some("queued" | "in_progress")) {
            break response;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "background response never finished"
        );
        std::thread::sleep(BACKGROUND_POLL);
    };
    assert_eq!(finished["status"], "completed", "{finished}");
    assert_eq!(finished["output_text"].as_str().unwrap_or_default(), text);

    let request = responses_request(json!({"stream": true, "background": true}));
    let mut stream = null_mut();
    let status = unsafe {
        inference_responses_stream_open(
            engine,
            request.as_ptr().cast::<c_char>(),
            request.len(),
            &mut stream,
        )
    };
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST);
    assert!(stream.is_null());
    let error: Value = serde_json::from_str(&last_error()).unwrap();
    assert_eq!(
        error["error"]["code"], "unsupported_parameter_combination",
        "{error}"
    );

    let bad_id = [0xff_u8];
    let mut response = null_mut();
    let status = unsafe {
        inference_responses_get(engine, bad_id.as_ptr().cast::<c_char>(), 1, &mut response)
    };
    assert_eq!(status, INFERENCE_ERR_INVALID_ARGUMENT);

    unsafe { inference_engine_free(engine) };
}

fn request_call(
    call: ResponseIdCall,
    engine: *const inference_engine,
    request: &Value,
) -> (inference_status, Value) {
    by_id(call, engine, &request.to_string())
}

#[test]
fn models_unload_reload_and_adapter_management_is_guarded() {
    let dir = support::tiny_checkpoint().unwrap();
    let adapter_root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let mut spec: Value = serde_json::from_str(&spec(dir.path())).unwrap();
    spec["adapters"] = json!({"runtime_updates": true, "root": adapter_root.path()});
    let (status, engine) = load(&spec.to_string());
    assert_eq!(status, INFERENCE_OK, "{}", last_error());

    let mut response = null_mut();
    assert_eq!(
        unsafe { inference_models_list(engine, &mut response) },
        INFERENCE_OK
    );
    let models: Value = serde_json::from_str(&take_string(response)).unwrap();
    assert_eq!(models["object"], "list", "{models}");
    assert_eq!(models["data"][0]["id"], "default", "{models}");
    assert_eq!(models["data"][1]["status"], "loaded", "{models}");
    let model_id = models["data"][1]["id"].as_str().unwrap().to_string();
    let model = json!({"model_id": model_id});

    for (call, expected) in [
        (inference_model_status as ResponseIdCall, "loaded"),
        (inference_model_unload, "unloaded"),
        (inference_model_unload, "unloaded"),
        (inference_model_status, "unloaded"),
        (inference_model_reload, "loaded"),
        (inference_model_reload, "loaded"),
    ] {
        let (status, response) = request_call(call, engine, &model);
        assert_eq!(status, INFERENCE_OK, "expecting {expected}: {response}");
        assert_eq!(response, json!({"model_id": model_id, "status": expected}));
    }
    let (status, _) = chat(engine, &chat_request(false));
    assert_eq!(
        status,
        INFERENCE_OK,
        "a reloaded model serves again: {}",
        last_error()
    );
    let (status, error) = request_call(
        inference_model_status,
        engine,
        &json!({"model_id": "no-such-model"}),
    );
    assert_eq!(status, INFERENCE_ERR_NOT_FOUND, "{error}");
    assert_eq!(error["error"]["code"], "model_not_found", "{error}");

    let load_adapter = |path: &str| {
        request_call(
            inference_lora_adapter_load,
            engine,
            &json!({"lora_name": "production", "lora_path": path}),
        )
    };
    let (status, error) = load_adapter("missing");
    assert_eq!(status, INFERENCE_ERR_NOT_FOUND, "{error}");
    assert_eq!(error["error"]["code"], "adapter_path_not_found", "{error}");
    let (status, error) = load_adapter(outside.path().to_str().unwrap());
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST, "{error}");
    assert_eq!(error["error"]["code"], "adapter_path_forbidden", "{error}");
    let (status, error) = request_call(inference_lora_adapters_list, engine, &json!({"model": 5}));
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST, "{error}");
    unsafe { inference_engine_free(engine) };

    // without runtime_updates in the spec, adapters can be listed but not changed
    let (status, engine) = load(&self::spec(dir.path()));
    assert_eq!(status, INFERENCE_OK, "{}", last_error());
    for (call, request) in [
        (
            inference_lora_adapter_load as ResponseIdCall,
            json!({"lora_name": "production", "lora_path": "anywhere"}),
        ),
        (
            inference_lora_adapter_unload,
            json!({"lora_name": "production"}),
        ),
    ] {
        let (status, error) = request_call(call, engine, &request);
        assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST, "{error}");
        assert_eq!(error["error"]["code"], "lora_updates_disabled", "{error}");
    }
    unsafe { inference_engine_free(engine) };
}
