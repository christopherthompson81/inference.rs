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
    assert_eq!(status, INFERENCE_ERR_INVALID_REQUEST, "{}", last_error());
    let error: Value = serde_json::from_str(&last_error()).unwrap();
    assert!(
        error["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no-such-model")),
        "{error}"
    );

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
            inference_status::INFERENCE_ERR_UNAVAILABLE as i32,
        ))
    };
    assert_eq!(name.to_str().unwrap(), "INFERENCE_ERR_UNAVAILABLE");
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
