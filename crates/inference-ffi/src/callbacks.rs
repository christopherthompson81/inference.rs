//! Host callbacks: C functions the agent loop calls for host tools and for web search.

use std::{
    ffi::{c_char, c_void},
    sync::Arc,
};

use inference_api::{
    engine::{EngineCallbacks, SearchResult, Tool, ToolCallbackKind, ToolCallbackWithTool},
    logits_processors::{self, CustomLogitsProcessor},
};

use crate::{Failure, FfiResult};

/// Filled by a callback with its answer; owned by the library.
#[allow(non_camel_case_types)]
pub struct inference_callback_result {
    outcome: Option<Result<String, String>>,
}

/// Mirrors `inference_tool_callback`.
#[allow(non_camel_case_types)]
pub type inference_tool_callback = unsafe extern "C" fn(
    user_data: *mut c_void,
    tool_name: *const c_char,
    arguments: *const c_char,
    arguments_len: usize,
    context: *const c_char,
    context_len: usize,
    result: *mut inference_callback_result,
);

/// Mirrors `inference_search_callback`.
#[allow(non_camel_case_types)]
pub type inference_search_callback = unsafe extern "C" fn(
    user_data: *mut c_void,
    query: *const c_char,
    query_len: usize,
    result: *mut inference_callback_result,
);

/// Mirrors `inference_logits_processor_callback`.
#[allow(non_camel_case_types)]
pub type inference_logits_processor_callback = unsafe extern "C" fn(
    user_data: *mut c_void,
    logits: *mut f32,
    vocab_size: usize,
    context: *const u32,
    context_len: usize,
) -> i32;

/// Mirrors `inference_host_tool`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct inference_host_tool {
    pub definition: *const c_char,
    pub definition_len: usize,
    pub callback: Option<inference_tool_callback>,
    pub user_data: *mut c_void,
}

/// Mirrors `inference_host_callbacks`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct inference_host_callbacks {
    pub tools: *const inference_host_tool,
    pub tool_count: usize,
    pub search: Option<inference_search_callback>,
    pub search_user_data: *mut c_void,
}

// The header makes the caller promise user_data may be used from any thread for the engine's lifetime.
struct UserData(*mut c_void);
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

impl UserData {
    // A method, so closures capture the Send + Sync wrapper rather than its raw pointer field.
    fn get(&self) -> *mut c_void {
        self.0
    }
}

/// The host's callback as a processor; any status but 0 fails the request it runs in.
pub(crate) fn host_logits_processor(
    name: &str,
    callback: inference_logits_processor_callback,
    user_data: *mut c_void,
) -> Arc<dyn CustomLogitsProcessor> {
    let user_data = UserData(user_data);
    let name = name.to_string();
    logits_processors::in_place(move |logits, context| {
        // Safety: both slices outlive the call, which the header lets edit `logits` and read `context`.
        let status = unsafe {
            callback(
                user_data.get(),
                logits.as_mut_ptr(),
                logits.len(),
                context.as_ptr(),
                context.len(),
            )
        };
        match status {
            0 => Ok(()),
            status => Err(format!(
                "logits processor `{name}` failed with status {status}"
            )),
        }
    })
}

const NO_RESULT: &str = "the host callback returned without setting a result";
const HOST_FAILED: &str = "the host callback failed";

// Runs a host callback against a fresh result and reads back what it set.
fn call_host(invoke: impl FnOnce(*mut inference_callback_result)) -> anyhow::Result<String> {
    let mut result = inference_callback_result { outcome: None };
    invoke(&mut result);
    match result.outcome {
        Some(Ok(text)) => Ok(text),
        Some(Err(message)) => Err(anyhow::anyhow!(message)),
        None => Err(anyhow::anyhow!(NO_RESULT)),
    }
}

// NUL-terminated copies for the callback, which is also given each copy's length; interior NULs are dropped.
fn c_string(text: &str) -> std::ffi::CString {
    std::ffi::CString::new(text.replace('\0', "")).expect("NULs were removed")
}

fn host_tool(tool: &inference_host_tool, index: usize) -> FfiResult<ToolCallbackWithTool> {
    let callback = tool
        .callback
        .ok_or_else(|| Failure::invalid(format!("tools[{index}].callback is NULL")))?;
    // Safety: the caller passes definition valid for definition_len bytes.
    let definition = unsafe {
        crate::engine::arg_bytes(
            tool.definition,
            tool.definition_len,
            &format!("tools[{index}].definition"),
        )?
    };
    let definition: Tool = serde_json::from_slice(definition)
        .map_err(|error| Failure::invalid(format!("tools[{index}].definition: {error}")))?;
    let user_data = UserData(tool.user_data);
    let tool_name = c_string(&definition.function.name);
    let run = move |called: &inference_api::engine::CalledFunction,
                    context: &inference_api::engine::ToolCallContext| {
        let arguments = c_string(&called.arguments);
        let context = c_string(
            &serde_json::json!({"session_id": context.session_id, "round": context.round})
                .to_string(),
        );
        call_host(|result| unsafe {
            callback(
                user_data.get(),
                tool_name.as_ptr(),
                arguments.as_ptr(),
                arguments.as_bytes().len(),
                context.as_ptr(),
                context.as_bytes().len(),
                result,
            )
        })
    };
    Ok(ToolCallbackWithTool {
        callback: ToolCallbackKind::Text(Arc::new(run)),
        tool: definition,
    })
}

/// Safety: `callbacks` is NULL or valid, with `tools` valid for `tool_count` entries.
pub(crate) unsafe fn engine_callbacks(
    callbacks: *const inference_host_callbacks,
) -> FfiResult<EngineCallbacks> {
    unsafe {
        let Some(callbacks) = callbacks.as_ref() else {
            return Ok(EngineCallbacks::default());
        };
        let tools = if callbacks.tool_count == 0 {
            Vec::new()
        } else if callbacks.tools.is_null() {
            return Err(Failure::invalid("tools is NULL but tool_count is not 0"));
        } else {
            std::slice::from_raw_parts(callbacks.tools, callbacks.tool_count)
                .iter()
                .enumerate()
                .map(|(index, tool)| host_tool(tool, index))
                .collect::<FfiResult<Vec<_>>>()?
        };
        let mut names = std::collections::HashSet::new();
        if let Some(tool) = tools
            .iter()
            .find(|tool| !names.insert(&tool.tool.function.name))
        {
            let name = &tool.tool.function.name;
            return Err(Failure::invalid(format!(
                "two host tools are named `{name}`"
            )));
        }
        let search = callbacks.search.map(|search| {
            let user_data = UserData(callbacks.search_user_data);
            let run = move |params: &inference_api::engine::SearchFunctionParameters| {
                let query = c_string(&params.query);
                let json = call_host(|result| {
                    search(
                        user_data.get(),
                        query.as_ptr(),
                        query.as_bytes().len(),
                        result,
                    )
                })?;
                Ok(serde_json::from_str::<Vec<SearchResult>>(&json)?)
            };
            Arc::new(run) as Arc<inference_api::engine::SearchCallback>
        });
        Ok(EngineCallbacks { tools, search })
    }
}

/// Safety: `result` is the handle passed to the running callback and `data` valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_callback_result_set(
    result: *mut inference_callback_result,
    data: *const c_char,
    len: usize,
) {
    unsafe {
        crate::guard_value((), || {
            let Some(result) = result.as_mut() else {
                return;
            };
            let bytes = match (data.is_null(), len) {
                (true, 0) => &[][..],
                (true, _) => return,
                (false, _) => std::slice::from_raw_parts(data.cast::<u8>(), len),
            };
            result.outcome = Some(Ok(String::from_utf8_lossy(bytes).into_owned()));
        })
    }
}

/// Safety: `result` is the handle passed to the running callback and `message` NULL or a C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inference_callback_result_fail(
    result: *mut inference_callback_result,
    message: *const c_char,
) {
    unsafe {
        crate::guard_value((), || {
            let Some(result) = result.as_mut() else {
                return;
            };
            let message = if message.is_null() {
                HOST_FAILED.to_string()
            } else {
                std::ffi::CStr::from_ptr(message)
                    .to_string_lossy()
                    .into_owned()
            };
            result.outcome = Some(Err(message));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ECHO_TOOL: &str =
        r#"{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}"#;

    unsafe extern "C" fn echo(
        user_data: *mut c_void,
        tool_name: *const c_char,
        arguments: *const c_char,
        arguments_len: usize,
        context: *const c_char,
        _context_len: usize,
        result: *mut inference_callback_result,
    ) {
        unsafe {
            let prefix = &*(user_data as *const String);
            let name = std::ffi::CStr::from_ptr(tool_name).to_str().unwrap();
            let arguments =
                std::str::from_utf8(std::slice::from_raw_parts(arguments.cast(), arguments_len))
                    .unwrap();
            let context = std::ffi::CStr::from_ptr(context).to_str().unwrap();
            let text = format!("{prefix} {name} {arguments} {context}");
            inference_callback_result_set(result, text.as_ptr().cast(), text.len());
        }
    }

    unsafe extern "C" fn silent(
        _: *mut c_void,
        _: *const c_char,
        _: *const c_char,
        _: usize,
        _: *const c_char,
        _: usize,
        _: *mut inference_callback_result,
    ) {
    }

    unsafe extern "C" fn search(
        _: *mut c_void,
        query: *const c_char,
        query_len: usize,
        result: *mut inference_callback_result,
    ) {
        unsafe {
            let query =
                std::str::from_utf8(std::slice::from_raw_parts(query.cast(), query_len)).unwrap();
            if query == "fail" {
                inference_callback_result_fail(result, c"search backend down".as_ptr());
                return;
            }
            let json = serde_json::json!([{"title": query, "description": "", "url": "https://example.com", "content": ""}])
            .to_string();
            inference_callback_result_set(result, json.as_ptr().cast(), json.len());
        }
    }

    fn tool(callback: inference_tool_callback, user_data: *mut c_void) -> inference_host_tool {
        inference_host_tool {
            definition: ECHO_TOOL.as_ptr().cast(),
            definition_len: ECHO_TOOL.len(),
            callback: Some(callback),
            user_data,
        }
    }

    fn run_tool(callbacks: &EngineCallbacks) -> anyhow::Result<String> {
        let ToolCallbackKind::Text(callback) = &callbacks.tools[0].callback else {
            unreachable!("host tools are text tools");
        };
        let called = inference_api::engine::CalledFunction {
            name: "echo".to_string(),
            arguments: r#"{"x":1}"#.to_string(),
        };
        let context = inference_api::engine::ToolCallContext {
            session_id: Some("s1".to_string()),
            round: Some(2),
            ..Default::default()
        };
        callback(&called, &context)
    }

    #[test]
    fn host_tools_and_search_round_trip_through_c() {
        let prefix = "host".to_string();
        let tools = [tool(echo, &prefix as *const String as *mut c_void)];
        let callbacks = inference_host_callbacks {
            tools: tools.as_ptr(),
            tool_count: 1,
            search: Some(search),
            search_user_data: std::ptr::null_mut(),
        };
        let callbacks = unsafe { engine_callbacks(&callbacks) }.ok().unwrap();
        assert_eq!(callbacks.tools[0].tool.function.name, "echo");
        let output = run_tool(&callbacks).unwrap();
        let (call, context) = output.split_at(output.rfind(' ').unwrap());
        assert_eq!(call, r#"host echo {"x":1}"#);
        let context: serde_json::Value = serde_json::from_str(context.trim()).unwrap();
        assert_eq!(context, serde_json::json!({"session_id": "s1", "round": 2}));

        let search = callbacks.search.unwrap();
        let query = |query: &str| {
            search(&inference_api::engine::SearchFunctionParameters {
                query: query.to_string(),
            })
        };
        assert_eq!(query("rust").unwrap()[0].title, "rust");
        assert_eq!(
            query("fail").unwrap_err().to_string(),
            "search backend down"
        );
    }

    #[test]
    fn an_empty_answer_may_be_null_and_the_last_answer_wins() {
        let answer = call_host(|result| unsafe {
            inference_callback_result_fail(result, std::ptr::null());
            inference_callback_result_set(result, std::ptr::null(), 0);
        });
        assert_eq!(answer.unwrap(), "");
    }

    #[test]
    fn a_callback_that_sets_nothing_fails_and_bad_tools_are_rejected() {
        let tools = [tool(silent, std::ptr::null_mut())];
        let callbacks = inference_host_callbacks {
            tools: tools.as_ptr(),
            tool_count: 1,
            search: None,
            search_user_data: std::ptr::null_mut(),
        };
        let callbacks = unsafe { engine_callbacks(&callbacks) }.ok().unwrap();
        assert_eq!(run_tool(&callbacks).unwrap_err().to_string(), NO_RESULT);

        let mut bad = tool(silent, std::ptr::null_mut());
        bad.definition = c"{}".as_ptr();
        bad.definition_len = 2;
        let bad = [bad];
        let callbacks = inference_host_callbacks {
            tools: bad.as_ptr(),
            tool_count: 1,
            search: None,
            search_user_data: std::ptr::null_mut(),
        };
        assert!(unsafe { engine_callbacks(&callbacks) }.is_err());
        let twice = [
            tool(silent, std::ptr::null_mut()),
            tool(silent, std::ptr::null_mut()),
        ];
        let callbacks = inference_host_callbacks {
            tools: twice.as_ptr(),
            tool_count: 2,
            search: None,
            search_user_data: std::ptr::null_mut(),
        };
        assert!(unsafe { engine_callbacks(&callbacks) }.is_err());
        let callbacks = inference_host_callbacks {
            tools: std::ptr::null(),
            tool_count: 1,
            search: None,
            search_user_data: std::ptr::null_mut(),
        };
        assert!(unsafe { engine_callbacks(&callbacks) }.is_err());
    }
}
