use std::collections::HashMap;

use base64::{Engine, engine::general_purpose::STANDARD};
use image::{DynamicImage, codecs::png::PngEncoder};
use serde_json::json;

use crate::{
    AgenticToolCallData, AgenticToolCallPhase, AgenticToolCallRecord, ChatCompletionResponse, File,
    Response,
};

// Files past this many stay reachable through the file store but are not embedded in the response body.
pub(crate) const MAX_FILES_PER_RESPONSE: usize = 64;

/// Folds the tool-call progress and file events before a non-streaming chat's final response into that response.
#[derive(Default)]
pub struct ChatResponseCollector {
    records: Vec<AgenticToolCallRecord>,
    pending_args: HashMap<(usize, String), String>,
    files: Vec<File>,
}

impl ChatResponseCollector {
    /// Keeps a progress or file event; any other response comes back for the caller.
    pub fn absorb(&mut self, response: Response) -> Option<Response> {
        match response {
            Response::AgenticToolCallProgress {
                round,
                tool_name,
                phase,
            } => {
                record_agentic_progress(
                    &mut self.records,
                    &mut self.pending_args,
                    round,
                    &tool_name,
                    &phase,
                );
                None
            }
            Response::File(file) => {
                if self.files.len() < MAX_FILES_PER_RESPONSE {
                    self.files.push(file);
                } else {
                    tracing::warn!(
                        "MAX_FILES_PER_RESPONSE ({MAX_FILES_PER_RESPONSE}) reached; remaining files are fetchable via /v1/files/{{id}}",
                    );
                }
                None
            }
            response => Some(response),
        }
    }

    pub fn finish(self, mut response: ChatCompletionResponse) -> ChatCompletionResponse {
        let Self {
            mut records, files, ..
        } = self;
        if !files.is_empty() {
            stamp_file_ids(&mut records, &files);
            response.files = Some(files);
        }
        if !records.is_empty() {
            response.agentic_tool_calls = Some(records);
        }
        response
    }
}

pub fn encode_agentic_tool_images(images: &[DynamicImage]) -> Vec<String> {
    images
        .iter()
        .filter_map(|image| {
            let mut buffer = Vec::new();
            match image.write_with_encoder(PngEncoder::new(&mut buffer)) {
                Ok(()) => Some(STANDARD.encode(buffer)),
                Err(e) => {
                    tracing::warn!("failed to encode agentic tool image: {e}");
                    None
                }
            }
        })
        .collect()
}

/// Arguments string from a Calling-phase `AgenticToolCallData`.
fn extract_arguments(data: &AgenticToolCallData) -> String {
    match data {
        AgenticToolCallData::CodeExecution {
            code: Some(code), ..
        } => serde_json::json!({"code": code}).to_string(),
        AgenticToolCallData::WebSearch {
            query: Some(query), ..
        } => serde_json::json!({"query": query}).to_string(),
        AgenticToolCallData::Shell { commands, .. } => {
            serde_json::json!({"commands": commands}).to_string()
        }
        AgenticToolCallData::Custom { arguments, .. } => arguments.clone(),
        _ => String::new(),
    }
}

/// Fold progress events into `AgenticToolCallRecord` for non-streaming responses. `pending_args` keeps Calling-phase args keyed by (round, tool_name).
fn record_agentic_progress(
    records: &mut Vec<AgenticToolCallRecord>,
    pending_args: &mut HashMap<(usize, String), String>,
    round: usize,
    tool_name: &str,
    phase: &AgenticToolCallPhase,
) {
    match phase {
        AgenticToolCallPhase::Calling(data) => {
            pending_args.insert((round, tool_name.to_string()), extract_arguments(data));
        }
        AgenticToolCallPhase::Complete(data) => {
            let arguments = pending_args
                .remove(&(round, tool_name.to_string()))
                .unwrap_or_default();

            let (result_content, result_images_base64) = match data {
                AgenticToolCallData::CodeExecution {
                    stdout,
                    stderr,
                    exception,
                    images,
                    ..
                } => {
                    let mut content_parts = Vec::new();
                    if let Some(s) = stdout {
                        content_parts.push(format!("stdout: {s}"));
                    }
                    if let Some(s) = stderr {
                        content_parts.push(format!("stderr: {s}"));
                    }
                    if let Some(e) = exception {
                        content_parts.push(format!("exception: {e}"));
                    }
                    (content_parts.join("\n"), encode_agentic_tool_images(images))
                }
                AgenticToolCallData::WebSearch {
                    results_count,
                    sources,
                    ..
                } => {
                    let mut parts = Vec::new();
                    if let Some(n) = results_count {
                        parts.push(format!("{n} results"));
                    }
                    if !sources.is_empty() {
                        parts.push(format!("sources: {}", sources.join(", ")));
                    }
                    let msg = parts.join("\n");
                    (msg, vec![])
                }
                AgenticToolCallData::Shell {
                    stdout,
                    stderr,
                    exit_code,
                    status,
                    working_directory,
                    timed_out,
                    ..
                } => {
                    let mut content = json!({
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "status": status,
                        "working_directory": working_directory,
                        "timed_out": timed_out,
                    });
                    if let Some(obj) = content.as_object_mut() {
                        obj.retain(|_, value| !value.is_null());
                    }
                    (content.to_string(), vec![])
                }
                AgenticToolCallData::Custom { content, .. } => (content.clone(), vec![]),
            };
            records.push(AgenticToolCallRecord {
                round,
                name: tool_name.to_string(),
                arguments,
                result_content,
                result_images_base64,
                file_ids: Vec::new(),
            });
        }
    }
}

/// Fill each record's `file_ids` from files whose `source.round` and `source.tool` match.
fn stamp_file_ids(records: &mut [AgenticToolCallRecord], files: &[File]) {
    for r in records.iter_mut() {
        let matched: Vec<String> = files
            .iter()
            .filter(|f| f.source.round == r.round && f.source.tool == r.name)
            .map(|f| f.id.clone())
            .collect();
        if !matched.is_empty() {
            r.file_ids = matched;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Usage, files::FileSource};

    fn response() -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: String::new(),
            choices: Vec::new(),
            created: 0,
            model: String::new(),
            system_fingerprint: String::new(),
            object: String::new(),
            usage: Usage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
                prompt_tokens_details: None,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
            adapter_generation: None,
            agentic_tool_calls: None,
            files: None,
            session_id: None,
        }
    }

    fn text_file(id: &str, source: FileSource) -> File {
        File::from_bytes(
            id.to_string(),
            "out.txt".to_string(),
            None,
            "assistants_output".to_string(),
            source,
            b"x".to_vec(),
        )
    }

    fn progress(phase: AgenticToolCallPhase) -> Response {
        Response::AgenticToolCallProgress {
            round: 1,
            tool_name: "lookup".to_string(),
            phase,
        }
    }

    #[test]
    fn tool_calls_and_their_files_land_on_the_final_response() {
        let data = |content: &str| AgenticToolCallData::Custom {
            arguments: r#"{"q":1}"#.to_string(),
            content: content.to_string(),
        };
        let mut collector = ChatResponseCollector::default();
        assert!(
            collector
                .absorb(progress(AgenticToolCallPhase::Calling(data(""))))
                .is_none()
        );
        assert!(
            collector
                .absorb(progress(AgenticToolCallPhase::Complete(data("found"))))
                .is_none()
        );
        let source = FileSource {
            tool: "lookup".to_string(),
            round: 1,
            turn: 0,
        };
        assert!(
            collector
                .absorb(Response::File(text_file("file_a", source)))
                .is_none()
        );
        let elsewhere = FileSource {
            tool: "lookup".to_string(),
            round: 2,
            turn: 0,
        };
        assert!(
            collector
                .absorb(Response::File(text_file("file_b", elsewhere)))
                .is_none()
        );
        assert!(matches!(
            collector.absorb(Response::InternalError(anyhow::anyhow!("x").into())),
            Some(Response::InternalError(_))
        ));

        let response = collector.finish(response());
        let records = response.agentic_tool_calls.expect("records attached");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].arguments, r#"{"q":1}"#);
        assert_eq!(records[0].result_content, "found");
        assert_eq!(records[0].file_ids, vec!["file_a".to_string()]);
        assert_eq!(response.files.map(|files| files.len()), Some(2));
    }

    #[test]
    fn a_plain_response_is_left_as_it_was() {
        let response = ChatResponseCollector::default().finish(response());
        assert!(response.agentic_tool_calls.is_none() && response.files.is_none());
    }

    #[test]
    fn files_past_the_cap_are_left_out_of_the_body() {
        let mut collector = ChatResponseCollector::default();
        for index in 0..=MAX_FILES_PER_RESPONSE {
            let source = FileSource {
                tool: "lookup".to_string(),
                round: 0,
                turn: 0,
            };
            collector.absorb(Response::File(text_file(&format!("file_{index}"), source)));
        }
        let files = collector.finish(response()).files.expect("files attached");
        assert_eq!(files.len(), MAX_FILES_PER_RESPONSE);
    }
}
