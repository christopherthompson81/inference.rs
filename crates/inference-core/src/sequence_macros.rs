use futures::future::BoxFuture;
use tokio::sync::Mutex;

use crate::{
    get_mut_arcmutex,
    pipeline::Pipeline,
    prefix_cacher::PrefixCacheManagerV2,
    response::{
        ChatCompletionResponse, Choice, CompletionChoice, CompletionResponse, Response,
        ResponseMessage, SYSTEM_FINGERPRINT,
    },
    sequence::{Sequence, SequenceState},
};

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error_stateaware_ok {
    ($fallible:expr, $seq:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                if let Err(_) = $seq
                    .responder()
                    .send(Response::InternalError(e.into()))
                    .await
                {
                    tracing::warn!("Receiver disconnected");
                }
                $seq.set_state(SequenceState::Error);
                return Ok(());
            }
        }
    };
}

// Boxed and out of line so every forward failure in the engine loop shares one copy of the error path.
pub(crate) fn report_pipeline_forward_error<'a>(
    stage: &'static str,
    message: String,
    detail: String,
    seqs: &'a mut [&mut Sequence],
    pipeline: &'a Mutex<dyn Pipeline>,
    prefix_cacher: &'a Mutex<PrefixCacheManagerV2>,
) -> BoxFuture<'a, ()> {
    Box::pin(async move {
        // Auto-retry on iOS Metal background GPU error: when the iOS app goes to background, Metal rejects
        // command buffers. Reset the cache, sleep, and let the engine loop retry; sequences stay Running.
        #[cfg(feature = "metal")]
        if message.contains("Insufficient Permission")
            || message.contains("BackgroundExecutionNotPermitted")
        {
            tracing::warn!(
                "Metal GPU background error detected (iOS app likely in background). \
                 Pausing 1s before retry..."
            );
            {
                let p = get_mut_arcmutex!(pipeline);
                if let Err(reset_err) = p.set_none_cache(seqs, true, true, false) {
                    tracing::error!("Failed to reset model cache: {reset_err}");
                }
            }
            get_mut_arcmutex!(prefix_cacher).evict_all_caches().unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            return;
        }

        let (tokenizer, pipeline_name) = {
            let pipeline = get_mut_arcmutex!(pipeline);
            (pipeline.tokenizer(), pipeline.name())
        };
        tracing::error!("{stage} - Model failed with error: {detail}");
        for seq in seqs.iter_mut() {
            // Step 1: Add all choices to groups
            let start = seq.prompt_tokens().min(seq.get_toks().len());
            let res = match &tokenizer {
                Some(tok) => tok
                    .decode(&seq.get_toks()[start..], false)
                    .unwrap_or_default(),
                None => String::new(),
            };

            if seq.get_mut_group().is_chat {
                let choice = Choice {
                    finish_reason: "error".to_string(),
                    stop_sequence: None,
                    index: seq.get_response_index(),
                    message: ResponseMessage {
                        content: Some(res),
                        role: "assistant".to_string(),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                };
                seq.add_choice_to_group(choice);
            } else {
                let choice = CompletionChoice {
                    finish_reason: "error".to_string(),
                    index: seq.get_response_index(),
                    text: res,
                    logprobs: None,
                };
                seq.add_completion_choice_to_group(choice);
            }
        }
        for seq in seqs.iter_mut() {
            // Step 2: Respond with all groups
            let group = seq.get_mut_group();

            if group.is_chat {
                let partial_completion_response = ChatCompletionResponse {
                    id: seq.id().to_string(),
                    choices: group.get_choices().to_vec(),
                    created: seq.creation_time(),
                    model: pipeline_name.clone(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "chat.completion".to_string(),
                    usage: group.get_usage(),
                    adapter_generation: seq
                        .adapter_generation()
                        .map(|generation| generation.to_string()),
                    agentic_tool_calls: None,
                    files: None,
                    session_id: None,
                };

                if let Err(send_err) = seq
                    .responder()
                    .send(Response::ModelError(
                        message.clone(),
                        partial_completion_response,
                    ))
                    .await
                {
                    tracing::warn!(
                        "Failed to send chat model error to client for seq {}: {} (client likely disconnected)",
                        seq.id(),
                        send_err
                    );
                }
            } else {
                let partial_completion_response = CompletionResponse {
                    id: seq.id().to_string(),
                    choices: group.get_completion_choices().to_vec(),
                    created: seq.creation_time(),
                    model: pipeline_name.clone(),
                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                    object: "text_completion".to_string(),
                    usage: group.get_usage(),
                    adapter_generation: seq
                        .adapter_generation()
                        .map(|generation| generation.to_string()),
                };

                if let Err(send_err) = seq
                    .responder()
                    .send(Response::CompletionModelError(
                        message.clone(),
                        partial_completion_response,
                    ))
                    .await
                {
                    tracing::warn!(
                        "Failed to send completion model error to client for seq {}: {} (client likely disconnected)",
                        seq.id(),
                        send_err
                    );
                }
            }
        }
        for seq in seqs.iter_mut() {
            // Step 3: Set state - This cannot be done in Step 2 as `group` is locking the refcell
            seq.set_state(SequenceState::Error);
        }

        let p = get_mut_arcmutex!(pipeline);
        // Also reset non granular state because:
        // - The sequence is gone
        // - We should reset the state then, including draft.
        if let Err(reset_err) = p.set_none_cache(seqs, true, true, false) {
            tracing::error!("Failed to reset model cache: {reset_err}");
        }
        get_mut_arcmutex!(prefix_cacher).evict_all_caches().unwrap();
    })
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_group {
    ($this:expr) => {
        loop {
            if let Ok(inner) = $this.group.try_lock() {
                break inner;
            }
            // Yield to allow other threads to make progress and release the lock.
            std::thread::yield_now();
        }
    };
}
