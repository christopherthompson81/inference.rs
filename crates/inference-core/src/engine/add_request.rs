use crate::{
    AudioInput, DiffusionGenerationParams, ModelCategory, RequestMessage, Response, VideoInput,
    pipeline::{
        KvCache, NormalCache, chat_template::is_chat_template_request_error,
        is_inputs_processor_validation_error,
    },
    prefix_cacher::MatchingCache,
    request::{
        DetokenizationRequest, ImageGenerationResponseFormat, NormalRequest, TokenizationRequest,
    },
    sequence::{SeqPreallocatedCache, SeqStepType},
    tools::{ToolCallFormat, ToolCallState, ToolChoice},
};
use candle_core::Tensor;
use either::Either;
use std::{
    ops::Deref,
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::warn;

use crate::{
    StopTokens, get_mut_arcmutex,
    request::Request,
    sampler::Sampler,
    sequence::{Sequence, SequenceGroup},
};

use super::{Engine, TERMINATE_ALL_NEXT_STEP, agentic_loop};

fn tools_for_chat_template(
    tools: Option<&[crate::Tool]>,
    tool_choice: Option<&ToolChoice>,
    format: Option<ToolCallFormat>,
) -> Vec<crate::Tool> {
    if format == Some(ToolCallFormat::Atem) && matches!(tool_choice, Some(ToolChoice::None)) {
        Vec::new()
    } else {
        tools.unwrap_or_default().to_vec()
    }
}

fn choice_seed(seed: Option<u64>, response_index: usize) -> Option<u64> {
    seed.map(|seed| {
        let stream = u64::try_from(response_index).expect("choice index must fit into u64");
        seed.wrapping_add(stream)
    })
}

// Fields of the request message the per-choice loop needs after the message itself is consumed for the prompt.
struct MessageExtras {
    is_chat: bool,
    echo_prompt: bool,
    best_of: Option<usize>,
    images: Option<Vec<image::DynamicImage>>,
    audios: Option<Vec<AudioInput>>,
    videos: Option<Vec<VideoInput>>,
    image_generation_format: Option<ImageGenerationResponseFormat>,
    seq_step_type: SeqStepType,
    diffusion_params: Option<DiffusionGenerationParams>,
    image_gen_save_file: Option<PathBuf>,
}

impl MessageExtras {
    fn of(messages: &RequestMessage) -> Self {
        let mut extras = Self {
            is_chat: false,
            echo_prompt: false,
            best_of: None,
            images: None,
            audios: None,
            videos: None,
            image_generation_format: None,
            seq_step_type: SeqStepType::PromptAndDecode,
            diffusion_params: None,
            image_gen_save_file: None,
        };
        match messages {
            RequestMessage::Chat { .. } => extras.is_chat = true,
            RequestMessage::MultimodalChat {
                images,
                audios,
                videos,
                ..
            } => {
                extras.is_chat = true;
                extras.images = Some(images.clone());
                extras.audios = Some(audios.clone());
                extras.videos = Some(videos.clone());
            }
            RequestMessage::Completion {
                echo_prompt,
                best_of,
                ..
            } => {
                extras.echo_prompt = *echo_prompt;
                extras.best_of = *best_of;
            }
            RequestMessage::CompletionTokens(_) => {}
            RequestMessage::ImageGeneration {
                format,
                generation_params,
                save_file,
                ..
            } => {
                extras.image_generation_format = Some(*format);
                extras.seq_step_type = SeqStepType::OneShot;
                extras.diffusion_params = Some(generation_params.clone());
                extras.image_gen_save_file = save_file.clone();
            }
            RequestMessage::SpeechGeneration { .. }
            | RequestMessage::Embedding { .. }
            | RequestMessage::EmbeddingTokens { .. } => extras.seq_step_type = SeqStepType::OneShot,
        }
        extras
    }
}

fn tool_call_state(
    has_tools: bool,
    requested: Option<ToolChoice>,
    tools: Option<&[crate::Tool]>,
    format: Option<ToolCallFormat>,
) -> Result<ToolCallState, Box<Response>> {
    let requested = requested.unwrap_or(ToolChoice::Auto);
    let tool_choice = if has_tools || requested.forced_function_name().is_some() {
        requested
    } else {
        ToolChoice::None
    };
    ToolCallState::new(tool_choice, tools, format)
        .map_err(|e| Box::new(Response::ValidationError(e.into())))
}

fn first_unknown_token_id(tokenizer: &tokenizers::Tokenizer, token_ids: &[u32]) -> Option<u32> {
    token_ids
        .iter()
        .copied()
        .find(|token_id| tokenizer.id_to_token(*token_id).is_none())
}

impl Engine {
    pub async fn handle_request(self: Arc<Self>, request: Request) {
        match request {
            Request::Normal(mut request) => {
                let is_chat = matches!(
                    &request.messages,
                    RequestMessage::Chat { .. } | RequestMessage::MultimodalChat { .. }
                );
                let in_agentic_loop =
                    request.max_tool_rounds == agentic_loop::AGENTIC_LOOP_REENTRY_SENTINEL;
                let has_tooling = self.tool_callbacks.keys().any(|name| {
                    agentic_loop::registered_tool_active_for_request(
                        name,
                        request.enable_code_execution,
                        request.enable_shell,
                    )
                });
                let has_search = request.web_search_options.is_some();
                let has_agentic =
                    request.max_tool_rounds.is_some() || request.tool_dispatch_url.is_some();
                let has_input_files =
                    cfg!(feature = "code-execution") && is_chat && !request.input_files.is_empty();

                if is_chat
                    && !in_agentic_loop
                    && (has_search || has_tooling || has_agentic || has_input_files)
                {
                    Box::pin(agentic_loop::agentic_loop(self.clone(), *request)).await;
                } else if request.files.as_ref().is_some_and(|f| !f.is_empty()) {
                    // `request.files` is set but nothing would produce them. Reject rather than silently degrading to a plain chat.
                    let _ = request
                        .response
                        .send(crate::Response::ValidationError(
                            "request.files is set but no agentic surface is enabled \
                             (enable_code_execution / tools / web_search / \
                             enable_shell / max_tool_rounds / tool_dispatch_url). Files cannot be \
                             produced without one of these."
                                .into(),
                        ))
                        .await;
                } else {
                    if is_chat && !request.input_files.is_empty() {
                        agentic_loop::inject_input_files_message(&mut request);
                    }
                    self.add_request(*request).await;
                }
            }
            Request::ReIsq(level) => {
                if let Err(e) = get_mut_arcmutex!(self.pipeline).re_isq_model(level) {
                    warn!("ISQ requantization failed: {e:?}");
                }
            }
            Request::Calibration(req) => {
                let result = {
                    let mut pipeline = get_mut_arcmutex!(self.pipeline);
                    match &req.action {
                        crate::CalibrationAction::Start => pipeline
                            .begin_calibration()
                            .and_then(|()| pipeline.calibration_status()),
                        crate::CalibrationAction::Status => pipeline.calibration_status(),
                        crate::CalibrationAction::Apply { save_cimatrix } => {
                            pipeline.apply_calibration(save_cimatrix.clone())
                        }
                    }
                };
                if let Err(e) = &result {
                    warn!("Calibration request failed: {e:?}");
                }
                let _ = req.response.send(result).await;
            }
            Request::Tokenize(req) => self.tokenize_text(req).await,
            Request::Detokenize(req) => self.detokenize_text(req).await,
            Request::Terminate => (),
            Request::TerminateAllSeqsNextStep => {
                TERMINATE_ALL_NEXT_STEP.store(true, Ordering::SeqCst)
            }
        }
    }

    pub(super) async fn add_request(&self, request: NormalRequest) {
        let response = request.response.clone();
        if let Err(rejection) = self.admit_request(request) {
            response
                .send(*rejection)
                .await
                .unwrap_or_else(|_| warn!("Receiver disconnected"));
        }
    }

    fn admit_request(&self, mut request: NormalRequest) -> Result<(), Box<Response>> {
        if request.response.is_closed() {
            return Ok(());
        }
        let adapter_lease = match request.adapter.as_ref() {
            Some(selection) => match selection.lease() {
                Some(lease) => Some(lease.clone()),
                None => {
                    return Err(Box::new(Response::InternalError(
                        "request adapter selection was not pinned before admission".into(),
                    )));
                }
            },
            None => None,
        };
        self.validate_request_kind(&request)?;
        let extras = MessageExtras::of(&request.messages);
        let truncate_sequence = request.truncate_sequence;

        let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
        let preferred_tool_call_format = self.preferred_tool_call_format();
        let uses_channel_tool_call_strategy = matches!(
            preferred_tool_call_format,
            Some(ToolCallFormat::Harmony | ToolCallFormat::Atem)
        );
        let validates_forced_tool_choice = request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| choice.forced_function_name().is_some());
        let needs_tool_call_state =
            has_tools || uses_channel_tool_call_strategy || validates_forced_tool_choice;

        let (prompt_tokens, prompt_text) = self.render_prompt(
            request.messages,
            request.tools.as_deref(),
            request.tool_choice.as_ref(),
            preferred_tool_call_format,
        )?;
        if prompt_tokens.is_empty() {
            return Err(Box::new(Response::ValidationError(
                "Received an empty prompt.".into(),
            )));
        }
        if request.response.is_closed() {
            return Ok(());
        }
        let prompt_tokens = self.fit_prompt_to_context(
            prompt_tokens,
            truncate_sequence,
            request.sampling_params.max_len,
            request.id,
        )?;

        if let Some(defaults) = get_mut_arcmutex!(self.pipeline).generation_defaults() {
            request.sampling_params.fill_model_defaults(&defaults);
        }
        let topk = request
            .sampling_params
            .top_k
            .map(|x| x as i64)
            .unwrap_or(-1);
        let topp = request.sampling_params.top_p.unwrap_or(1.0);
        let minp = request.sampling_params.min_p.unwrap_or(0.0);
        let num_hidden_layers = get_mut_arcmutex!(self.pipeline)
            .get_metadata()
            .num_hidden_layers;

        let (stop_toks, stop_strings) =
            self.stop_criteria(request.sampling_params.stop_toks.as_ref())?;

        let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            request.sampling_params.n_choices,
            request.is_streaming,
            extras.is_chat,
            extras.best_of,
        )));

        let tokenizer = get_mut_arcmutex!(self.pipeline).tokenizer();

        let sampler = Sampler::new(
            Some(request.sampling_params.temperature.unwrap_or(1.0)),
            request.sampling_params.top_n_logprobs,
            tokenizer,
            request.sampling_params.frequency_penalty,
            request.sampling_params.presence_penalty,
            request.sampling_params.repetition_penalty,
            request.sampling_params.dry_params,
            topk,
            topp,
            minp,
            request.sampling_params.logits_bias.unwrap_or_default(),
            request.logits_processors.unwrap_or_default(),
        );
        let sampler = sampler.map_err(|e| Response::ValidationError(e.into()))?;

        if request.sampling_params.n_choices == 0 {
            return Err(Box::new(Response::ValidationError(
                "Number of choices must be greater than 0.".into(),
            )));
        }

        let mut added_seq = false;
        for response_index in 0..request.sampling_params.n_choices {
            if request.response.is_closed() {
                return Ok(());
            }
            let factory = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .llg_factory
                .clone();
            let recognizer = match Self::build_sequence_recognizer(&factory, &request.constraint) {
                Ok(recognizer) => recognizer,
                Err(err) => {
                    return Err(Box::new(Response::ValidationError(
                        format!("Invalid grammar. {err}").into(),
                    )));
                }
            };

            let block_size = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .cache_config
                .clone()
                .map(|conf| conf.block_size);

            let eos_toks = get_mut_arcmutex!(self.pipeline)
                .get_metadata()
                .eos_tok
                .clone();

            let seq_preallocated_cache = self.preallocated_seq_cache(prompt_tokens.len())?;

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time travel has occurred!");
            let tool_call_state = if needs_tool_call_state {
                Some(tool_call_state(
                    has_tools,
                    request.tool_choice.clone(),
                    request.tools.as_deref(),
                    preferred_tool_call_format,
                )?)
            } else {
                None
            };
            let mut seq = Sequence::new_waiting(
                prompt_tokens.clone(),
                prompt_text.clone(),
                *get_mut_arcmutex!(self.next_seq_id).deref(),
                now.as_millis(),
                num_hidden_layers,
                request.response.clone(),
                sampler.clone(),
                stop_toks.clone(),
                stop_strings.clone(),
                request.sampling_params.max_len,
                request.return_logprobs,
                get_mut_arcmutex!(self.pipeline).get_metadata().is_xlora,
                group.clone(),
                response_index,
                now.as_secs(),
                recognizer,
                request.suffix.clone(),
                if extras.echo_prompt {
                    Some(prompt_text.clone())
                } else {
                    None
                },
                extras.images.clone(),
                extras.audios.clone(),
                extras.videos.clone(),
                block_size,
                tool_call_state,
                extras.image_generation_format,
                extras.seq_step_type,
                extras.diffusion_params.clone(),
                extras.image_gen_save_file.clone(),
                seq_preallocated_cache,
                request.return_raw_logits,
                request.sampling_params.ignore_eos,
                eos_toks,
                choice_seed(request.seed, response_index),
            );
            if let Some(adapter_lease) = &adapter_lease {
                seq.bind_adapter(adapter_lease.clone());
            }

            self.enable_template_reasoning(&mut seq);
            self.prepare_multimodal_prompt(&mut seq)?;
            if request.response.is_closed() {
                return Ok(());
            }

            let prefill_cache = if seq.return_raw_logits {
                None
            } else {
                get_mut_arcmutex!(self.prefix_cacher)
                    .search_for_matching_cache(
                        seq.get_toks(),
                        seq.adapter_generation(),
                        seq.mm_features(),
                        seq.image_hashes(),
                        seq.audio_hashes(),
                        seq.video_hashes(),
                    )
                    .map_err(|e| Response::InternalError(e.into()))?
            };

            self.assign_recurrent_slot(&mut seq)?;

            if matches!(extras.seq_step_type, SeqStepType::PromptAndDecode) {
                self.logger.add_new_sequence();
            }
            if let Some(cache) = prefill_cache {
                seq = self.apply_prefix_cache_hit(seq, cache)?;
            }

            if request.response.is_closed() {
                self.release_recurrent_slot(&mut seq, "for abandoned request");
                return Ok(());
            }

            *get_mut_arcmutex!(self.next_seq_id) += 1;
            get_mut_arcmutex!(self.scheduler).add_seq(seq);
            added_seq = true;
        }
        if added_seq {
            self.pending_notify.notify_one();
        }
        Ok(())
    }

    // Rejects requests the loaded model cannot serve, before any tokenization work.
    fn validate_request_kind(&self, request: &NormalRequest) -> Result<(), Box<Response>> {
        let is_text_generation = matches!(
            &request.messages,
            RequestMessage::Chat { .. }
                | RequestMessage::Completion { .. }
                | RequestMessage::CompletionTokens(_)
                | RequestMessage::MultimodalChat { .. }
        );
        if is_text_generation && request.sampling_params.max_len == Some(0) {
            return Err(Box::new(Response::ValidationError(
                "max_tokens must be at least 1.".into(),
            )));
        }
        let is_chat = matches!(
            request.messages,
            RequestMessage::Chat { .. } | RequestMessage::MultimodalChat { .. }
        );
        if is_chat
            && !get_mut_arcmutex!(self.pipeline)
                .get_chat_template()
                .as_ref()
                .is_some_and(|ch_t| ch_t.has_chat_template())
        {
            return Err(Box::new(Response::ValidationError(
                        "Received messages for a model which does not have a chat template. Either use a different model or pass a single string as the prompt".into(),
                    )));
        }

        match (
            get_mut_arcmutex!(self.pipeline).category(),
            &request.messages,
        ) {
            (
                ModelCategory::Text | ModelCategory::Multimodal { .. },
                RequestMessage::Chat { .. }
                | RequestMessage::MultimodalChat { .. }
                | RequestMessage::Completion { .. }
                | RequestMessage::CompletionTokens(_),
            ) => Ok(()),
            (ModelCategory::Diffusion, RequestMessage::ImageGeneration { .. }) => Ok(()),
            (ModelCategory::Speech, RequestMessage::SpeechGeneration { .. }) => Ok(()),
            (
                ModelCategory::Embedding,
                RequestMessage::Embedding { .. } | RequestMessage::EmbeddingTokens { .. },
            ) => Ok(()),
            _ => Err(Box::new(Response::ValidationError(
                "Received a request incompatible for this model's category.".into(),
            ))),
        }
    }

    fn preferred_tool_call_format(&self) -> Option<ToolCallFormat> {
        let preferred = get_mut_arcmutex!(self.pipeline)
            .get_chat_template()
            .and_then(|chat_template| chat_template.tool_call_format());
        if preferred == Some(ToolCallFormat::Harmony)
            && !crate::reasoning_parsers::harmony::is_harmony_encoding_ready()
            && let Err(e) = tokio::task::block_in_place(|| {
                crate::reasoning_parsers::harmony::prewarm_harmony_encoding();
                Ok::<(), anyhow::Error>(())
            })
        {
            warn!("Failed to initialize Harmony encoding: {e}");
        }
        preferred
    }

    fn render_prompt(
        &self,
        messages: RequestMessage,
        tools: Option<&[crate::Tool]>,
        tool_choice: Option<&ToolChoice>,
        preferred_tool_call_format: Option<ToolCallFormat>,
    ) -> Result<(Vec<u32>, String), Box<Response>> {
        match messages {
            RequestMessage::Chat {
                messages,
                enable_thinking,
                reasoning_effort,
            }
            | RequestMessage::MultimodalChat {
                images: _,
                audios: _,
                videos: _,
                messages,
                enable_thinking,
                reasoning_effort,
            } => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tools = tools_for_chat_template(tools, tool_choice, preferred_tool_call_format);
                let template = pipeline.get_processor().process(
                    pipeline,
                    messages,
                    true,
                    true,
                    enable_thinking,
                    reasoning_effort,
                    tools,
                );
                template.map_err(|error| {
                    Box::new(if is_chat_template_request_error(&error) {
                        Response::ValidationError(error.into())
                    } else {
                        Response::InternalError(error.into())
                    })
                })
            }
            RequestMessage::Completion { text, .. }
            | RequestMessage::Embedding { prompt: text } => {
                let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
                    return Err(Box::new(Response::ValidationError(
                        "Completion requests require the pipeline to have a tokenizer".into(),
                    )));
                };
                let prompt = tokenizer
                    .encode_fast(text.clone(), true)
                    .map_err(anyhow::Error::msg);
                Ok((
                    prompt
                        .map_err(|e| Response::InternalError(e.into()))?
                        .get_ids()
                        .to_vec(),
                    text,
                ))
            }
            RequestMessage::ImageGeneration { prompt, .. }
            | RequestMessage::SpeechGeneration { prompt } => Ok((vec![u32::MAX], prompt)),
            RequestMessage::CompletionTokens(it)
            | RequestMessage::EmbeddingTokens { prompt: it } => {
                let Some(tokenizer) = &get_mut_arcmutex!(self.pipeline).tokenizer() else {
                    return Err(Box::new(Response::ValidationError(
                            "Completion requests w/ raw tokens require the pipeline to have a tokenizer".into(),
                        )));
                };
                if let Some(token_id) = first_unknown_token_id(tokenizer, &it) {
                    return Err(Box::new(Response::ValidationError(
                        format!(
                            "Token ID {token_id} is not present in the selected model tokenizer."
                        )
                        .into(),
                    )));
                }
                let prompt = tokenizer
                    .decode(&it, false)
                    .map_err(|e| anyhow::Error::msg(e.to_string()));
                Ok((it, prompt.map_err(|e| Response::InternalError(e.into()))?))
            }
        }
    }

    // Text/vision prompts over the limit keep their end (leaving room to generate); embeddings keep their start.
    fn fit_prompt_to_context(
        &self,
        prompt_tokens: Vec<u32>,
        truncate_sequence: bool,
        requested_max_len: Option<usize>,
        request_id: usize,
    ) -> Result<Vec<u32>, Box<Response>> {
        let (category, max_len) = {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            (pipeline.category(), pipeline.get_metadata().max_seq_len)
        };
        let is_generative = matches!(
            category,
            ModelCategory::Text | ModelCategory::Multimodal { .. }
        );
        if !(is_generative || matches!(category, ModelCategory::Embedding))
            || prompt_tokens.len() <= max_len
        {
            return Ok(prompt_tokens);
        }
        if !truncate_sequence {
            return Err(Box::new(Response::ValidationError(
                format!("Prompt sequence length is greater than {max_len}, perhaps consider using `truncate_sequence`?").into(),
            )));
        }
        let prompt_len = prompt_tokens.len();
        let currently_over = prompt_len - max_len;
        if is_generative {
            let sampling_max =
                requested_max_len.map_or(1, |sampling_max| sampling_max.min(max_len));
            let tokens_to_keep = max_len.saturating_sub(sampling_max);
            let slice_start = prompt_len.saturating_sub(tokens_to_keep);
            warn!(
                "Prompt for request {request_id} was {currently_over} tokens over the model maximum length. The first {slice_start} tokens were truncated to make space for generation."
            );
            Ok(prompt_tokens[slice_start..].to_vec())
        } else {
            warn!(
                "Prompt for request {request_id} was {currently_over} tokens over the model maximum length. The last {currently_over} tokens were truncated to make space for generation."
            );
            Ok(prompt_tokens[..max_len].to_vec())
        }
    }

    // A stop token that prefixes others (e.g. ` `) would fire early: rejected as an id, matched as a string otherwise.
    fn stop_criteria(
        &self,
        stop: Option<&StopTokens>,
    ) -> Result<(Vec<u32>, Vec<String>), Box<Response>> {
        match stop {
            None => Ok((vec![], vec![])),
            Some(StopTokens::Ids(ids)) => {
                let tok_env = get_mut_arcmutex!(self.pipeline).get_metadata().tok_env();
                if let Some(tok_env) = tok_env.as_ref() {
                    let tok_trie = tok_env.tok_trie();
                    for id in ids {
                        if tok_trie.has_extensions(tok_trie.token(*id)) {
                            return Err(Box::new(Response::ValidationError(
                                    format!("Stop token {:?} is also a prefix of other tokens and cannot be used as a stop token.", tok_trie.token_str(*id)).into(),
                                )));
                        }
                    }
                }
                Ok((ids.clone(), vec![]))
            }
            Some(StopTokens::Seqs(seqs)) => {
                let mut stop_toks = Vec::new();
                let mut stop_strings: Vec<String> = Vec::new();
                let (tok_env, tokenizer) = {
                    let pipeline = get_mut_arcmutex!(self.pipeline);
                    (pipeline.get_metadata().tok_env(), pipeline.tokenizer())
                };
                for stop_txt in seqs {
                    let Some(tokenizer) = &tokenizer else {
                        return Err(Box::new(Response::ValidationError(
                            "Completion requests require the pipeline to have a tokenizer".into(),
                        )));
                    };
                    let encoded = tokenizer.encode_fast(stop_txt.to_string(), true);
                    let toks = encoded.map_err(Response::InternalError)?.get_ids().to_vec();
                    let single_unambiguous = toks.len() == 1
                        && !tok_env.as_ref().is_some_and(|tok_env| {
                            let tok_trie = tok_env.tok_trie();
                            tok_trie.has_extensions(tok_trie.token(toks[0]))
                        });
                    if single_unambiguous {
                        stop_toks.push(toks[0]);
                    } else {
                        stop_strings.push(stop_txt.clone());
                    }
                }
                Ok((stop_toks, stop_strings))
            }
        }
    }

    // Per-sequence KV templates for layers that own a normal cache, sized to the prompt in CACHE_GROW_SIZE steps.
    fn preallocated_seq_cache(
        &self,
        n_tokens: usize,
    ) -> Result<Option<SeqPreallocatedCache>, Box<Response>> {
        let (metadata, device, needs_preallocated_cache) = {
            let pipeline = get_mut_arcmutex!(self.pipeline);
            if !matches!(
                pipeline.category(),
                ModelCategory::Text | ModelCategory::Multimodal { .. }
            ) {
                return Ok(None);
            }
            let needs_preallocated_cache: Vec<bool> = match pipeline.cache() {
                crate::pipeline::EitherCache::Normal(normal) => normal
                    .lock()
                    .unwrap()
                    .0
                    .iter()
                    .map(|cache| matches!(cache, KvCache::Normal { .. }))
                    .collect(),
                _ => Vec::new(),
            };
            (
                pipeline.get_metadata(),
                pipeline.device(),
                needs_preallocated_cache,
            )
        };
        let model_metadata = metadata
            .model_metadata
            .as_ref()
            .expect("If a model has a NormalCache it must have a model metadata");
        let required_blocks = n_tokens.div_ceil(NormalCache::CACHE_GROW_SIZE);
        let max_seq_len = required_blocks * NormalCache::CACHE_GROW_SIZE;
        let mut dtype = metadata.activation_dtype;
        // matches the f16 conversion KvCache::append applies on CPU
        if device.is_cpu() && dtype == candle_core::DType::F32 && crate::kv_cache::cpu_kv_f16() {
            dtype = candle_core::DType::F16;
        }
        let alloc = |shape: (usize, usize, usize, usize)| {
            Tensor::zeros(shape, dtype, &device).map_err(|err| {
                Box::new(Response::InternalError(
                    err.context("Failed to allocate preallocated KV cache.")
                        .into(),
                ))
            })
        };
        let mut layer_caches = Vec::with_capacity(model_metadata.num_layers());
        for layer_idx in 0..model_metadata.num_layers() {
            if !needs_preallocated_cache
                .get(layer_idx)
                .copied()
                .unwrap_or(false)
                || !model_metadata.uses_own_kv_cache_for_layer(layer_idx)
            {
                layer_caches.push(None);
                continue;
            }
            let num_kv_heads = model_metadata.num_kv_heads_for_layer(layer_idx);
            let k_shape = (
                1usize,
                num_kv_heads,
                max_seq_len,
                model_metadata.k_head_dim_for_layer(layer_idx),
            );
            let v_shape = (
                1usize,
                num_kv_heads,
                max_seq_len,
                model_metadata.v_head_dim_for_layer(layer_idx),
            );
            let k_seq_cache = alloc(k_shape)?;
            let v_seq_cache = if k_shape == v_shape {
                k_seq_cache.clone()
            } else {
                alloc(v_shape)?
            };
            layer_caches.push(Some((k_seq_cache, v_seq_cache)));
        }
        Ok(Some(layer_caches))
    }

    fn enable_template_reasoning(&self, seq: &mut Sequence) {
        use crate::reasoning_parsers::{
            ReasoningMode, TagReasoningContext, tag_based::THINK_OPEN_TAG,
        };

        let pipeline = get_mut_arcmutex!(self.pipeline);
        let Some(chat_template) = pipeline.get_chat_template() else {
            return;
        };
        let ctx = if chat_template.uses_channel_tags() && !chat_template.is_harmony_format() {
            // Gemma 4: <|channel>thought\n...<channel|>
            if seq.get_initial_prompt().contains("<|think|>") {
                TagReasoningContext::new_gemma_channel_with_implicit_thinking()
            } else {
                TagReasoningContext::new_gemma_channel()
            }
        } else if chat_template.uses_think_tags() {
            // DeepSeek, QwQ, SmolLM3: <think>...</think>
            if seq
                .get_initial_prompt()
                .trim_end()
                .ends_with(THINK_OPEN_TAG)
            {
                TagReasoningContext::new_in_think_block()
            } else {
                TagReasoningContext::new_think_tags()
            }
        } else if chat_template.uses_gemma_turns() {
            // Gemma-family thinking (MedGemma 1.5 etc): <unused94>thought\n...<unused95>; no-op if never emitted.
            TagReasoningContext::new_gemma_thought()
        } else {
            return;
        };
        seq.enable_reasoning(ReasoningMode::TagBased, Box::new(ctx));
    }

    // Runs before the prefix-cache lookup, keyed on the seq's own media so reconstructed multi-turn history is covered.
    fn prepare_multimodal_prompt(&self, seq: &mut Sequence) -> Result<(), Box<Response>> {
        if !(seq.has_images() || seq.has_audios() || seq.has_videos()) {
            return Ok(());
        }
        let pipeline = get_mut_arcmutex!(self.pipeline);
        pipeline
            .get_processor()
            .inputs_processor()
            .prepare_for_paged_prompt_planning(
                pipeline.tokenizer(),
                &mut [&mut *seq],
                &pipeline.device(),
                pipeline.get_input_processor_config(),
                None,
            )
            .map_err(|error| {
                Box::new(if is_inputs_processor_validation_error(&error) {
                    Response::ValidationError(error.into())
                } else {
                    Response::InternalError(error.into())
                })
            })
    }

    fn assign_recurrent_slot(&self, seq: &mut Sequence) -> Result<(), Box<Response>> {
        let pipeline = get_mut_arcmutex!(self.pipeline);
        if pipeline.get_metadata().no_kv_cache || !pipeline.cache().is_hybrid() {
            return Ok(());
        }
        let defer_initialization = pipeline.get_metadata().cache_config.is_some();
        let mut hybrid_cache = pipeline.cache().hybrid();
        let generation_before = hybrid_cache.recurrent_storage_generation();
        let slot = if defer_initialization {
            hybrid_cache.reserve_seq_uninitialized(*seq.id())
        } else {
            hybrid_cache.allocate_seq(*seq.id())
        };
        let storage_changed = hybrid_cache.recurrent_storage_generation() != generation_before;
        drop(hybrid_cache);
        if storage_changed {
            pipeline.cleanup_cuda_graphs();
            if let Some(ctx) = &self.graph_precapture_ctx {
                pipeline.precapture_cuda_decode_graphs(ctx);
            }
        }
        drop(pipeline);
        let slot_idx = slot.map_err(|err| Box::new(Response::InternalError(err.into())))?;
        seq.set_recurrent_state_idx(Some(slot_idx));
        Ok(())
    }

    fn release_recurrent_slot(&self, seq: &mut Sequence, context: &str) {
        let Some(slot_idx) = seq.recurrent_state_idx() else {
            return;
        };
        let pipeline = get_mut_arcmutex!(self.pipeline);
        if pipeline.cache().is_hybrid() {
            match pipeline.cache().hybrid().release_seq(*seq.id(), slot_idx) {
                Ok(_) => seq.set_recurrent_state_idx(None),
                Err(err) => tracing::error!("Failed to release recurrent state {context}: {err}"),
            }
        }
    }

    fn apply_prefix_cache_hit(
        &self,
        mut seq: Sequence,
        cache: MatchingCache,
    ) -> Result<Sequence, Box<Response>> {
        let MatchingCache::Normal {
            normal,
            recurrent_snapshots,
            images_to_keep,
            audios_to_keep,
            video_frames_to_keep,
            toks,
            offset,
        } = cache;
        if seq.record_prefix_cache_hit() {
            self.logger.add_prefix_cache_hit();
        }

        if let (Some(snapshots), Some(slot_idx)) = (recurrent_snapshots, seq.recurrent_state_idx())
        {
            let restore_result = {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                if pipeline.cache().is_hybrid() {
                    pipeline.cache().hybrid().restore_recurrent_state(
                        *seq.id(),
                        slot_idx,
                        &snapshots,
                    )
                } else {
                    Ok(())
                }
            };
            if let Err(err) = restore_result {
                self.release_recurrent_slot(&mut seq, "after restore error");
                return Err(Box::new(Response::InternalError(err.into())));
            }
        }

        if !get_mut_arcmutex!(self.pipeline)
            .get_processor()
            .retain_prefix_cached_images()
        {
            seq.keep_num_images(images_to_keep);
        }
        seq.keep_num_audios(audios_to_keep);
        seq.keep_num_video_frames(video_frames_to_keep);
        Ok(seq.prefill_v2_normal(normal, toks, offset))
    }

    async fn tokenize_text(&self, request: TokenizationRequest) {
        match request.text {
            Either::Left(messages) => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tools = request.tools.unwrap_or_default();
                let template = pipeline.get_processor().process(
                    pipeline,
                    messages,
                    request.add_generation_prompt,
                    request.add_special_tokens,
                    request.enable_thinking,
                    request.reasoning_effort,
                    tools,
                );
                let toks = match template {
                    Ok((toks, _)) => toks,
                    Err(e) => {
                        request
                            .response
                            .send(Err(e))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    }
                };
                request
                    .response
                    .send(Ok(toks))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
            }
            Either::Right(text) => {
                let pipeline = &*get_mut_arcmutex!(self.pipeline);
                let tokenizer = pipeline.tokenizer();
                let tokenizer = match tokenizer {
                    Some(tokenizer) => tokenizer,
                    None => {
                        request
                            .response
                            .send(Err(anyhow::Error::msg(
                                "Pipeline does not include a tokenizer.",
                            )))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    }
                };
                let toks = tokenizer.encode_fast(text, request.add_special_tokens);
                let toks = match toks {
                    Ok(tokenizer) => tokenizer,
                    Err(e) => {
                        request
                            .response
                            .send(Err(anyhow::Error::msg(e)))
                            .await
                            .unwrap_or_else(|_| warn!("Receiver disconnected"));
                        return;
                    }
                };
                request
                    .response
                    .send(Ok(toks.get_ids().to_vec()))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
            }
        };
    }

    async fn detokenize_text(&self, request: DetokenizationRequest) {
        let pipeline = &*get_mut_arcmutex!(self.pipeline);
        let tokenizer = pipeline.tokenizer();
        let tokenizer = match tokenizer {
            Some(tokenizer) => tokenizer,
            None => {
                request
                    .response
                    .send(Err(anyhow::Error::msg(
                        "Pipeline does not include a tokenizer.",
                    )))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
                return;
            }
        };
        let txt = tokenizer.decode(&request.tokens, request.skip_special_tokens);
        let txt = match txt {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                request
                    .response
                    .send(Err(anyhow::Error::msg(e)))
                    .await
                    .unwrap_or_else(|_| warn!("Receiver disconnected"));
                return;
            }
        };
        request
            .response
            .send(Ok(txt))
            .await
            .unwrap_or_else(|_| warn!("Receiver disconnected"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::chat_template::{ChatTemplateValue, apply_chat_template_to};
    use crate::{Function, Tool, ToolType};
    use ahash::AHashMap;
    use indexmap::IndexMap;
    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    #[test]
    fn choice_seeds_are_stable_and_distinct() {
        assert_eq!(choice_seed(None, 0), None);
        assert_eq!(choice_seed(Some(42), 0), Some(42));
        assert_eq!(choice_seed(Some(42), 1), Some(43));
        assert_eq!(choice_seed(Some(u64::MAX), 1), Some(0));
    }

    #[test]
    fn raw_token_ids_must_exist_in_the_selected_tokenizer() {
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(AHashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap(),
        );

        assert_eq!(first_unknown_token_id(&tokenizer, &[0, 1]), None);
        assert_eq!(first_unknown_token_id(&tokenizer, &[0, 2]), Some(2));
    }

    fn tool() -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        }
    }

    #[test]
    fn atem_tool_choice_none_omits_tools_from_the_rendered_prompt() {
        let tools = vec![tool()];
        let rendered_tools = tools_for_chat_template(
            Some(&tools),
            Some(&ToolChoice::None),
            Some(ToolCallFormat::Atem),
        );
        let messages = vec![IndexMap::from([
            ("role".to_string(), Either::Left("user".to_string())),
            ("content".to_string(), Either::Left("hello".to_string())),
        ])];
        let template = ChatTemplateValue(Either::Left(
            "{% if tools %}tools{% else %}no-tools{% endif %}<atem:function_calls><atem:invoke"
                .to_string(),
        ));

        let rendered = apply_chat_template_to(
            messages,
            true,
            None,
            None,
            &template,
            None,
            None,
            None,
            rendered_tools,
        )
        .unwrap();

        assert!(rendered.starts_with("no-tools"));
        let qwen_tools = tools_for_chat_template(
            Some(&tools),
            Some(&ToolChoice::None),
            Some(ToolCallFormat::Qwen),
        );
        assert_eq!(qwen_tools.len(), 1);
        assert_eq!(qwen_tools[0].function.name, "get_weather");
    }
}
