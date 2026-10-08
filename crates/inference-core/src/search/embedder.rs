use std::sync::Arc;

use anyhow::Context;
use inference_quant::log::once_log_info;
use inference_tensor::{DType, Device, Error as E};
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;

use crate::pipeline::ForwardInputsResult;
use crate::pipeline::LoadOptions;
use crate::{
    AutoDeviceMapParams, DeviceMapSetting, EmbeddingLoaderBuilder, EmbeddingSpecificConfig,
    ModelDType, Pipeline, TokenSource,
    embedding_models::inputs_processor::{ModelInputs, make_prompt_chunk},
    engine::SearchEmbeddingModel,
    get_mut_arcmutex,
    pipeline::EmbeddingLoadContext,
};

const EMBEDDING_BATCH: usize = 64;
/// Files that must be cached for the search embedding model to load without hitting the network.
const SEARCH_MODEL_CORE_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

/// The embedding model web search ranks fetched pages with.
pub struct SearchEmbedder {
    model: Arc<TokioMutex<dyn Pipeline + Send + Sync>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    has_causal_attention: bool,
    max_seq_len: usize,
}

impl SearchEmbedder {
    pub fn new(model: SearchEmbeddingModel, runner_device: &Device) -> anyhow::Result<Self> {
        let model_id = model.hf_model_id().to_string();

        // Quiet load when cached; on first run, surface download progress for the model files.
        let cached =
            crate::pipeline::hf::files_cached_locally(&model_id, "main", SEARCH_MODEL_CORE_FILES);
        if cached {
            once_log_info(format!("Loading embedding model ({model_id})."));
        } else {
            once_log_info(format!(
                "Downloading embedding model ({model_id}); cached for future runs."
            ));
        }

        let loader = EmbeddingLoaderBuilder::new(
            EmbeddingSpecificConfig::default(),
            None,
            Some(model_id.clone()),
        )
        .with_load_context(EmbeddingLoadContext::Search)
        .build(None)?;

        let options = LoadOptions {
            dtype: &ModelDType::Auto,
            device: runner_device,
            silent: cached,
            mapper: DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
            in_situ_quant: None,
            paged_attn_config: None,
        };
        let pipeline = loader.load_model_from_hf(None, TokenSource::CacheToken, options)?;

        let guard = get_mut_arcmutex!(pipeline);
        let tokenizer = guard
            .tokenizer()
            .with_context(|| "Embedding model did not expose a tokenizer")?
            .clone();
        let device = guard.device();
        let max_seq_len = guard.get_metadata().max_seq_len;
        drop(guard);

        Ok(Self {
            model: pipeline,
            tokenizer,
            device,
            has_causal_attention: false,
            max_seq_len,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn embed(&mut self, prompts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        let encoded: Vec<(usize, Vec<u32>)> = prompts
            .iter()
            .enumerate()
            .map(|(idx, prompt)| {
                let encoding = self
                    .tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(E::msg)?;
                Ok((idx, encoding.get_ids().to_vec()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut outputs = vec![Vec::new(); prompts.len()];
        // make_prompt_chunk packs variable-length prompts via flash attention cumulative sequence lengths.
        for chunk_entries in encoded.chunks(EMBEDDING_BATCH) {
            let slices: Vec<&[u32]> = chunk_entries
                .iter()
                .map(|(_, ids)| ids.as_slice())
                .collect();
            let chunk = make_prompt_chunk(
                0,
                slices,
                &self.device,
                None,
                self.has_causal_attention,
                None,
            )?;
            let inputs = Box::new(ModelInputs {
                input_ids: chunk.input,
                flash_meta: chunk.flash_meta,
            });
            let mut pipeline = get_mut_arcmutex!(self.model);
            let ForwardInputsResult::Embeddings { embeddings } =
                pipeline.forward_inputs(inputs, false)?
            else {
                anyhow::bail!("Embedding pipeline returned non-embedding output");
            };
            drop(pipeline);
            let vecs = embeddings
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            for ((idx, _), embedding) in chunk_entries.iter().zip(vecs) {
                outputs[*idx] = embedding;
            }
        }

        Ok(outputs)
    }
}
