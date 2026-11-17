//! EXAONE 4.5: Qwen2.5-VL's windowed vision tower with GQA attention beside the EXAONE 4 text stack, on plain 1D
//! positions (no M-RoPE), with `<vision><|image_pad|></vision>` placeholders.

use inference_quant::ShardedVarBuilder;
use inference_tensor::Result;
use serde::Serialize;

use crate::{
    decoder::{CausalLm, MtpSpec, exaone4},
    model::NormalLoadingMetadata,
    paged_attention::AttentionImplementation,
    qwen2vl::{QwenVlModel, QwenVlVision},
    vision::qwen2_5_vl::{Qwen2_5VLVisionModel, VisionConfig},
};

const TEXT_PREFIX: &str = "model.language_model";
const VISION_PREFIX: &str = "model.visual";
// DeepSeek's layout under EXAONE's names: `fc(concat(norm(embedding), norm(hidden)))`, one post-norm layer, a norm
const MTP: MtpSpec = MtpSpec {
    embedding_norm: "mtp.pre_fc_norm_embedding",
    hidden_norm: "mtp.pre_fc_norm_hidden",
    projection: "mtp.fc",
    block: "mtp.layers.0",
    final_norm: Some("mtp.norm"),
    dense_block: true,
    mask_first_position: false,
    at_root: true,
};

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct Config {
    pub text_config: exaone4::Config,
    #[serde(skip_serializing)]
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Injected by the loader when `--mtp` asks for the checkpoint's MTP layer.
    #[serde(default, rename = "_inference_mtp")]
    pub mtp: bool,
}

impl Config {
    pub fn decoder_spec(&self) -> Result<crate::decoder::DecoderSpec> {
        let mut spec = self.text_config.decoder_spec()?;
        spec.module_names.model = TEXT_PREFIX;
        spec.tie_word_embeddings = self.tie_word_embeddings;
        if self.mtp {
            if self.text_config.num_nextn_predict_layers == 0 {
                inference_tensor::bail!(
                    "`--mtp` asked for EXAONE 4.5's MTP layer, which this config has none of"
                );
            }
            spec.mtp = Some(MTP);
        }
        Ok(spec)
    }
}

pub type Exaone4_5Model = QwenVlModel<Qwen2_5VLVisionModel>;

pub fn exaone4_5_model(
    cfg: &Config,
    vb: ShardedVarBuilder,
    is_gptx: bool,
    normal_loading_metadata: NormalLoadingMetadata,
    attention_mechanism: AttentionImplementation,
) -> Result<Exaone4_5Model> {
    let vision = <Qwen2_5VLVisionModel as QwenVlVision>::new(
        &cfg.vision_config,
        vb.pp(VISION_PREFIX)
            .without_lora_registry()
            .set_device(normal_loading_metadata.real_device.clone()),
        &normal_loading_metadata.mapper.get_comm_for(0)?,
    )?;
    let text = CausalLm::new(
        &cfg.decoder_spec()?,
        vb,
        is_gptx,
        normal_loading_metadata,
        attention_mechanism,
    )?;
    Ok(QwenVlModel::assemble(
        text,
        true,
        vision,
        VISION_PREFIX,
        cfg.vision_config.spatial_merge_size,
        (cfg.image_token_id, cfg.video_token_id),
    ))
}
