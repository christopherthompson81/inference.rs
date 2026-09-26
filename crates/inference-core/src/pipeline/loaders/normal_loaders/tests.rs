use super::*;

fn loading_metadata(rope_pairing: Option<RopePairing>) -> NormalLoadingMetadata {
    NormalLoadingMetadata {
        mapper: Box::new(crate::device_map::DummyDeviceMapper {
            nm_device: Device::Cpu,
        }),
        loading_isq: false,
        real_device: Device::Cpu,
        multi_progress: Arc::new(crate::utils::progress::new_multi_progress()),
        matformer_slicing_config: None,
        rope_pairing,
    }
}

#[test]
fn persisted_qk_rope_layout_overrides_loader_default() -> Result<()> {
    let loader = LlamaLoader;
    assert!(loader.is_gptx("{}")?);
    assert!(!loader.is_gptx_for(
        r#"{"_inference_qk_rope_layout":"adjacent"}"#,
        &loading_metadata(None),
    )?);
    assert!(loader.is_gptx_for(
        r#"{"_inference_qk_rope_layout":"half_split"}"#,
        &loading_metadata(None),
    )?);
    assert!(!loader.is_gptx_for("{}", &loading_metadata(Some(RopePairing::Adjacent)))?);
    Ok(())
}

const PROMOTED_TENSORS: [&str; 3] = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "lm_head.bias",
];
const NON_PROMOTED_TENSORS: [&str; 10] = [
    "embed_tokens.weight",
    "prefix.model.embed_tokens.weight",
    "model.embed_tokens.bias",
    "model.embed_tokens.extra.weight",
    "model.embed_tokens.weight.extra",
    "model.layers.0.model.embed_tokens.weight",
    "model.lm_head.weight",
    "lm_head",
    "lm_head.weight.extra",
    "model.layers.0.lm_head.weight",
];

fn assert_promoted_isq_predicates(loader_name: &str, loader: &dyn IsqModelLoader, config: &str) {
    let predicates = loader.promoted_isq_predicates(config).unwrap();

    for tensor in PROMOTED_TENSORS {
        assert!(
            predicates
                .iter()
                .any(|predicate| predicate.is_match(tensor)),
            "{loader_name} did not promote {tensor}"
        );
    }
    for tensor in NON_PROMOTED_TENSORS {
        assert!(
            predicates
                .iter()
                .all(|predicate| !predicate.is_match(tensor)),
            "{loader_name} promoted lookalike tensor {tensor}"
        );
    }
}

const FUSED_EXPERT_PROJECTIONS: &[&str] = &["gate_proj", "up_proj", "down_proj"];
const GPT_OSS_EXPERT_PROJECTIONS: &[&str] = &["gate_up_proj", "gate_proj", "up_proj", "down_proj"];
const GRANITE_EXPERT_PROJECTIONS: &[&str] = &["input_linear", "output_linear"];

fn assert_expert_isq_predicates(
    loader_name: &str,
    loader: &dyn IsqModelLoader,
    config: &str,
    prefix: &str,
    projections: &[&str],
) -> Result<()> {
    let predicate_sets = [
        ("isq", loader.isq_layer_regexes(config)?),
        ("immediate", loader.immediate_isq_predicates(config)?),
        ("moqe", loader.isq_layer_regexes_moqe(config)?),
        (
            "immediate moqe",
            loader.immediate_isq_predicates_moqe(config)?,
        ),
    ];
    for (kind, predicates) in predicate_sets {
        for projection in projections {
            let key = format!("{prefix}.{projection}.weight");
            assert!(
                predicates.iter().any(|predicate| predicate.is_match(&key)),
                "{loader_name} {kind} predicates did not match {key}"
            );
        }
    }
    Ok(())
}

fn assert_default_isq_paths(
    loader_name: &str,
    loader: &dyn IsqModelLoader,
    config: &str,
    expected: &[&str],
    rejected: &[&str],
) -> Result<()> {
    for (kind, predicates) in [
        ("isq", loader.isq_layer_regexes(config)?),
        ("immediate", loader.immediate_isq_predicates(config)?),
    ] {
        for path in expected {
            assert!(
                predicates.iter().any(|predicate| predicate.is_match(path)),
                "{loader_name} {kind} predicates did not match {path}"
            );
        }
        for path in rejected {
            assert!(
                predicates.iter().all(|predicate| !predicate.is_match(path)),
                "{loader_name} {kind} predicates matched {path}"
            );
        }
    }
    Ok(())
}

fn assert_moqe_isq_paths(
    loader_name: &str,
    loader: &dyn IsqModelLoader,
    expected: &[&str],
    rejected: &[&str],
) -> Result<()> {
    for (kind, predicates) in [
        ("moqe", loader.isq_layer_regexes_moqe("")?),
        ("immediate moqe", loader.immediate_isq_predicates_moqe("")?),
    ] {
        for path in expected {
            assert!(
                predicates.iter().any(|predicate| predicate.is_match(path)),
                "{loader_name} {kind} predicates did not match {path}"
            );
        }
        for path in rejected {
            assert!(
                predicates.iter().all(|predicate| !predicate.is_match(path)),
                "{loader_name} {kind} predicates matched {path}"
            );
        }
    }
    Ok(())
}

fn deepseek_moe_config() -> String {
    serde_json::json!({
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "n_shared_experts": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "rope_scaling": null,
        "attention_bias": false,
        "q_lora_rank": null,
        "qk_rope_head_dim": 2,
        "kv_lora_rank": 2,
        "v_head_dim": 2,
        "qk_nope_head_dim": 2,
        "quantization_config": null,
        "n_group": 1,
        "topk_group": 1
    })
    .to_string()
}

fn glm4_moe_config() -> String {
    serde_json::json!({
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "q_lora_rank": 2,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "partial_rotary_factor": 1.0,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "head_dim": null,
        "quantization_config": null
    })
    .to_string()
}

struct ExpertIsqCase<'a> {
    name: &'static str,
    loader: Box<dyn IsqModelLoader>,
    config: &'a str,
    prefix: &'static str,
    projections: &'static [&'static str],
}

struct NativeIsqNamespaceCase<'a> {
    name: &'static str,
    loader: Box<dyn IsqModelLoader>,
    config: &'a str,
    paths: &'static [&'static str],
}

#[test]
fn native_gguf_adapter_isq_namespace_matrix() -> Result<()> {
    let deepseek_config = deepseek_moe_config();
    let glm4_moe_config = glm4_moe_config();
    let cases = vec![
        NativeIsqNamespaceCase {
            name: "Mistral",
            loader: Box::new(MistralLoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Gemma",
            loader: Box::new(GemmaLoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Mixtral",
            loader: Box::new(MixtralLoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.block_sparse_moe.experts.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Llama",
            loader: Box::new(LlamaLoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Phi2",
            loader: Box::new(Phi2Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.fc1.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Phi3",
            loader: Box::new(Phi3Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.qkv_proj.weight",
                "model.layers.0.mlp.gate_up_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Qwen2",
            loader: Box::new(Qwen2Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Gemma2",
            loader: Box::new(Gemma2Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Starcoder2",
            loader: Box::new(Starcoder2Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.c_fc.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Phi3.5 MoE",
            loader: Box::new(Phi3_5MoELoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.block_sparse_moe.experts.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "DeepSeek V2",
            loader: Box::new(DeepSeekV2Loader),
            config: &deepseek_config,
            paths: &[
                "model.layers.0.self_attn.k_b_proj.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "DeepSeek V3",
            loader: Box::new(DeepSeekV3Loader),
            config: &deepseek_config,
            paths: &[
                "model.layers.0.self_attn.v_b_proj.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Qwen3",
            loader: Box::new(Qwen3Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "GLM4",
            loader: Box::new(GLM4Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_up_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "GLM4 MoE Lite",
            loader: Box::new(GLM4MoeLiteLoader),
            config: &glm4_moe_config,
            paths: &[
                "model.layers.0.self_attn.k_b_proj.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "GLM4 MoE",
            loader: Box::new(GLM4MoeLoader),
            config: &glm4_moe_config,
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
                "model.layers.0.mlp.shared_experts.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Qwen3 MoE",
            loader: Box::new(Qwen3MoELoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "SmolLM3",
            loader: Box::new(SmolLm3Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Granite",
            loader: Box::new(GraniteMoeHybridLoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mamba.in_proj.weight",
                "model.layers.0.shared_mlp.input_linear.weight",
                "model.layers.0.block_sparse_moe.input_linear.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "GPT-OSS",
            loader: Box::new(GptOssLoader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                "model.layers.0.mlp.experts.down_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "HunYuan dense",
            loader: Box::new(HunYuanDenseV1Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "HunYuan MoE",
            loader: Box::new(HunYuanMoEV1Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
                "model.layers.0.mlp.shared_mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Qwen3Next",
            loader: Box::new(Qwen3NextLoader),
            config: "",
            paths: &[
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                "model.layers.0.linear_attn.in_proj_z.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "Qwen3.5",
            loader: Box::new(Qwen3_5TextLoader),
            config: "",
            paths: &[
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                "model.language_model.layers.0.mlp.gate_proj.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "LFM2",
            loader: Box::new(Lfm2Loader),
            config: "",
            paths: &[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.conv.in_proj.weight",
                "model.layers.0.feed_forward.w1.weight",
            ],
        },
        NativeIsqNamespaceCase {
            name: "LFM2 MoE",
            loader: Box::new(Lfm2Loader),
            config: "",
            paths: &[
                "model.layers.0.conv.out_proj.weight",
                "model.layers.0.feed_forward.experts.gate_proj.weight",
            ],
        },
    ];

    for case in cases {
        let promoted = case.loader.promoted_isq_predicates(case.config)?;
        let embedding = if case.name == "Qwen3.5" {
            "model.language_model.embed_tokens.weight"
        } else {
            "model.embed_tokens.weight"
        };
        for path in [embedding, "lm_head.weight"] {
            assert!(
                promoted.iter().any(|predicate| predicate.is_match(path)),
                "{} promoted predicates did not match {path}",
                case.name
            );
        }
        assert_default_isq_paths(
            case.name,
            case.loader.as_ref(),
            case.config,
            case.paths,
            &[],
        )?;
    }
    Ok(())
}

#[test]
fn normal_moe_loaders_match_canonical_expert_stacks() -> Result<()> {
    let deepseek_config = deepseek_moe_config();
    let glm4_config = glm4_moe_config();
    let cases = [
        ExpertIsqCase {
            name: "MixtralLoader",
            loader: Box::new(MixtralLoader),
            config: "",
            prefix: "model.layers.0.block_sparse_moe.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "Phi3_5MoELoader",
            loader: Box::new(Phi3_5MoELoader),
            config: "",
            prefix: "model.layers.0.block_sparse_moe.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "DeepSeekV2Loader",
            loader: Box::new(DeepSeekV2Loader),
            config: &deepseek_config,
            prefix: "model.layers.0.mlp.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "DeepSeekV3Loader",
            loader: Box::new(DeepSeekV3Loader),
            config: &deepseek_config,
            prefix: "model.layers.0.mlp.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "HunYuanMoEV1Loader",
            loader: Box::new(HunYuanMoEV1Loader),
            config: "",
            prefix: "model.layers.0.mlp.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "GLM4MoeLiteLoader",
            loader: Box::new(GLM4MoeLiteLoader),
            config: &glm4_config,
            prefix: "model.layers.0.mlp.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "GLM4MoeLoader",
            loader: Box::new(GLM4MoeLoader),
            config: &glm4_config,
            prefix: "model.layers.0.mlp.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "Qwen3MoELoader",
            loader: Box::new(Qwen3MoELoader),
            config: "",
            prefix: "model.layers.0.mlp.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "Qwen3NextLoader",
            loader: Box::new(Qwen3NextLoader),
            config: "",
            prefix: "model.layers.0.mlp.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "Lfm2Loader",
            loader: Box::new(Lfm2Loader),
            config: "",
            prefix: "model.layers.0.feed_forward.experts",
            projections: FUSED_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "GraniteMoeHybridLoader",
            loader: Box::new(GraniteMoeHybridLoader),
            config: "",
            prefix: "model.layers.0.block_sparse_moe",
            projections: GRANITE_EXPERT_PROJECTIONS,
        },
        ExpertIsqCase {
            name: "GptOssLoader",
            loader: Box::new(GptOssLoader),
            config: "",
            prefix: "model.layers.0.mlp.experts",
            projections: GPT_OSS_EXPERT_PROJECTIONS,
        },
    ];

    for case in cases {
        assert_expert_isq_predicates(
            case.name,
            case.loader.as_ref(),
            case.config,
            case.prefix,
            case.projections,
        )?;
    }
    Ok(())
}

#[test]
fn native_gguf_isq_predicates_match_model_linear_sites() -> Result<()> {
    assert_default_isq_paths(
        "Starcoder2Loader",
        &Starcoder2Loader,
        "",
        &["model.layers.0.mlp.c_fc.weight"],
        &[
            "model.layers.0.mlp.fc1.weight",
            "model.layers.0.mlp.c_fc_extra.weight",
        ],
    )?;
    for (name, loader) in [
        ("Phi3Loader", &Phi3Loader as &dyn IsqModelLoader),
        ("GLM4Loader", &GLM4Loader as &dyn IsqModelLoader),
    ] {
        assert_default_isq_paths(
            name,
            loader,
            "",
            &["model.layers.0.mlp.gate_up_proj.weight"],
            &[
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.gate_up_projector.weight",
            ],
        )?;
    }

    let deepseek_config = deepseek_moe_config();
    let glm4_config = glm4_moe_config();
    for (name, loader, config) in [
        (
            "DeepSeekV2Loader",
            &DeepSeekV2Loader as &dyn IsqModelLoader,
            deepseek_config.as_str(),
        ),
        (
            "DeepSeekV3Loader",
            &DeepSeekV3Loader as &dyn IsqModelLoader,
            deepseek_config.as_str(),
        ),
        (
            "GLM4MoeLiteLoader",
            &GLM4MoeLiteLoader as &dyn IsqModelLoader,
            glm4_config.as_str(),
        ),
    ] {
        assert_default_isq_paths(
            name,
            loader,
            config,
            &[
                "model.layers.0.self_attn.kv_b_proj.weight",
                "model.layers.0.self_attn.k_b_proj.weight",
                "model.layers.0.self_attn.v_b_proj.weight",
            ],
            &[
                "model.layers.0.self_attn.key_b_proj.weight",
                "model.layers.0.self_attn.k_b_projector.weight",
            ],
        )?;
    }

    assert_default_isq_paths(
        "GraniteMoeHybridLoader",
        &GraniteMoeHybridLoader,
        "",
        &[
            "model.layers.0.mamba.in_proj.weight",
            "model.layers.0.mamba.out_proj.weight",
        ],
        &[
            "model.layers.0.mamba.conv1d.weight",
            "model.layers.0.mamba.input_proj.weight",
        ],
    )?;

    assert_default_isq_paths(
        "Qwen3NextLoader",
        &Qwen3NextLoader,
        "",
        &[
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_z.weight",
            "model.layers.0.linear_attn.in_proj_ba.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            "model.layers.0.linear_attn.in_proj_a.weight",
        ],
        &[
            "model.layers.0.linear_attn.in_proj_qkvzz.weight",
            "model.layers.0.linear_attn.in_proj_beta.weight",
        ],
    )?;

    Ok(())
}

#[test]
fn native_gguf_moqe_predicates_exclude_the_shared_trunk() -> Result<()> {
    assert_moqe_isq_paths(
        "MixtralLoader",
        &MixtralLoader,
        &[
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.gate_proj.weight",
        ],
        &[
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
        ],
    )?;

    assert_moqe_isq_paths(
        "Phi3_5MoELoader",
        &Phi3_5MoELoader,
        &[
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.gate_proj.weight",
        ],
        &[
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
        ],
    )?;

    for (name, loader) in [
        ("DeepSeekV2Loader", &DeepSeekV2Loader as &dyn IsqModelLoader),
        ("DeepSeekV3Loader", &DeepSeekV3Loader as &dyn IsqModelLoader),
        (
            "HunYuanMoEV1Loader",
            &HunYuanMoEV1Loader as &dyn IsqModelLoader,
        ),
        (
            "GLM4MoeLiteLoader",
            &GLM4MoeLiteLoader as &dyn IsqModelLoader,
        ),
        ("GLM4MoeLoader", &GLM4MoeLoader as &dyn IsqModelLoader),
        ("Qwen3MoELoader", &Qwen3MoELoader as &dyn IsqModelLoader),
        ("Qwen3NextLoader", &Qwen3NextLoader as &dyn IsqModelLoader),
    ] {
        assert_moqe_isq_paths(
            name,
            loader,
            &[
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.mlp.experts.gate_proj.weight",
            ],
            &[
                "lm_head.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.shared_mlp.gate_proj.weight",
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
                "model.layers.0.mlp.shared_experts.gate_proj.weight",
            ],
        )?;
    }

    assert_moqe_isq_paths(
        "GraniteMoeHybridLoader",
        &GraniteMoeHybridLoader,
        &[
            "model.layers.0.block_sparse_moe.input_linear.weight",
            "model.layers.0.block_sparse_moe.output_linear.weight",
        ],
        &[
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.shared_mlp.input_linear.weight",
            "model.layers.0.block_sparse_moe.router.weight",
        ],
    )?;
    assert_moqe_isq_paths(
        "GptOssLoader",
        &GptOssLoader,
        &[
            "model.layers.0.mlp.experts.gate_up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
        ],
        &[
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.router.weight",
        ],
    )?;
    assert_moqe_isq_paths(
        "Lfm2Loader",
        &Lfm2Loader,
        &[
            "model.layers.0.feed_forward.experts.0.w1.weight",
            "model.layers.0.feed_forward.experts.gate_proj.weight",
        ],
        &[
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.feed_forward.gate.weight",
        ],
    )?;

    Ok(())
}

#[test]
fn concrete_normal_loaders_scope_promoted_isq_tensors() {
    let loaders: [(&str, &dyn IsqModelLoader); 25] = [
        ("MistralLoader", &MistralLoader),
        ("GemmaLoader", &GemmaLoader),
        ("LlamaLoader", &LlamaLoader),
        ("MixtralLoader", &MixtralLoader),
        ("Phi2Loader", &Phi2Loader),
        ("Phi3Loader", &Phi3Loader),
        ("Qwen2Loader", &Qwen2Loader),
        ("Gemma2Loader", &Gemma2Loader),
        ("Starcoder2Loader", &Starcoder2Loader),
        ("Phi3_5MoELoader", &Phi3_5MoELoader),
        ("DeepSeekV2Loader", &DeepSeekV2Loader),
        ("DeepSeekV3Loader", &DeepSeekV3Loader),
        ("Qwen3Loader", &Qwen3Loader),
        ("HunYuanDenseV1Loader", &HunYuanDenseV1Loader),
        ("HunYuanMoEV1Loader", &HunYuanMoEV1Loader),
        ("GLM4Loader", &GLM4Loader),
        ("GLM4MoeLiteLoader", &GLM4MoeLiteLoader),
        ("GLM4MoeLoader", &GLM4MoeLoader),
        ("Qwen3MoELoader", &Qwen3MoELoader),
        ("SmolLm3Loader", &SmolLm3Loader),
        ("GraniteMoeHybridLoader", &GraniteMoeHybridLoader),
        ("GptOssLoader", &GptOssLoader),
        ("Qwen3NextLoader", &Qwen3NextLoader),
        ("Qwen3_5TextLoader", &Qwen3_5TextLoader),
        ("Lfm2Loader", &Lfm2Loader),
    ];

    for (loader_name, loader) in loaders {
        assert_promoted_isq_predicates(loader_name, loader, "");
    }
}

#[test]
fn auto_normal_loader_delegates_promoted_isq_predicates() {
    let config = r#"{"architectures":["LlamaForCausalLM"]}"#;

    assert_promoted_isq_predicates("AutoNormalLoader", &AutoNormalLoader, config);
}

#[test]
fn granite_estimates_attention_mamba_and_moe_storage() {
    let mut config = serde_json::json!({
        "hidden_size": 8,
        "intermediate_size": 6,
        "shared_intermediate_size": 4,
        "vocab_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 0.00001,
        "max_position_embeddings": 128,
        "rope_scaling": null,
        "quantization_config": null,
        "layer_types": ["attention", "mamba"],
        "mamba_n_heads": 4,
        "mamba_n_groups": 1,
        "mamba_d_state": 2,
        "mamba_d_head": 4,
        "mamba_d_conv": 3,
        "mamba_expand": 2,
        "mamba_conv_bias": true,
        "mamba_proj_bias": true,
        "num_local_experts": 3
    });

    let sizes = GraniteMoeHybridLoader
        .layer_sizes_in_bytes(&config.to_string(), DType::F32, 2, None)
        .unwrap();
    assert_eq!(sizes, vec![2464, 4496]);

    config["shared_intermediate_size"] = serde_json::Value::Null;
    config["num_hidden_layers"] = serde_json::json!(1);
    config["layer_types"] = serde_json::json!(["attention"]);
    let pure_moe = GraniteMoeHybridLoader
        .layer_sizes_in_bytes(&config.to_string(), DType::F32, 2, None)
        .unwrap();
    assert_eq!(pure_moe, vec![2272]);

    config["num_local_experts"] = serde_json::json!(0);
    let pure_dense = GraniteMoeHybridLoader
        .layer_sizes_in_bytes(&config.to_string(), DType::F32, 2, None)
        .unwrap();
    assert_eq!(pure_dense, vec![736]);
}

#[test]
fn gpt_oss_estimates_split_and_mxfp4_experts() {
    let mut config = serde_json::json!({
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "sliding_window": 16,
        "head_dim": 4,
        "quantization_config": null,
        "num_local_experts": 3,
        "num_experts_per_tok": 2,
        "layer_types": ["full_attention"],
        "attention_bias": true,
        "rope_scaling": null
    });

    let split = GptOssLoader
        .layer_sizes_in_bytes(&config.to_string(), DType::F32, 2, None)
        .unwrap();
    assert_eq!(split, vec![1764]);

    config["quantization_config"] = serde_json::json!({"quant_method": "mxfp4"});
    let mxfp4 = GptOssLoader
        .layer_sizes_in_bytes(&config.to_string(), DType::F32, 2, None)
        .unwrap();
    assert_eq!(mxfp4, vec![1816]);
}

#[test]
fn every_architecture_round_trips_through_its_names() {
    use std::collections::HashSet;
    use strum::IntoEnumIterator;

    let (mut cli, mut hf) = (HashSet::new(), HashSet::new());
    for arch in NormalLoaderType::iter() {
        let name = arch.to_string();
        assert_eq!(
            name.parse::<NormalLoaderType>().unwrap(),
            arch,
            "cli name `{name}`"
        );
        assert_eq!(
            NormalLoaderType::from_causal_lm_name(arch.causal_lm_name()).unwrap(),
            arch
        );
        assert!(cli.insert(name), "duplicate cli name for {arch:?}");
        assert!(
            hf.insert(arch.causal_lm_name()),
            "duplicate HF class for {arch:?}"
        );
    }
    let err = "nope".parse::<NormalLoaderType>().unwrap_err();
    assert!(
        err.contains("`mistral`") && err.contains("`lfm2_moe`"),
        "{err}"
    );
}
