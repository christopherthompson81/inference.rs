//! A tiny random-weight Qwen3.8-Flash-Next (qwen4exp) checkpoint: hyper-connections, a sigmoid GDN gate and PLE.

use inference::{Model, ModelDType, PromptInput, PromptLogitsRequest, TextModelBuilder};
use inference_models_qwen::qwen3_5::{Qwen3_5TextModel, TextConfig};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
#[path = "../support/recording.rs"]
mod recording;
#[path = "../support/tiny_drafter.rs"]
mod tiny_drafter;
#[path = "../support/traces.rs"]
mod traces;

const NUM_LAYERS: usize = 4;
const HIDDEN: usize = 64;
const MOE_EXPERTS: usize = 4;
const MOE_INTERMEDIATE: usize = 32;
// the layout detection reads the fused experts' shapes before any tensor
const EXPERTS_PREFIX: &str = "language_model.model.layers";
const VOCAB: usize = 266;
const EOS: u32 = 2;
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
// an EOS inside the prompt, so the n-grams after it start fresh
const PROMPT: [u32; 18] = [
    11, 57, 203, 4, 98, 140, 33, 77, EOS, 19, 64, 222, 5, 180, 91, 42, 7, 130,
];
const GENERATED: usize = 12;
const STEP_LOGPROB_TOLERANCE: f32 = 1e-3;
const PREFIX_CACHE_SEQS: usize = 4;
// a few paged blocks, so the prefix snapshot lands inside the prompt
const LONG_PROMPT_LEN: usize = 100;
const LONG_PROMPT_STRIDE: usize = 7;
const CACHED_LOGPROB_TOLERANCE: f32 = 1e-3;
// how much more BF16 error a split model may show than the model all on the GPU
const SPLIT_ERROR_RATIO: f32 = 1.25;
// speculative verification against plain decode, both BF16
const SPECULATIVE_LOGPROB_TOLERANCE: f32 = 0.06;
const DFLASH_TAPS: [usize; 2] = [1, 3];
const DFLASH_BLOCK: usize = 4;
const DFLASH_N_PREDICT: usize = 2;
const MTP_N_PREDICT: usize = 2;
// every drafter layer windowed, which its decode graphs need
const DFLASH_WINDOW: usize = 32;
// above every test's context, so QSA keeps every key
const DENSE_BUDGET: usize = 1024;
// two blocks of 4: QSA selects from the 12th token on
const SPARSE_BUDGET: usize = 8;
// past several of QSA's 4-token blocks, so decode pools and selects at the sparse budget
const GRAPH_MAX_LEN: usize = 24;
// rows at different positions, one a single token (a one-token prompt is a decode-shaped step that is not one)
const GRAPH_PROMPT_LENS: [usize; 5] = [18, 6, 1, 30, 11];
// rounds of concurrent requests, so the decode batches fill different graph buckets
const GRAPH_BATCHES: [usize; 3] = [1, 3, 5];

fn tiny_flash_next(budget: usize) -> anyhow::Result<tempfile::TempDir> {
    tiny_flash_next_with(budget, false)
}

// With `mtp`, the checkpoint carries a built-in MTP head (one QSA + MoE block with its own mixer).
fn tiny_flash_next_with(budget: usize, mtp: bool) -> anyhow::Result<tempfile::TempDir> {
    let mut config = serde_json::json!({
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "hidden_act": "silu",
        "num_hidden_layers": NUM_LAYERS,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_experts": MOE_EXPERTS,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "shared_expert_intermediate_size": 32,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": true
        }
    });
    // apart from the base fields, since one json! this long passes the macro recursion limit
    let flash_next = serde_json::json!({
        "hc_count": 4,
        "hc_lowrank": 8,
        "output_gate_type": "sigmoid",
        "ple_layer_ids": [2],
        "ple_embed_dim": 64,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 1000,
        "make_ngram_vocab_size_divisible_by": 128,
        "split_ngram_parts": 1,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": budget,
        "indexer_compress_ratio": 4,
        "eos_token_id": EOS,
        "bos_token_id": 1,
        "tie_word_embeddings": false,
        "mtp_num_hidden_layers": usize::from(mtp)
    });
    let serde_json::Value::Object(flash_next) = flash_next else {
        unreachable!("a json! object literal")
    };
    config
        .as_object_mut()
        .expect("a json! object literal")
        .extend(flash_next);
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let files = [
        config_path,
        format!("{FIXTURES}/paddleocr_vl/tiny/tokenizer.json").into(),
        format!("{FIXTURES}/llama_tiny/tokenizer_config.json").into(),
    ];
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let cfg: TextConfig = serde_json::from_value(config)?;
    let shapes = (0..NUM_LAYERS)
        .map(|layer| format!("{EXPERTS_PREFIX}.{layer}.mlp.experts"))
        .chain(mtp.then(|| "mtp.layers.0.mlp.experts".to_string()))
        .flat_map(|p| {
            [
                (
                    format!("{p}.gate_up_proj"),
                    vec![MOE_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN],
                ),
                (
                    format!("{p}.down_proj"),
                    vec![MOE_EXPERTS, HIDDEN, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect();
    recording::record_checkpoint_seeded_with_shapes(
        &files,
        NUM_LAYERS,
        &[],
        shapes,
        |vb, metadata| {
            Qwen3_5TextModel::new(
                &cfg,
                vb,
                false,
                mtp,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )
}

// the CUDA recurrence kernels take f16/bf16
async fn paged_prefix_model(checkpoint: &tempfile::TempDir) -> anyhow::Result<Model> {
    Ok(TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS))
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?)
}

fn long_prompt() -> Vec<u32> {
    (0..LONG_PROMPT_LEN)
        .map(|i| (3 + i * LONG_PROMPT_STRIDE % (VOCAB - 3)) as u32)
        .collect()
}

// A greedy completion of `prompt` token ids: each generated token with its logprob, and the cached prompt tokens
async fn complete(
    model: &Model,
    prompt: &[u32],
    max_tokens: usize,
) -> anyhow::Result<(traces::Trace, usize)> {
    let request = serde_json::json!({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
        "logprobs": 1, "ignore_eos": true,
    });
    let response: serde_json::Value = serde_json::from_str(
        &model
            .completion_json(request.to_string().as_bytes())
            .await?,
    )?;
    let trace = response["choices"][0]["logprobs"]["content"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("no logprobs in {response}"))?
        .iter()
        .map(|step| {
            let top = &step["top_logprobs"][0];
            (
                top["token"].as_u64().unwrap_or_default() as u32,
                top["logprob"].as_f64().unwrap_or_default() as f32,
            )
        })
        .collect();
    let cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_default() as usize;
    Ok((trace, cached))
}

// Step-by-step decode carries PLE's token history and conv state, so it scores its own tokens as one prefill does.
#[tokio::test]
async fn flash_next_decode_matches_prefill_scoring() -> anyhow::Result<()> {
    let checkpoint = tiny_flash_next(DENSE_BUDGET)?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let (decoded, _) = complete(&model, &PROMPT, GENERATED).await?;
    let mut tokens = PROMPT.to_vec();
    tokens.extend(decoded.iter().map(|(token, _)| *token));
    let scored = model
        .prompt_logits(PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(tokens),
            output: Default::default(),
        })
        .await?;
    let prefilled = scored.token_logprobs[PROMPT.len()..]
        .iter()
        .map(|logprob| logprob.unwrap_or(f32::NAN))
        .collect::<Vec<_>>();
    for (step, ((_, decode), prefill)) in decoded.iter().zip(&prefilled).enumerate() {
        anyhow::ensure!(
            (decode - prefill).abs() < STEP_LOGPROB_TOLERANCE,
            "step {step}: decode logprob {decode}, prefill {prefill}"
        );
    }
    Ok(())
}

// Each prompt's trace, per round of concurrent requests (batches of 1, 3, then 5 rows at different positions).
async fn sparse_paged_rounds() -> anyhow::Result<decode_graphs::Rounds> {
    let checkpoint = tiny_flash_next(SPARSE_BUDGET)?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;
    let prompts = graph_prompts();
    let mut rounds = Vec::new();
    for batch in GRAPH_BATCHES {
        let traces = futures::future::try_join_all(
            prompts[..batch]
                .iter()
                .map(|prompt| complete(&model, prompt, GRAPH_MAX_LEN)),
        )
        .await?;
        rounds.push(traces.into_iter().map(|(trace, _)| trace).collect());
    }
    Ok(rounds)
}

// prompts of different lengths, each from a different offset of the long prompt
fn graph_prompts() -> Vec<Vec<u32>> {
    let long = long_prompt();
    GRAPH_PROMPT_LENS
        .iter()
        .enumerate()
        .map(|(i, &len)| long[i * LONG_PROMPT_STRIDE..i * LONG_PROMPT_STRIDE + len].to_vec())
        .collect()
}

// Each GRAPH_PROMPTS entry's greedy (token, logprob) per step, at the sparse budget.
const GRAPH_TRACES: [&[(u32, f32)]; GRAPH_PROMPT_LENS.len()] = [
    &[
        (60, -2.5534904),
        (70, -2.3135185),
        (42, -2.4721172),
        (228, -1.8240471),
        (189, -1.8332641),
        (239, -2.9075077),
        (41, -1.3090783),
        (3, -1.7174006),
        (244, -2.3841913),
        (25, -1.9992934),
        (246, -3.1183913),
        (122, -2.0252943),
        (193, -2.7879453),
        (2, -0.4540634),
        (136, -2.695309),
        (57, -1.7656752),
        (168, -0.66245633),
        (68, -2.108305),
        (128, -2.22468),
        (122, -1.3781013),
        (33, -2.8063502),
        (153, -1.9863819),
        (153, -1.7493929),
        (188, -1.812524),
    ],
    &[
        (40, -2.0535254),
        (81, -2.0802085),
        (264, -1.8805361),
        (147, -2.533749),
        (227, -0.272414),
        (82, -2.3933406),
        (117, -1.9762601),
        (264, -1.7551174),
        (264, -1.8713763),
        (189, -1.7934773),
        (100, -2.1364064),
        (122, -0.8744887),
        (209, -1.6428919),
        (76, -1.6746298),
        (70, -2.6219382),
        (23, -1.6738174),
        (188, -0.5108403),
        (254, -1.1494402),
        (204, -1.9505264),
        (103, -2.1379142),
        (210, -1.9221743),
        (154, -0.82791966),
        (254, -1.7986124),
        (188, -1.1445597),
    ],
    &[
        (142, -1.0860167),
        (130, -2.4738207),
        (128, -2.9085805),
        (9, -2.0025253),
        (186, -1.5429668),
        (103, -2.159467),
        (104, -2.6143),
        (104, -1.9917951),
        (250, -1.2910833),
        (12, -2.780106),
        (96, -2.0226889),
        (215, -1.0329583),
        (122, -2.149296),
        (12, -1.9449666),
        (107, -2.0361485),
        (214, -1.8467127),
        (9, -2.048369),
        (8, -2.4041233),
        (9, -1.7959902),
        (119, -2.3149245),
        (264, -1.1169198),
        (79, -1.9506046),
        (122, -1.5057116),
        (173, -2.4235408),
    ],
    &[
        (9, -2.3677666),
        (135, -0.96073306),
        (54, -2.8854082),
        (173, -1.7692586),
        (167, -0.64461845),
        (188, -1.841511),
        (76, -2.2547326),
        (120, -1.8371007),
        (115, -1.5641576),
        (12, -2.531162),
        (120, -1.3355253),
        (116, -1.7832023),
        (122, -1.8450652),
        (229, -1.9831185),
        (211, -1.4042349),
        (122, -1.1627425),
        (127, -1.5906926),
        (167, -2.3557105),
        (19, -1.8252149),
        (50, -2.500065),
        (12, -0.5008647),
        (64, -1.5714575),
        (70, -2.3207943),
        (67, -2.6226983),
    ],
    &[
        (168, -2.2089527),
        (185, -1.7278543),
        (172, -2.2864778),
        (173, -1.7341173),
        (93, -2.1984992),
        (231, -1.7880979),
        (211, -1.7926258),
        (242, -2.1178164),
        (168, -1.5733213),
        (54, -2.4383707),
        (167, -1.5423478),
        (167, -0.67659986),
        (175, -2.8122938),
        (96, -1.8376199),
        (167, -1.6300827),
        (258, -1.604981),
        (82, -2.4263382),
        (107, -1.4796622),
        (201, -1.1285124),
        (210, -1.1276088),
        (101, -2.2140083),
        (66, -1.2020565),
        (2, -0.69250077),
        (107, -1.6482325),
    ],
];

// Decode steps replay captured graphs (PLE's rows gathered by a host function, QSA's plan built on the device), in
// batches of rows at different positions, and decode as the eager steps do.
#[tokio::test]
async fn flash_next_decode_replays_cuda_graphs() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    let rounds = sparse_paged_rounds().await?;
    decode_graphs::assert_replayed(&decode_graphs::Counters::take(&snapshotter));
    decode_graphs::assert_traces_up_to_ties(&rounds, &GRAPH_TRACES);
    Ok(())
}

#[tokio::test]
async fn flash_next_decode_without_cuda_graphs_matches() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    decode_graphs::disable_graphs();
    let snapshotter = decode_graphs::recorder();
    let rounds = sparse_paged_rounds().await?;
    decode_graphs::assert_disabled(&decode_graphs::Counters::take(&snapshotter));
    decode_graphs::assert_traces_up_to_ties(&rounds, &GRAPH_TRACES);
    Ok(())
}

// Each position's logprob of the next token of `tokens`, as one prefill scores them.
async fn score(model: &Model, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
    let scored = model
        .prompt_logits(PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(tokens.to_vec()),
            output: Default::default(),
        })
        .await?;
    Ok(scored
        .token_logprobs
        .iter()
        .skip(1)
        .map(|logprob| logprob.unwrap_or(f32::NAN))
        .collect())
}

// Layers on the CPU beside the GPU run in F32 and hand BF16 back: a split model scores the F32 reference no worse
// than the model all on the GPU in BF16 does (this random checkpoint amplifies BF16 error at a few positions, so
// the comparison is of the mean).
#[tokio::test]
async fn flash_next_split_across_gpu_and_cpu_scores_as_on_the_gpu() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || inference_tensor::Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let checkpoint = tiny_flash_next(DENSE_BUDGET)?;
    let tokens = long_prompt();
    let build = |builder: TextModelBuilder| async move { builder.build().await };
    let path = checkpoint.path().to_string_lossy().to_string();
    let reference = build(
        TextModelBuilder::new(&path)
            .with_dtype(ModelDType::F32)
            .with_force_cpu(),
    )
    .await?;
    let expected = score(&reference, &tokens).await?;
    drop(reference);
    let mean_error = |scored: &[f32]| {
        expected
            .iter()
            .zip(scored)
            .map(|(e, a)| (e - a).abs())
            .sum::<f32>()
            / expected.len() as f32
    };
    let gpu = build(TextModelBuilder::new(&path).with_dtype(ModelDType::BF16)).await?;
    let on_gpu = mean_error(&score(&gpu, &tokens).await?);
    drop(gpu);
    let split = build(
        TextModelBuilder::new(&path)
            .with_dtype(ModelDType::BF16)
            .with_device_layers(vec![format!("0:{}", NUM_LAYERS / 2)]),
    )
    .await?;
    let scored = score(&split, &tokens).await?;
    let split_error = mean_error(&scored);
    anyhow::ensure!(
        scored.len() == expected.len() && split_error <= on_gpu * SPLIT_ERROR_RATIO,
        "the split model scores worse: mean logprob error {split_error} against {on_gpu} on the GPU"
    );
    Ok(())
}

// A DFlash drafter on Flash-Next drafts from the mean of the hyper-connection streams, and greedy output holds up to
// BF16 near ties (verification batches rows unlike plain decode): rejected drafts leave PLE's conv state, QSA's pooled
// keys and the indexer cache to be undone.
#[tokio::test]
async fn flash_next_dflash_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || inference_tensor::Device::new_cuda(0).is_err() {
        return Ok(());
    }
    for budget in [DENSE_BUDGET, SPARSE_BUDGET] {
        dflash_keeps_greedy_output(budget).await?;
    }
    Ok(())
}

async fn dflash_keeps_greedy_output(budget: usize) -> anyhow::Result<()> {
    let checkpoint = tiny_flash_next(budget)?;
    let plain = paged_prefix_model(&checkpoint).await?;
    let (expected, _) = complete(&plain, &PROMPT, GENERATED).await?;
    drop(plain);
    let drafter = tiny_drafter::tiny_drafter(
        &tiny_drafter::DrafterTarget {
            hidden: HIDDEN,
            vocab: VOCAB,
            taps: &DFLASH_TAPS,
        },
        serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "block_size": DFLASH_BLOCK,
            "layer_types": ["sliding_attention"],
            "sliding_window": DFLASH_WINDOW,
        }),
        None,
    )?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .with_mtp_model(drafter.path().to_string_lossy(), Some(DFLASH_N_PREDICT))
        .build()
        .await?;
    let (actual, _) = complete(&model, &PROMPT, GENERATED).await?;
    let drafts: usize = model
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    anyhow::ensure!(drafts > 0, "DFlash never drafted");
    same_up_to_ties(&actual, &expected, &format!("DFlash, budget {budget}"))
}

// A speculative trace matches plain decode's until both pick different tokens at near-equal logprobs (a BF16 tie
// that verification's batching breaks the other way).
fn same_up_to_ties(
    actual: &traces::Trace,
    expected: &traces::Trace,
    label: &str,
) -> anyhow::Result<()> {
    for (step, (got, want)) in actual.iter().zip(expected).enumerate() {
        anyhow::ensure!(
            (got.1 - want.1).abs() < SPECULATIVE_LOGPROB_TOLERANCE,
            "{label}, step {step}: speculative {actual:?} against plain {expected:?}"
        );
        if got.0 != want.0 {
            return Ok(());
        }
    }
    anyhow::ensure!(
        actual.len() == expected.len(),
        "{label}: decode ended early: {actual:?}"
    );
    Ok(())
}

// Flash-Next's built-in MTP head drafts from the target's pre-mixer streams through its own QSA + MoE block and
// mixer, and greedy output holds up to BF16 near ties.
#[tokio::test]
async fn flash_next_builtin_mtp_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || inference_tensor::Device::new_cuda(0).is_err() {
        return Ok(());
    }
    for budget in [DENSE_BUDGET, SPARSE_BUDGET] {
        let checkpoint = tiny_flash_next_with(budget, true)?;
        let plain = paged_prefix_model(&checkpoint).await?;
        let (expected, _) = complete(&plain, &PROMPT, GENERATED).await?;
        drop(plain);
        let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
            .with_builtin_mtp(Some(MTP_N_PREDICT))
            .build()
            .await?;
        let (actual, _) = complete(&model, &PROMPT, GENERATED).await?;
        let drafts: usize = model
            .speculative_stats()?
            .data
            .iter()
            .map(|m| m.drafts)
            .sum();
        anyhow::ensure!(drafts > 0, "MTP never drafted at budget {budget}");
        same_up_to_ties(&actual, &expected, &format!("MTP, budget {budget}"))?;
    }
    Ok(())
}

// A repeated prompt restores PLE's state with the recurrent snapshot and decodes as the cold run did.
#[tokio::test]
async fn flash_next_prefix_hit_restores_the_ple_state() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_flash_next(DENSE_BUDGET)?;
    let model = paged_prefix_model(&checkpoint).await?;
    let prompt = long_prompt();
    let (first, first_cached) = complete(&model, &prompt, GENERATED).await?;
    let (second, second_cached) = complete(&model, &prompt, GENERATED).await?;
    anyhow::ensure!(
        first_cached == 0 && second_cached > 0,
        "cached prompt tokens: {first_cached} then {second_cached}"
    );
    anyhow::ensure!(
        traces::close(&first, &second, CACHED_LOGPROB_TOLERANCE),
        "a prefix hit changed decoding: {first:?} vs {second:?}"
    );
    Ok(())
}

// Past QSA's budget a prefix hit restores the indexer's pooled keys with their pages and decodes as the cold run did,
// and the selection changes what the dense budget decodes.
#[tokio::test]
async fn flash_next_sparse_prefix_hit_restores_the_indexer_keys() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let prompt = long_prompt();
    let mut traces = Vec::new();
    for budget in [SPARSE_BUDGET, DENSE_BUDGET] {
        let checkpoint = tiny_flash_next(budget)?;
        let model = paged_prefix_model(&checkpoint).await?;
        let (first, first_cached) = complete(&model, &prompt, GENERATED).await?;
        let (second, second_cached) = complete(&model, &prompt, GENERATED).await?;
        anyhow::ensure!(
            first_cached == 0 && second_cached > 0,
            "cached prompt tokens: {first_cached} then {second_cached}"
        );
        anyhow::ensure!(
            traces::close(&first, &second, CACHED_LOGPROB_TOLERANCE),
            "a prefix hit changed decoding at budget {budget}: {first:?} vs {second:?}"
        );
        traces.push(first);
    }
    anyhow::ensure!(
        !traces::close(&traces[0], &traces[1], CACHED_LOGPROB_TOLERANCE),
        "QSA's selection left decoding as the dense budget's"
    );
    Ok(())
}

// A UQFF written from an ISQ load reloads every Flash-Next projection: hyper-connections, PLE, the indexer.
#[tokio::test]
async fn flash_next_isq_load_and_a_reload_of_its_uqff_decode_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_flash_next(DENSE_BUDGET)?;
    let uqff = tempfile::tempdir()?;
    let quantized = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_isq(inference::IsqType::Q8_0)
        .write_uqff(uqff.path().join("model.uqff"))
        .build()
        .await?;
    let (expected, _) = complete(&quantized, &PROMPT, GENERATED).await?;
    drop(quantized);
    let written = traces::uqff_files(uqff.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = inference::UqffTextModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        vec![written[0].clone()],
    )
    .into_inner()
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    let (actual, _) = complete(&reloaded, &PROMPT, GENERATED).await?;
    anyhow::ensure!(
        traces::close(&expected, &actual, STEP_LOGPROB_TOLERANCE),
        "the UQFF reload decodes differently: {expected:?} vs {actual:?}"
    );
    Ok(())
}
