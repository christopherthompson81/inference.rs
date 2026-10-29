use anyhow::{Result, bail};
use inference_quant::{GgufArchive, GgufBindingMap};

use inference_nn::loaders::MultimodalLoaderType;

#[cfg(any(test, feature = "models-gemma"))]
use super::{gemma3_bindings::build_gemma3_bindings, gemma3n_bindings::build_gemma3n_bindings};
use super::{
    granite_multimodal_bindings::{
        self, build_granite_speech_bindings, build_granite4_vision_bindings,
    },
    idefics3_bindings::build_idefics3_bindings,
    lfm2_vl_bindings::build_lfm2_vl_bindings,
    llama4_bindings::build_llama4_bindings,
    mistral3_bindings::build_mistral3_bindings,
    multimodal_binding_utils::{metadata_string, projector_type},
    muse_glimmer_bindings::build_muse_glimmer_bindings,
    normal_registry::RopePairing,
    paddleocr_vl_bindings::{self, build_paddleocr_vl_bindings},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeMultimodalGgufFamily {
    Gemma3,
    Gemma3n,
    Granite4Vision,
    GraniteSpeech,
    Idefics3,
    Mistral3,
    Llama4,
    Lfm2Vl,
    MuseGlimmer,
    PaddleOcrVl,
}

pub struct NativeMultimodalGguf {
    pub loader_type: MultimodalLoaderType,
    pub bindings: GgufBindingMap,
    pub rope_pairing: RopePairing,
}

pub fn resolve_native_multimodal_gguf(
    archive: &GgufArchive,
) -> Result<Option<NativeMultimodalGguf>> {
    let architecture = metadata_string(archive, "general.architecture")?
        .ok_or_else(|| anyhow::anyhow!("GGUF metadata `general.architecture` is required"))?;
    let Some(family) = family_from_names(architecture, projector_type(archive)?)? else {
        return Ok(None);
    };
    Ok(Some(NativeMultimodalGguf {
        loader_type: family.loader_type(),
        bindings: family.build_bindings(archive)?,
        rope_pairing: family.rope_pairing(),
    }))
}

impl NativeMultimodalGgufFamily {
    fn loader_type(self) -> MultimodalLoaderType {
        match self {
            Self::Gemma3 => MultimodalLoaderType::Gemma3,
            Self::Gemma3n => MultimodalLoaderType::Gemma3n,
            Self::Granite4Vision => MultimodalLoaderType::Granite4Vision,
            Self::GraniteSpeech => MultimodalLoaderType::GraniteSpeech,
            Self::Idefics3 => MultimodalLoaderType::Idefics3,
            Self::Mistral3 => MultimodalLoaderType::Mistral3,
            Self::Llama4 => MultimodalLoaderType::Llama4,
            Self::Lfm2Vl => MultimodalLoaderType::Lfm2Vl,
            Self::MuseGlimmer => MultimodalLoaderType::MuseGlimmer,
            Self::PaddleOcrVl => MultimodalLoaderType::PaddleOcrVl,
        }
    }

    fn rope_pairing(self) -> RopePairing {
        match self {
            // llama.cpp's Ernie 4.5 converter keeps q/k in Hugging Face's half-split order
            Self::Gemma3 | Self::Gemma3n | Self::Lfm2Vl | Self::PaddleOcrVl => {
                RopePairing::HalfSplit
            }
            Self::Granite4Vision
            | Self::GraniteSpeech
            | Self::Idefics3
            | Self::Mistral3
            | Self::Llama4
            | Self::MuseGlimmer => RopePairing::Adjacent,
        }
    }

    fn build_bindings(self, archive: &GgufArchive) -> Result<GgufBindingMap> {
        match self {
            #[cfg(any(test, feature = "models-gemma"))]
            Self::Gemma3 => build_gemma3_bindings(archive),
            #[cfg(any(test, feature = "models-gemma"))]
            Self::Gemma3n => build_gemma3n_bindings(archive),
            #[cfg(not(any(test, feature = "models-gemma")))]
            Self::Gemma3 | Self::Gemma3n => {
                bail!("GGUF Gemma 3 models are not built in; enable the `models-gemma` feature")
            }
            Self::Granite4Vision => build_granite4_vision_bindings(archive),
            Self::GraniteSpeech => build_granite_speech_bindings(archive),
            Self::Idefics3 => build_idefics3_bindings(archive),
            Self::Mistral3 => build_mistral3_bindings(archive),
            Self::Llama4 => build_llama4_bindings(archive),
            Self::Lfm2Vl => build_lfm2_vl_bindings(archive),
            Self::MuseGlimmer => build_muse_glimmer_bindings(archive),
            Self::PaddleOcrVl => build_paddleocr_vl_bindings(archive),
        }
    }
}

// (vision projector type, the text architecture it pairs with, family)
const FAMILY_NAMES: [(&str, &str, NativeMultimodalGgufFamily); 10] = [
    ("gemma3", "gemma3", NativeMultimodalGgufFamily::Gemma3),
    ("gemma3nv", "gemma3n", NativeMultimodalGgufFamily::Gemma3n),
    (
        granite_multimodal_bindings::VISION_PROJECTOR,
        granite_multimodal_bindings::TEXT_ARCHITECTURE,
        NativeMultimodalGgufFamily::Granite4Vision,
    ),
    (
        granite_multimodal_bindings::SPEECH_PROJECTOR,
        granite_multimodal_bindings::TEXT_ARCHITECTURE,
        NativeMultimodalGgufFamily::GraniteSpeech,
    ),
    ("idefics3", "llama", NativeMultimodalGgufFamily::Idefics3),
    ("pixtral", "mistral3", NativeMultimodalGgufFamily::Mistral3),
    ("llama4", "llama4", NativeMultimodalGgufFamily::Llama4),
    ("lfm2", "lfm2", NativeMultimodalGgufFamily::Lfm2Vl),
    (
        "muse-glimmer",
        "muse-glimmer",
        NativeMultimodalGgufFamily::MuseGlimmer,
    ),
    (
        paddleocr_vl_bindings::PROJECTOR,
        paddleocr_vl_bindings::ARCHITECTURE,
        NativeMultimodalGgufFamily::PaddleOcrVl,
    ),
];

fn family_from_names(
    architecture: &str,
    projector: Option<&str>,
) -> Result<Option<NativeMultimodalGgufFamily>> {
    let Some((projector, expected, family)) =
        projector.and_then(|projector| FAMILY_NAMES.iter().find(|(name, ..)| *name == projector))
    else {
        return Ok(None);
    };
    require_architecture(architecture, expected, projector)?;
    Ok(Some(*family))
}

/// Whether `architecture` is the text side of a native multimodal GGUF family (loaded with its projector).
pub fn is_multimodal_text_architecture(architecture: &str) -> bool {
    FAMILY_NAMES
        .iter()
        .any(|(_, expected, _)| *expected == architecture)
}

fn require_architecture(architecture: &str, expected: &str, projector: &str) -> Result<()> {
    if architecture != expected {
        bail!(
            "vision projector `{projector}` requires `{expected}` GGUF architecture, \
             found `{architecture}`"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_converter_pairs() {
        for (architecture, projector, family, pairing) in [
            (
                "gemma3",
                "gemma3",
                NativeMultimodalGgufFamily::Gemma3,
                RopePairing::HalfSplit,
            ),
            (
                "gemma3n",
                "gemma3nv",
                NativeMultimodalGgufFamily::Gemma3n,
                RopePairing::HalfSplit,
            ),
            (
                "llama",
                "idefics3",
                NativeMultimodalGgufFamily::Idefics3,
                RopePairing::Adjacent,
            ),
            (
                "mistral3",
                "pixtral",
                NativeMultimodalGgufFamily::Mistral3,
                RopePairing::Adjacent,
            ),
            (
                "llama4",
                "llama4",
                NativeMultimodalGgufFamily::Llama4,
                RopePairing::Adjacent,
            ),
            (
                "lfm2",
                "lfm2",
                NativeMultimodalGgufFamily::Lfm2Vl,
                RopePairing::HalfSplit,
            ),
            (
                "muse-glimmer",
                "muse-glimmer",
                NativeMultimodalGgufFamily::MuseGlimmer,
                RopePairing::Adjacent,
            ),
            (
                "paddleocr",
                "paddleocr",
                NativeMultimodalGgufFamily::PaddleOcrVl,
                RopePairing::HalfSplit,
            ),
        ] {
            let detected = family_from_names(architecture, Some(projector))
                .unwrap()
                .unwrap();
            assert_eq!(detected, family);
            assert_eq!(detected.rope_pairing(), pairing);
        }
    }

    #[test]
    fn rejects_cross_family_projector_pair() {
        let error = family_from_names("llama", Some("pixtral")).unwrap_err();
        assert!(error.to_string().contains("requires `mistral3`"));
    }

    #[test]
    fn leaves_other_projectors_for_other_registries() {
        assert_eq!(
            family_from_names("qwen3vl", Some("qwen3vl_merger")).unwrap(),
            None
        );
        assert_eq!(family_from_names("gemma4", Some("gemma4v")).unwrap(), None);
    }

    #[test]
    fn excludes_non_intersecting_converter_formats() {
        for (architecture, projector) in [
            ("qwen2", "resampler"),
            ("qwen3", "minicpmv4_6"),
            ("phi3", "phi3v"),
            ("llama", "idefics2"),
        ] {
            assert_eq!(
                family_from_names(architecture, Some(projector)).unwrap(),
                None
            );
        }
    }
}
