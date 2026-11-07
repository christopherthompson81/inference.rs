use std::collections::HashMap;

use anyhow::Result;
use inference_tensor::quantized::gguf_file::Value;
use tracing::info;

use crate::metadata::ContentMetadata;

use super::Content;

// llama.cpp's built-in templates a GGUF may name in place of carrying one, as its converter writes them
// (models/templates/llama-cpp-*.jinja, llama.cpp 970b51ce7, MIT)
const NAMED_TEMPLATES: &[(&str, &str)] =
    &[("rwkv-world", include_str!("../templates/rwkv-world.jinja"))];

struct PropsGGUFTemplate {
    chat_template: Option<String>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUFTemplate {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> Result<Self, Self::Error> {
        // No required keys

        let props = Self {
            chat_template: c.get_option_value("chat_template")?,
        };

        Ok(props)
    }
}

// Get chat template from GGUF metadata if it exists
pub fn get_gguf_chat_template<R: std::io::Seek + std::io::Read>(
    content: &Content<'_, R>,
) -> Result<Option<String>> {
    get_gguf_chat_template_from_metadata(content.get_metadata())
}

pub fn get_gguf_chat_template_from_metadata(
    metadata: &HashMap<String, Value>,
) -> Result<Option<String>> {
    let metadata = ContentMetadata {
        path_prefix: "tokenizer",
        metadata,
    };
    let mut props = PropsGGUFTemplate::try_from(metadata)?;
    if let Some((_, template)) = props
        .chat_template
        .as_deref()
        .and_then(|name| NAMED_TEMPLATES.iter().find(|(known, _)| *known == name))
    {
        props.chat_template = Some(template.to_string());
    }
    if let Some(ref chat_template) = props.chat_template {
        info!(
            "Discovered and using GGUF chat template: `{}`",
            chat_template.replace('\n', "\\n")
        );
    }
    Ok(props.chat_template)
}

#[cfg(test)]
mod tests {
    use super::*;

    // a GGUF naming llama.cpp's RWKV world template gets the template, not its name as the prompt
    #[test]
    fn a_named_llama_cpp_template_reads_as_its_jinja() -> Result<()> {
        let metadata = HashMap::from([(
            "tokenizer.chat_template".to_string(),
            Value::String("rwkv-world".to_string()),
        )]);
        let template = get_gguf_chat_template_from_metadata(&metadata)?.expect("a template");
        assert!(template.contains("'User: '") && template.contains("'Assistant:'"));
        Ok(())
    }
}
