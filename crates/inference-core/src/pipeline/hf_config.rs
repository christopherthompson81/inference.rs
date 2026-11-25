use std::str::FromStr;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct HfConfigOverrides(Value);

inference_json::via_serde!(HfConfigOverrides);

impl<'de> Deserialize<'de> for HfConfigOverrides {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

impl HfConfigOverrides {
    pub fn new(value: Value) -> Result<Self> {
        ensure!(
            value.is_object(),
            "HF config overrides must be a JSON object"
        );
        Ok(Self(value))
    }

    pub fn apply(&self, config: &str) -> Result<String> {
        let mut config: Value =
            serde_json::from_str(config).context("failed to parse model config.json")?;
        ensure!(
            config.is_object(),
            "model config.json must be a JSON object"
        );
        merge_json(&mut config, &self.0);
        serde_json::to_string(&config).context("failed to serialize overridden model config")
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    /// `overrides` with a looped model's (Inkling 2's) load options merged in: LoopCD-Logits and the passes to run,
    /// as top-level keys the loader lays over a nested `text_config`.
    pub fn with_loop(overrides: Option<Self>, logits: bool, passes: Option<u64>) -> Option<Self> {
        let mut loop_options = serde_json::Map::new();
        if logits {
            loop_options.insert("loop_guidance_logits".into(), Value::Bool(true));
        }
        if let Some(passes) = passes {
            loop_options.insert(
                "loop_config".into(),
                serde_json::json!({ "run_passes": passes }),
            );
        }
        if loop_options.is_empty() {
            return overrides;
        }
        let mut value = overrides.map_or_else(|| Value::Object(Default::default()), |o| o.0);
        merge_json(&mut value, &Value::Object(loop_options));
        Some(Self(value))
    }
}

impl FromStr for HfConfigOverrides {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let value = serde_json::from_str(value).context("invalid JSON for --hf-overrides")?;
        Self::new(value)
    }
}

fn merge_json(target: &mut Value, overrides: &Value) {
    match (target, overrides) {
        (Value::Object(target), Value::Object(overrides)) => {
            for (key, value) in overrides {
                match target.get_mut(key) {
                    Some(target) => merge_json(target, value),
                    None => {
                        target.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (target, value) => *target = value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_options_merge_into_other_overrides() -> Result<()> {
        let overrides: HfConfigOverrides =
            r#"{"loop_config": {"core_start": 6}, "rope_theta": 1}"#.parse()?;
        assert_eq!(
            HfConfigOverrides::with_loop(Some(overrides.clone()), false, None),
            Some(overrides.clone())
        );
        assert_eq!(HfConfigOverrides::with_loop(None, false, None), None);
        let merged = HfConfigOverrides::with_loop(Some(overrides), true, Some(2)).expect("options");
        assert_eq!(
            merged.as_value(),
            &serde_json::json!({"loop_config": {"core_start": 6, "run_passes": 2}, "rope_theta": 1,
                "loop_guidance_logits": true})
        );
        Ok(())
    }

    #[test]
    fn recursively_merges_nested_objects() -> Result<()> {
        let overrides: HfConfigOverrides = r#"{
            "text_config": {
                "max_position_embeddings": 131072,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 4.0
                }
            },
            "architectures": ["OverriddenArchitecture"]
        }"#
        .parse()?;
        let merged = overrides.apply(
            r#"{
                "text_config": {
                    "hidden_size": 4096,
                    "max_position_embeddings": 32768,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 1000000
                    }
                },
                "architectures": ["OriginalArchitecture"]
            }"#,
        )?;
        let merged: Value = serde_json::from_str(&merged)?;

        assert_eq!(merged["text_config"]["hidden_size"], 4096);
        assert_eq!(merged["text_config"]["max_position_embeddings"], 131072);
        assert_eq!(
            merged["text_config"]["rope_parameters"]["rope_type"],
            "yarn"
        );
        assert_eq!(
            merged["text_config"]["rope_parameters"]["rope_theta"],
            1000000
        );
        assert_eq!(merged["text_config"]["rope_parameters"]["factor"], 4.0);
        assert_eq!(merged["architectures"][0], "OverriddenArchitecture");
        Ok(())
    }

    #[test]
    fn rejects_non_object_overrides() {
        assert!(HfConfigOverrides::from_str("[]").is_err());
        assert!(HfConfigOverrides::from_str("null").is_err());
    }
}
