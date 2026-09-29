use anyhow::Result;
use regex::Regex;

// One shared builder, so the 50-odd ISQ loaders do not each expand a `Regex::new(..)?` error path per pattern.
pub fn isq_regexes<S: AsRef<str>>(patterns: &[S]) -> Result<Vec<Regex>> {
    patterns
        .iter()
        .map(|pattern| Ok(Regex::new(pattern.as_ref())?))
        .collect()
}

/// Trait for loading models with ISQ.
pub trait IsqModelLoader {
    /// Exact checkpoint tensor paths whose default ISQ type should be promoted.
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>>;

    /// Regex to match layers which will have standard *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }

    /// Regex to match layers which will have standard ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
}
