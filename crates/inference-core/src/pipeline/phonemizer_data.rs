use std::{
    collections::HashSet,
    path::Path,
    sync::{Arc, OnceLock},
};

use hf_hub::{Repo, RepoType, api::sync::ApiRepo};
use inference_models_speech::kokoro::g2p::{
    DATA_REPO, DATA_REVISION, DataError, DataSource, resolve_data_root, set_data_source,
};

use crate::pipeline::{TokenSource, hf};

/// The phonemizer's data read file by file from its Hugging Face repo, so a language downloads on first use.
struct HubData {
    api: ApiRepo,
    repo: String,
    revision: String,
    // answers the optional files the phonemizer probes for without a 404 each; only a successful listing is kept
    files: OnceLock<HashSet<String>>,
}

impl HubData {
    fn new(api: ApiRepo, repo: &str, revision: &str) -> Self {
        Self {
            api,
            repo: repo.to_string(),
            revision: revision.to_string(),
            files: OnceLock::new(),
        }
    }

    fn missing(&self, key: &str, detail: String) -> DataError {
        DataError::Missing {
            key: key.to_string(),
            detail: format!("{}@{}: {detail}", self.repo, self.revision),
        }
    }
}

impl DataSource for HubData {
    fn read(&self, key: &str) -> Result<Vec<u8>, DataError> {
        let id = Path::new(&self.repo);
        let files = match self.files.get() {
            Some(files) => files,
            None => {
                let listed = hf::list_repo_files(&self.api, id, true, &self.revision)
                    .map_err(|e| self.missing(key, e.to_string()))?;
                self.files.get_or_init(|| HashSet::from_iter(listed))
            }
        };
        if !files.contains(key) {
            return Err(self.missing(key, "not in the repo".into()));
        }
        hf::get_file(&self.api, id, key, &self.revision)
            .and_then(|path| Ok(std::fs::read(path)?))
            .map_err(|e| self.missing(key, e.to_string()))
    }
}

/// Points the phonemizer at its Hugging Face data unless `VERNACULA_DATA_DIR` (or a source checkout) supplies it.
pub(crate) fn install(token_source: &TokenSource, progress: bool) -> anyhow::Result<()> {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.get().is_some() || resolve_data_root().is_some() {
        return Ok(());
    }
    let api = hf::build_api(token_source, progress)?.repo(Repo::with_revision(
        DATA_REPO.to_string(),
        RepoType::Model,
        DATA_REVISION.to_string(),
    ));
    set_data_source(Some(Arc::new(HubData::new(api, DATA_REPO, DATA_REVISION))));
    let _ = INSTALLED.set(());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use hf_hub::Cache;

    use super::*;

    const REPO: &str = "example/phonemizer-data";
    const TAG: &str = "0a1b2c3d";
    const COMMIT: &str = "9f8e7d6c5b4a39281706f5e4d3c2b1a098765432";

    // An offline cache holding one tagged snapshot answers present keys with their bytes and absent ones as missing
    #[test]
    fn reads_keys_from_a_cached_snapshot() -> anyhow::Result<()> {
        let cache = tempfile::tempdir()?;
        let folder = cache.path().join("models--example--phonemizer-data");
        let snapshot = folder.join("snapshots").join(COMMIT);
        fs::create_dir_all(snapshot.join("languages/english"))?;
        fs::create_dir_all(folder.join("refs"))?;
        fs::write(folder.join("refs").join(TAG), COMMIT)?;
        fs::write(snapshot.join("languages/english/table.tsv"), "a\tb\n")?;
        // SAFETY: nextest runs each test in its own process
        unsafe {
            std::env::set_var(hf::HF_HUB_OFFLINE_ENV, "1");
            std::env::set_var("HF_HUB_CACHE", cache.path());
        }
        let api = hf_hub::api::sync::ApiBuilder::from_cache(Cache::new(cache.path().into()))
            .build()?
            .repo(Repo::with_revision(
                REPO.into(),
                RepoType::Model,
                TAG.into(),
            ));
        let data = HubData::new(api, REPO, TAG);
        assert_eq!(data.read("languages/english/table.tsv")?, b"a\tb\n");
        assert!(matches!(
            data.read("languages/english/absent.tsv"),
            Err(DataError::Missing { .. })
        ));
        Ok(())
    }
}
