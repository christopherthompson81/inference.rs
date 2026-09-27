//! Uploaded agent skills: storage, versions and the objects both skill APIs return.

use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Read},
    path::{Component, Path, PathBuf},
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use inference_core::ShellSkillMount;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

use crate::{
    api_error::{ApiError, ApiErrorKind},
    openai::OpenAiShellSkillReference,
};

#[doc(hidden)]
pub const SKILL_OBJECT: &str = "skill";
const SKILL_VERSION_OBJECT: &str = "skill.version";
const ANTHROPIC_SKILL_VERSION_OBJECT: &str = "skill_version";
#[doc(hidden)]
pub const CUSTOM_SKILL_SOURCE: &str = "custom";
#[doc(hidden)]
pub const ANTHROPIC_SKILL_SOURCE: &str = "anthropic";
const SKILL_METADATA_FILE: &str = "skill.json";
const SKILL_CONTENT_DIR: &str = "content";
const MAX_SKILL_UPLOAD_BYTES: usize = 50 * 1024 * 1024;
const MAX_SKILL_FILES: usize = 500;

#[derive(Clone)]
pub struct SkillStore {
    root: PathBuf,
    skills: Arc<RwLock<HashMap<String, SkillMetadata>>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SkillMetadata {
    id: String,
    name: String,
    description: String,
    created_at: u64,
    versions: Vec<SkillVersionMetadata>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SkillVersionMetadata {
    version: u64,
    created_at: u64,
    source_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct SkillObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub name: String,
    pub description: String,
    pub latest_version: u64,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct SkillVersionObject {
    pub id: String,
    pub object: &'static str,
    pub skill_id: String,
    pub created_at: u64,
    pub version: u64,
    pub name: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct SkillListObject {
    pub object: &'static str,
    pub data: Vec<SkillObject>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct SkillListQuery {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub page: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillObject {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: &'static str,
    pub created_at: String,
    pub updated_at: String,
    pub display_title: String,
    pub latest_version: String,
    pub source: &'static str,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillListObject {
    pub data: Vec<AnthropicSkillObject>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillVersionObject {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: &'static str,
    pub skill_id: String,
    pub created_at: String,
    pub version: String,
    pub name: String,
    pub description: String,
    pub directory: String,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillVersionListObject {
    pub data: Vec<AnthropicSkillVersionObject>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

struct SkillUpload {
    name: String,
    description: String,
    source_path: PathBuf,
    staging_path: PathBuf,
}

#[doc(hidden)]
pub fn invalid_skill_upload(message: impl Into<String>) -> anyhow::Error {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        message,
        Some("invalid_skill_upload"),
        Some("files"),
    )
    .into()
}

#[doc(hidden)]
pub fn skill_upload_too_large(message: impl Into<String>) -> anyhow::Error {
    ApiError::new(
        ApiErrorKind::PayloadTooLarge,
        message,
        Some("request_body_too_large"),
        Some("files"),
    )
    .into()
}

fn skill_not_found(message: impl Into<String>) -> anyhow::Error {
    ApiError::new(
        ApiErrorKind::NotFound,
        message,
        Some("skill_not_found"),
        Some("skill_id"),
    )
    .into()
}

fn invalid_skill_reference(message: impl Into<String>) -> anyhow::Error {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        message,
        Some("invalid_skill_reference"),
        Some("tools"),
    )
    .into()
}

impl SkillStore {
    pub fn default_root() -> PathBuf {
        std::env::temp_dir().join("inference-skills")
    }

    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        let store = Self {
            root,
            skills: Arc::new(RwLock::new(HashMap::new())),
        };
        store.load()?;
        Ok(store)
    }

    pub fn list(&self) -> Result<Vec<SkillObject>> {
        let skills = self
            .skills
            .read()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        Ok(skills.values().map(SkillObject::from).collect())
    }

    pub fn list_versions(&self, skill_id: &str) -> Result<Vec<SkillVersionObject>> {
        let skills = self
            .skills
            .read()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        let metadata = skills
            .get(skill_id)
            .ok_or_else(|| skill_not_found(format!("Skill `{skill_id}` was not found.")))?;
        metadata
            .versions
            .iter()
            .map(|version| SkillVersionObject::from_metadata(metadata, version.version))
            .collect()
    }

    pub fn create_skill(&self, files: SkillFiles) -> Result<SkillObject> {
        let upload = stage_upload(files)?;
        let id = format!("skill_{}", uuid::Uuid::new_v4().simple());
        let created_at = unix_now();
        let source_path = self.store_version_content(&id, 1, &upload.source_path)?;
        let _ = fs::remove_dir_all(&upload.staging_path);
        let metadata = SkillMetadata {
            id: id.clone(),
            name: upload.name,
            description: upload.description,
            created_at,
            versions: vec![SkillVersionMetadata {
                version: 1,
                created_at,
                source_path,
            }],
        };
        self.persist_metadata(&metadata)?;
        self.skills
            .write()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?
            .insert(id, metadata.clone());
        Ok(SkillObject::from(&metadata))
    }

    pub fn create_version(&self, skill_id: &str, files: SkillFiles) -> Result<SkillVersionObject> {
        let upload = stage_upload(files)?;
        let mut skills = self
            .skills
            .write()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        let metadata = skills
            .get_mut(skill_id)
            .ok_or_else(|| skill_not_found(format!("Skill `{skill_id}` was not found.")))?;
        let version = metadata.versions.last().map(|v| v.version + 1).unwrap_or(1);
        let created_at = unix_now();
        let source_path = self.store_version_content(skill_id, version, &upload.source_path)?;
        let _ = fs::remove_dir_all(&upload.staging_path);
        metadata.name = upload.name;
        metadata.description = upload.description;
        metadata.versions.push(SkillVersionMetadata {
            version,
            created_at,
            source_path,
        });
        self.persist_metadata(metadata)?;
        SkillVersionObject::from_metadata(metadata, version)
    }

    pub fn resolve_references(
        &self,
        refs: &[OpenAiShellSkillReference],
    ) -> Result<inference_core::ShellOptions> {
        let mut skills = Vec::new();
        for reference in refs {
            skills.push(self.resolve_reference(reference)?);
        }
        Ok(inference_core::ShellOptions { skills })
    }

    fn resolve_reference(&self, reference: &OpenAiShellSkillReference) -> Result<ShellSkillMount> {
        let skills = self
            .skills
            .read()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        let metadata = skills.get(&reference.skill_id).ok_or_else(|| {
            skill_not_found(format!("Skill `{}` was not found.", reference.skill_id))
        })?;
        let version = match &reference.version {
            None => metadata.versions.last(),
            Some(Value::String(s)) if s == "latest" => metadata.versions.last(),
            Some(Value::String(s)) => {
                let parsed = s.parse::<u64>().map_err(|_| {
                    invalid_skill_reference(format!("Invalid skill version `{s}`."))
                })?;
                metadata
                    .versions
                    .iter()
                    .find(|version| version.version == parsed)
            }
            Some(Value::Number(n)) => {
                let parsed = n.as_u64().ok_or_else(|| {
                    invalid_skill_reference(format!("Invalid skill version `{n}`."))
                })?;
                metadata
                    .versions
                    .iter()
                    .find(|version| version.version == parsed)
            }
            Some(other) => {
                return Err(invalid_skill_reference(format!(
                    "Unsupported skill version value `{other}`."
                )))
            }
        }
        .ok_or_else(|| {
            skill_not_found(format!(
                "Skill `{}` version was not found.",
                reference.skill_id
            ))
        })?;

        Ok(ShellSkillMount {
            name: metadata.name.clone(),
            description: metadata.description.clone(),
            source_path: version.source_path.clone(),
        })
    }

    fn store_version_content(
        &self,
        skill_id: &str,
        version: u64,
        source: &Path,
    ) -> Result<PathBuf> {
        let version_root = self
            .root
            .join(skill_id)
            .join("versions")
            .join(version.to_string());
        let content_dir = version_root.join(SKILL_CONTENT_DIR);
        if content_dir.exists() {
            fs::remove_dir_all(&content_dir)?;
        }
        copy_dir_all(source, &content_dir)?;
        Ok(content_dir)
    }

    fn persist_metadata(&self, metadata: &SkillMetadata) -> Result<()> {
        let skill_dir = self.root.join(&metadata.id);
        fs::create_dir_all(&skill_dir)?;
        let bytes = serde_json::to_vec_pretty(metadata)?;
        fs::write(skill_dir.join(SKILL_METADATA_FILE), bytes)?;
        Ok(())
    }

    fn load(&self) -> Result<()> {
        let mut loaded = HashMap::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let metadata_path = entry.path().join(SKILL_METADATA_FILE);
            if !metadata_path.exists() {
                continue;
            }
            let bytes = fs::read(&metadata_path)?;
            let metadata: SkillMetadata = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", metadata_path.display()))?;
            loaded.insert(metadata.id.clone(), metadata);
        }
        *self
            .skills
            .write()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))? = loaded;
        Ok(())
    }
}

impl From<&SkillMetadata> for SkillObject {
    fn from(value: &SkillMetadata) -> Self {
        Self {
            id: value.id.clone(),
            object: SKILL_OBJECT,
            created_at: value.created_at,
            name: value.name.clone(),
            description: value.description.clone(),
            latest_version: value.versions.last().map(|v| v.version).unwrap_or(0),
        }
    }
}

impl From<&SkillObject> for AnthropicSkillObject {
    fn from(value: &SkillObject) -> Self {
        let created_at = unix_to_rfc3339(value.created_at);
        Self {
            id: value.id.clone(),
            tp: SKILL_OBJECT,
            created_at: created_at.clone(),
            updated_at: created_at,
            display_title: value.name.clone(),
            latest_version: value.latest_version.to_string(),
            source: CUSTOM_SKILL_SOURCE,
        }
    }
}

impl SkillVersionObject {
    fn from_metadata(metadata: &SkillMetadata, version: u64) -> Result<Self> {
        let version_metadata = metadata
            .versions
            .iter()
            .find(|v| v.version == version)
            .ok_or_else(|| {
                anyhow::anyhow!("Skill `{}` version `{version}` missing", metadata.id)
            })?;
        Ok(Self {
            id: format!("{}_v{}", metadata.id, version),
            object: SKILL_VERSION_OBJECT,
            skill_id: metadata.id.clone(),
            created_at: version_metadata.created_at,
            version,
            name: metadata.name.clone(),
            description: metadata.description.clone(),
        })
    }
}

impl From<&SkillVersionObject> for AnthropicSkillVersionObject {
    fn from(value: &SkillVersionObject) -> Self {
        Self {
            id: value.id.clone(),
            tp: ANTHROPIC_SKILL_VERSION_OBJECT,
            skill_id: value.skill_id.clone(),
            created_at: unix_to_rfc3339(value.created_at),
            version: value.version.to_string(),
            name: value.name.clone(),
            description: value.description.clone(),
            directory: value.name.clone(),
        }
    }
}

/// The files of one skill upload, checked against the size and count limits as they are added.
#[derive(Default)]
pub struct SkillFiles {
    files: Vec<(String, Vec<u8>)>,
    total_bytes: usize,
}

impl SkillFiles {
    pub fn push(&mut self, file_name: String, bytes: Vec<u8>) -> Result<()> {
        self.total_bytes = self.total_bytes.saturating_add(bytes.len());
        if self.total_bytes > MAX_SKILL_UPLOAD_BYTES {
            return Err(skill_upload_too_large(format!(
                "Skill upload exceeds the {MAX_SKILL_UPLOAD_BYTES} byte limit."
            )));
        }
        self.files.push((file_name, bytes));
        if self.files.len() > MAX_SKILL_FILES {
            return Err(invalid_skill_upload(format!(
                "Skill upload may contain at most {MAX_SKILL_FILES} files."
            )));
        }
        Ok(())
    }
}

fn stage_upload(upload: SkillFiles) -> Result<SkillUpload> {
    let staging = tempfile::Builder::new()
        .prefix("inference-skill-upload-")
        .tempdir()?;
    let files = upload.files;

    if files.is_empty() {
        return Err(invalid_skill_upload(
            "Skill upload requires multipart file field `files`.",
        ));
    }

    if files.len() == 1 && looks_like_zip(&files[0].0, &files[0].1) {
        extract_zip(&files[0].1, staging.path())?;
    } else {
        for (file_name, bytes) in files {
            let rel = safe_relative_path(&file_name)?;
            let dest = staging.path().join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(dest, bytes)?;
        }
    }

    let skill_root = find_skill_root(staging.path())?;
    let (name, description) = read_skill_metadata(&skill_root.join("SKILL.md"))?;
    let skill_root_rel = skill_root
        .strip_prefix(staging.path())
        .unwrap_or_else(|_| Path::new(""))
        .to_path_buf();
    let persisted_staging = staging.keep();
    Ok(SkillUpload {
        name,
        description,
        source_path: persisted_staging.join(skill_root_rel),
        staging_path: persisted_staging,
    })
}

fn looks_like_zip(file_name: &str, bytes: &[u8]) -> bool {
    file_name.ends_with(".zip") || bytes.starts_with(b"PK\x03\x04")
}

fn extract_zip(bytes: &[u8], dest: &Path) -> Result<()> {
    let reader = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|error| invalid_skill_upload(format!("Invalid skill zip: {error}")))?;
    if archive.len() > MAX_SKILL_FILES {
        return Err(invalid_skill_upload(format!(
            "Skill zip may contain at most {MAX_SKILL_FILES} files."
        )));
    }
    let mut total_bytes = 0usize;
    for i in 0..archive.len() {
        let file = archive
            .by_index(i)
            .map_err(|error| invalid_skill_upload(format!("Invalid skill zip: {error}")))?;
        if file.is_dir() {
            continue;
        }
        if file
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            return Err(invalid_skill_upload("Skill zip may not contain symlinks."));
        }
        let rel = file
            .enclosed_name()
            .ok_or_else(|| invalid_skill_upload("Skill zip contains an unsafe path."))?;
        let out = dest.join(rel);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut buf = Vec::new();
        let remaining = MAX_SKILL_UPLOAD_BYTES.saturating_sub(total_bytes);
        file.take(remaining.saturating_add(1) as u64)
            .read_to_end(&mut buf)
            .map_err(|error| invalid_skill_upload(format!("Invalid skill zip: {error}")))?;
        total_bytes = total_bytes.saturating_add(buf.len());
        if total_bytes > MAX_SKILL_UPLOAD_BYTES {
            return Err(skill_upload_too_large(format!(
                "Skill upload exceeds the {MAX_SKILL_UPLOAD_BYTES} byte limit."
            )));
        }
        fs::write(out, buf)?;
    }
    Ok(())
}

fn safe_relative_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        return Err(invalid_skill_upload("Skill file paths must be relative."));
    }
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => clean.push(part),
            Component::CurDir => {}
            _ => {
                return Err(invalid_skill_upload(format!(
                    "Skill file path `{}` is not allowed.",
                    path.display()
                )))
            }
        }
    }
    if clean.as_os_str().is_empty() {
        return Err(invalid_skill_upload("Skill file path may not be empty."));
    }
    Ok(clean)
}

fn find_skill_root(staging: &Path) -> Result<PathBuf> {
    if staging.join("SKILL.md").is_file() {
        return Ok(staging.to_path_buf());
    }
    let entries = fs::read_dir(staging)?.collect::<std::io::Result<Vec<_>>>()?;
    let mut dirs = Vec::new();
    for entry in &entries {
        if entry.file_type()?.is_dir() {
            dirs.push(entry);
        }
    }
    if dirs.len() != 1 {
        return Err(invalid_skill_upload(
            "Skill upload must contain exactly one top-level folder with SKILL.md.",
        ));
    }
    let root = dirs[0].path();
    if !root.join("SKILL.md").is_file() {
        return Err(invalid_skill_upload(
            "Skill upload top-level folder must contain SKILL.md.",
        ));
    }
    Ok(root)
}

fn read_skill_metadata(skill_md: &Path) -> Result<(String, String)> {
    let bytes = fs::read(skill_md)?;
    let text = String::from_utf8(bytes)
        .map_err(|_| invalid_skill_upload("SKILL.md must contain valid UTF-8."))?;
    let Some(frontmatter) = text
        .strip_prefix("---")
        .and_then(|rest| rest.split_once("---").map(|(meta, _)| meta))
    else {
        return Err(invalid_skill_upload(
            "SKILL.md must start with YAML frontmatter containing `name` and `description`.",
        ));
    };
    let mut name = None;
    let mut description = None;
    for line in frontmatter.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        match key.trim() {
            "name" => name = Some(value),
            "description" => description = Some(value),
            _ => {}
        }
    }
    let name =
        name.ok_or_else(|| invalid_skill_upload("SKILL.md frontmatter is missing `name`."))?;
    let description = description
        .ok_or_else(|| invalid_skill_upload("SKILL.md frontmatter is missing `description`."))?;
    if name.trim().is_empty() || description.trim().is_empty() {
        return Err(invalid_skill_upload(
            "SKILL.md `name` and `description` must be non-empty.",
        ));
    }
    Ok((name, description))
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &dest)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), dest)?;
        }
    }
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_to_rfc3339(timestamp: u64) -> String {
    DateTime::<Utc>::from_timestamp(timestamp as i64, 0)
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
        .to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> (tempfile::TempDir, SkillStore) {
        let root = tempfile::tempdir().unwrap();
        let store = SkillStore::new(root.path().to_path_buf()).unwrap();
        (root, store)
    }

    #[test]
    fn a_poisoned_store_is_an_internal_error() {
        let (_root, store) = test_store();
        let skills = store.skills.clone();
        assert!(std::panic::catch_unwind(move || {
            let _guard = skills.write().unwrap();
            panic!("poison skill store");
        })
        .is_err());
        let error = store.list_versions("skill_missing").unwrap_err();
        let error = ApiError::from_error(error.as_ref(), ApiErrorKind::Internal);
        assert_eq!(error.kind, ApiErrorKind::Internal);
        assert!(!error.message.contains("poison"));
    }

    #[test]
    fn a_missing_version_is_not_found() {
        let (_root, store) = test_store();
        store.skills.write().unwrap().insert(
            "skill_abc".to_string(),
            SkillMetadata {
                id: "skill_abc".to_string(),
                name: "test".to_string(),
                description: "test".to_string(),
                created_at: 1,
                versions: vec![SkillVersionMetadata {
                    version: 1,
                    created_at: 1,
                    source_path: PathBuf::new(),
                }],
            },
        );
        let error = store
            .resolve_references(&[OpenAiShellSkillReference {
                skill_id: "skill_abc".to_string(),
                version: Some(Value::String("2".to_string())),
            }])
            .unwrap_err();
        let error = ApiError::from_error(error.as_ref(), ApiErrorKind::InvalidRequest);
        assert_eq!(error.kind, ApiErrorKind::NotFound);
        assert_eq!(error.code.as_deref(), Some("skill_not_found"));
    }
}
