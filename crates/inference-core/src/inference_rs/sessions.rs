use super::*;

// Each lookup takes the caller's owner (`None` when unscoped) and sees only the entries that owner stored.
impl InferenceRs {
    /// Look up `owner`'s file across all loaded engines. `None` if missing or expired.
    pub fn find_file(&self, id: &str, owner: Option<&str>) -> Option<Arc<files::File>> {
        self.try_find_file(id, owner).ok().flatten()
    }

    /// Fallible variant of [`Self::find_file`].
    pub fn try_find_file(
        &self,
        id: &str,
        owner: Option<&str>,
    ) -> Result<Option<Arc<files::File>>, InferenceRsError> {
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(engines
            .values()
            .find_map(|instance| instance.file_store.get(id, owner)))
    }

    /// Every non-expired file of `owner`'s across all loaded engines, including session-less runs. Order unspecified.
    pub fn list_files(&self, owner: Option<&str>) -> Vec<Arc<files::File>> {
        self.try_list_files(owner).unwrap_or_default()
    }

    /// Fallible variant of [`Self::list_files`].
    pub fn try_list_files(
        &self,
        owner: Option<&str>,
    ) -> Result<Vec<Arc<files::File>>, InferenceRsError> {
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(engines
            .values()
            .flat_map(|instance| instance.file_store.list_all(owner))
            .collect())
    }

    /// Tags `owner`'s file, wherever it is stored, so `try_list_tagged_files(tag)` finds it; false if it is not stored.
    pub fn try_tag_file(
        &self,
        id: &str,
        tag: &str,
        owner: Option<&str>,
    ) -> Result<bool, InferenceRsError> {
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(engines
            .values()
            .any(|instance| instance.file_store.attach_to_session(id, tag, owner)))
    }

    /// `owner`'s non-expired files carrying `tag` (a session id, or a Responses container id), oldest first per engine.
    pub fn try_list_tagged_files(
        &self,
        tag: &str,
        owner: Option<&str>,
    ) -> Result<Vec<Arc<files::File>>, InferenceRsError> {
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(engines
            .values()
            .flat_map(|instance| instance.file_store.list_for_session(tag, owner))
            .collect())
    }

    /// Returns whether `owner`'s file existed.
    pub fn remove_file(&self, id: &str, owner: Option<&str>) -> bool {
        self.try_remove_file(id, owner).unwrap_or(false)
    }

    /// Fallible variant of [`Self::remove_file`].
    pub fn try_remove_file(&self, id: &str, owner: Option<&str>) -> Result<bool, InferenceRsError> {
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(engines
            .values()
            .any(|instance| instance.file_store.remove(id, owner)))
    }

    pub fn insert_file(
        &self,
        model_id: Option<&str>,
        file: files::File,
        session_id: Option<String>,
        owner: Option<&str>,
    ) -> Result<(), InferenceRsError> {
        self.get_file_store(model_id)?
            .insert(file, session_id, owner);
        Ok(())
    }

    pub fn attach_file_to_session(
        &self,
        model_id: Option<&str>,
        id: &str,
        session_id: &str,
        owner: Option<&str>,
    ) -> Result<bool, InferenceRsError> {
        Ok(self
            .get_file_store(model_id)?
            .attach_to_session(id, session_id, owner))
    }

    /// Agentic session store for `model_id` (or the default model). Returns an `Arc` to lock for inspect/mutate.
    pub fn get_session_store(
        &self,
        model_id: Option<&str>,
    ) -> Result<Arc<std::sync::Mutex<engine::agentic_session::AgenticSessionStore>>, InferenceRsError>
    {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        engines
            .get(&resolved_model_id)
            .map(|e| Arc::clone(&e.session_store))
            .ok_or(InferenceRsError::ModelNotFound(resolved_model_id))
    }

    fn get_file_store(&self, model_id: Option<&str>) -> Result<files::FileStore, InferenceRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        engines
            .get(&resolved_model_id)
            .map(|e| e.file_store.clone())
            .ok_or(InferenceRsError::ModelNotFound(resolved_model_id))
    }

    /// Export `owner`'s agentic session by ID. Bundles the session's files (full bodies). `None` if missing.
    pub fn export_session(
        &self,
        model_id: Option<&str>,
        session_id: &str,
        owner: Option<&str>,
    ) -> Result<Option<engine::agentic_session::SerializedSession>, InferenceRsError> {
        let store = self.get_session_store(model_id)?;
        let exported = {
            let mut guard = store.lock().map_err(|_| InferenceRsError::EnginePoisoned)?;
            guard
                .export(session_id, owner)
                .map_err(|e| InferenceRsError::Other(e.to_string()))?
        };
        let Some(mut session) = exported else {
            return Ok(None);
        };
        let file_store = self.get_file_store(model_id)?;
        session.files = file_store
            .list_for_session(session_id, owner)
            .into_iter()
            .map(|arc| (*arc).clone())
            .collect();
        Ok(Some(session))
    }

    /// Replaces `owner`'s session under the same ID and restores its files; a file id already stored keeps its body.
    pub fn import_session(
        &self,
        model_id: Option<&str>,
        session_id: String,
        session: engine::agentic_session::SerializedSession,
        owner: Option<&str>,
    ) -> Result<(), InferenceRsError> {
        let files = session.files.clone();
        let store = self.get_session_store(model_id)?;
        {
            let mut guard = store.lock().map_err(|_| InferenceRsError::EnginePoisoned)?;
            guard
                .import(session_id.clone(), session, owner)
                .map_err(|e| InferenceRsError::Other(e.to_string()))?;
        }
        let file_store = self.get_file_store(model_id)?;
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        for f in files {
            // ids resolve in every engine's store, and a body the import didn't write must not change under its id
            let retagged = engines.values().any(|instance| {
                instance
                    .file_store
                    .attach_to_session(&f.id, &session_id, owner)
            });
            let held = retagged
                || engines
                    .values()
                    .any(|instance| instance.file_store.holds(&f.id));
            if !held {
                file_store.insert(f, Some(session_id.clone()), owner);
            }
        }
        Ok(())
    }

    /// Clone the first `num_turns` complete turns from `src` into `dest`. A turn ends at the
    /// first assistant message without `tool_calls`. Used for branching: the new session diverges
    /// cleanly from the truncated prefix, so the branch's later edits don't bleed back.
    pub fn fork_session(
        &self,
        model_id: Option<&str>,
        src_session_id: &str,
        dest_session_id: String,
        num_turns: usize,
        owner: Option<&str>,
    ) -> Result<(), InferenceRsError> {
        let store = self.get_session_store(model_id)?;
        let mut guard = store.lock().map_err(|_| InferenceRsError::EnginePoisoned)?;
        guard
            .fork(src_session_id, dest_session_id, num_turns, owner)
            .map_err(|e| InferenceRsError::Other(e.to_string()))
    }

    /// Delete `owner`'s agentic session. Returns whether the session existed.
    pub fn delete_session(
        &self,
        model_id: Option<&str>,
        session_id: &str,
        owner: Option<&str>,
    ) -> Result<bool, InferenceRsError> {
        let store = self.get_session_store(model_id)?;
        let mut guard = store.lock().map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(guard.delete(session_id, owner))
    }

    /// `owner`'s stored session IDs.
    pub fn list_session_ids(
        &self,
        model_id: Option<&str>,
        owner: Option<&str>,
    ) -> Result<Vec<String>, InferenceRsError> {
        let store = self.get_session_store(model_id)?;
        let guard = store.lock().map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(guard.list_ids(owner))
    }
}
