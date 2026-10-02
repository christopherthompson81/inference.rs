use futures::future::BoxFuture;

use super::*;

impl InferenceRs {
    fn resolve_alias(&self, model_id: &str) -> Result<String, InferenceRsError> {
        let aliases = self
            .model_aliases
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if let Some(primary_id) = aliases.get(model_id) {
            Ok(primary_id.clone())
        } else {
            Ok(model_id.to_string())
        }
    }

    /// The registered id `model_id` (or the default model, for `None`) names.
    pub fn resolve_alias_or_default(
        &self,
        model_id: Option<&str>,
    ) -> Result<String, InferenceRsError> {
        match model_id {
            Some(id) => self.resolve_alias(id),
            None => {
                let default_lock = self
                    .default_engine_id
                    .read()
                    .map_err(|_| InferenceRsError::EnginePoisoned)?;
                Ok(default_lock
                    .as_ref()
                    .ok_or_else(|| InferenceRsError::ModelNotFound("default".to_string()))?
                    .clone())
            }
        }
    }

    /// Register an alternate model ID that resolves to an existing model.
    pub fn register_model_alias(
        &self,
        alias: impl Into<String>,
        model_id: &str,
    ) -> Result<(), String> {
        let alias = alias.into();
        let resolved_model_id = self.resolve_alias(model_id).map_err(|e| e.to_string())?;

        if alias == resolved_model_id {
            return Ok(());
        }

        let reloading = self
            .reloading_models
            .read()
            .map_err(|_| "Failed to acquire read lock on reloading_models")?;
        let model_reloading = reloading.contains(&resolved_model_id);
        let alias_conflict = reloading.contains(&alias);
        drop(reloading);

        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        let model_loaded = engines.contains_key(&resolved_model_id);
        let alias_conflict = alias_conflict || engines.contains_key(&alias);
        drop(engines);

        let unloaded = self
            .unloaded_models
            .read()
            .map_err(|_| "Failed to acquire read lock on unloaded_models")?;
        let model_unloaded = unloaded.contains_key(&resolved_model_id);
        let alias_conflict = alias_conflict || unloaded.contains_key(&alias);
        drop(unloaded);

        if !(model_loaded || model_unloaded || model_reloading) {
            return Err(format!("Model {resolved_model_id} not found"));
        }

        if alias_conflict {
            return Err(format!(
                "Alias '{}' conflicts with an existing model ID",
                alias
            ));
        }

        let mut aliases = self
            .model_aliases
            .write()
            .map_err(|_| "Failed to acquire write lock on model_aliases")?;
        if let Some(existing) = aliases.get(&alias) {
            if existing == &resolved_model_id {
                return Ok(());
            }
            return Err(format!(
                "Alias '{}' is already assigned to model '{}'",
                alias, existing
            ));
        }
        aliases.insert(alias, resolved_model_id);
        Ok(())
    }

    /// Check if a model is known (loaded, unloaded, or reloading), resolving aliases if needed.
    pub fn model_exists(&self, model_id: &str) -> Result<bool, InferenceRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;

        let reloading = self
            .reloading_models
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if reloading.contains(&resolved_model_id) {
            return Ok(true);
        }
        drop(reloading);

        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if engines.contains_key(&resolved_model_id) {
            return Ok(true);
        }
        drop(engines);

        let unloaded = self
            .unloaded_models
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if unloaded.contains_key(&resolved_model_id) {
            return Ok(true);
        }

        Ok(false)
    }

    /// Get model category for a specific model. If model_id is None, uses default engine.
    pub fn get_model_category(
        &self,
        model_id: Option<&str>,
    ) -> Result<ModelCategory, InferenceRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.category.clone())
        } else {
            Err(InferenceRsError::EnginePoisoned)
        }
    }

    /// Get the maximum supported sequence length for a model, if applicable.
    pub fn max_sequence_length(
        &self,
        model_id: Option<&str>,
    ) -> Result<Option<usize>, InferenceRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;

        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if let Some(engine_instance) = engines.get(&resolved_model_id) {
            Ok(engine_instance.config.max_seq_len)
        } else {
            Err(InferenceRsError::EnginePoisoned)
        }
    }

    /// Add a new model engine to the InferenceRs instance
    pub fn add_model<'a>(
        &'a self,
        model_id: String,
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        config: AddModelConfig,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(self.add_model_inner(model_id, pipeline, method, config))
    }

    async fn add_model_inner(
        &self,
        model_id: String,
        pipeline: Arc<tokio::sync::Mutex<dyn Pipeline>>,
        method: SchedulerConfig,
        config: AddModelConfig,
    ) -> Result<(), String> {
        {
            let reloading = self
                .reloading_models
                .read()
                .map_err(|_| "Failed to acquire read lock on reloading_models")?;
            if reloading.contains(&model_id) {
                return Err(format!("Model {model_id} is currently reloading"));
            }
        }
        {
            let engines = self
                .engines
                .read()
                .map_err(|_| "Failed to acquire read lock on engines")?;
            if engines.contains_key(&model_id) {
                return Err(format!("Model {model_id} already exists"));
            }
        }
        {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| "Failed to acquire read lock on unloaded_models")?;
            if unloaded.contains_key(&model_id) {
                return Err(format!("Model {model_id} already exists (unloaded)"));
            }
        }
        {
            let aliases = self
                .model_aliases
                .read()
                .map_err(|_| "Failed to acquire read lock on model_aliases")?;
            if aliases.contains_key(&model_id) {
                return Err(format!(
                    "Model ID '{}' conflicts with an existing alias",
                    model_id
                ));
            }
        }

        let mut engine_config = config.engine_config;
        Self::init_external_tool_callbacks(
            &pipeline,
            &mut engine_config.tool_callbacks,
            config.mcp_client_config.as_ref(),
            config.code_exec_config.as_ref(),
            config.shell_config.as_ref(),
        )
        .await;

        let reboot_state = RebootState {
            pipeline,
            method,
            engine_config,
            mcp_client_config: config.mcp_client_config.clone(),
            loader_config: config.loader_config.clone(),
        };

        let engine_instance = Self::create_engine_instance(reboot_state)?;

        let mut engines = self
            .engines
            .write()
            .map_err(|_| "Failed to acquire write lock on engines")?;
        engines.insert(model_id.clone(), engine_instance);

        // If this is the first model, set it as default
        if engines.len() == 1 {
            let mut default_lock = self
                .default_engine_id
                .write()
                .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
            *default_lock = Some(model_id.clone());
            info!("First model added, setting '{}' as default", model_id);
        }

        Ok(())
    }

    /// Remove a model engine from the InferenceRs instance
    pub fn remove_model(&self, model_id: &str) -> Result<(), String> {
        let resolved_model_id = self.resolve_alias(model_id).map_err(|e| e.to_string())?;
        let mut engines = self
            .engines
            .write()
            .map_err(|_| "Failed to acquire write lock on engines")?;

        if engines.len() <= 1 {
            return Err("Cannot remove the last model from InferenceRs".to_string());
        }

        match engines.remove(&resolved_model_id) {
            Some(engine_instance) => {
                // If this was the default engine, set a new default
                let mut default_lock = self
                    .default_engine_id
                    .write()
                    .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
                if let Some(ref default_id) = *default_lock
                    && default_id == &resolved_model_id
                {
                    // Set the first available engine as the new default
                    *default_lock = engines.keys().next().cloned();
                }
                drop(default_lock);
                drop(engines);

                // Remove any aliases pointing to the removed model
                let mut aliases = self
                    .model_aliases
                    .write()
                    .map_err(|_| "Failed to acquire write lock on model_aliases")?;
                aliases.retain(|_, target| target != &resolved_model_id);
                drop(aliases);

                // Sent with no lock held: a full request channel would otherwise stall every lookup until it drained.
                let _ = engine_instance.sender.blocking_send(Request::Terminate);
                Ok(())
            }
            _ => Err(format!("Model {resolved_model_id} not found")),
        }
    }

    /// List all available model IDs
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        Ok(engines.keys().cloned().collect())
    }

    /// Get the current default model ID
    pub fn get_default_model_id(&self) -> Result<Option<String>, String> {
        let default_lock = self
            .default_engine_id
            .read()
            .map_err(|_| "Failed to acquire read lock on default_engine_id")?;
        Ok(default_lock.clone())
    }

    /// Set the default model ID
    pub fn set_default_model_id(&self, model_id: &str) -> Result<(), String> {
        let resolved_model_id = self.resolve_alias(model_id).map_err(|e| e.to_string())?;
        let engines = self
            .engines
            .read()
            .map_err(|_| "Failed to acquire read lock on engines")?;
        if !engines.contains_key(&resolved_model_id) {
            return Err(format!("Model {resolved_model_id} not found"));
        }
        drop(engines);

        let mut default_lock = self
            .default_engine_id
            .write()
            .map_err(|_| "Failed to acquire write lock on default_engine_id")?;
        let old_default = default_lock.clone();
        *default_lock = Some(resolved_model_id.clone());

        // Log the change
        info!(
            "Default model changed: {:?} -> {:?}",
            old_default, resolved_model_id
        );

        Ok(())
    }

    /// Unload a model from memory while preserving its configuration for later reload.
    /// The model can be reloaded automatically when a request is sent to it, or manually
    /// using `reload_model()`.
    ///
    /// Note: The model must have been added with a `ModelLoaderConfig` for auto-reload to work.
    /// Models added via `InferenceRsBuilder` without explicit loader config cannot be reloaded.
    pub fn unload_model(&self, model_id: &str) -> Result<(), InferenceRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        // Check if already unloaded
        {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            if unloaded.contains_key(&resolved_model_id) {
                return Err(InferenceRsError::ModelAlreadyUnloaded(
                    resolved_model_id.clone(),
                ));
            }
        }

        // Get the engine instance and create UnloadedModelState
        let mut engines = self
            .engines
            .write()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;

        let engine_instance = engines
            .get(&resolved_model_id)
            .ok_or_else(|| InferenceRsError::ModelNotFound(resolved_model_id.clone()))?;

        let loader_config = engine_instance
            .reboot_state
            .loader_config
            .clone()
            .ok_or_else(|| InferenceRsError::NoLoaderConfig(resolved_model_id.clone()))?;
        let engine_instance = engines
            .remove(&resolved_model_id)
            .expect("engine was present while holding the write lock");

        // Create the unloaded state
        let unloaded_state = UnloadedModelState {
            loader_config,
            scheduler_config: engine_instance.reboot_state.method.clone(),
            engine_config: engine_instance.reboot_state.engine_config.clone(),
            mcp_client_config: engine_instance.reboot_state.mcp_client_config.clone(),
            category: engine_instance.category.clone(),
            inference_config: engine_instance.config.clone(),
        };

        // Send terminate signal to the engine
        let _ = engine_instance.sender.try_send(Request::Terminate);

        drop(engines);

        // Store the unloaded state
        let mut unloaded = self
            .unloaded_models
            .write()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        unloaded.insert(resolved_model_id.to_string(), unloaded_state);

        // Update default if needed
        let mut default_lock = self
            .default_engine_id
            .write()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        if let Some(ref default_id) = *default_lock
            && default_id == &resolved_model_id
        {
            // Set the first available engine as the new default
            let engines = self
                .engines
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            *default_lock = engines.keys().next().cloned();
        }

        info!("Model {} unloaded successfully", resolved_model_id);
        Ok(())
    }

    /// Manually reload a previously unloaded model.
    /// This is also called automatically by `get_sender()` when a request targets an unloaded model.
    pub fn reload_model<'a>(
        &'a self,
        model_id: &'a str,
    ) -> BoxFuture<'a, Result<(), InferenceRsError>> {
        Box::pin(self.reload_model_inner(model_id))
    }

    async fn reload_model_inner(&self, model_id: &str) -> Result<(), InferenceRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        // Marked before the checks so two reloads cannot both pass them; the guard clears it even if this is dropped.
        if !self
            .reloading_models
            .write()
            .map_err(|_| InferenceRsError::EnginePoisoned)?
            .insert(resolved_model_id.clone())
        {
            return Err(InferenceRsError::ModelReloading(resolved_model_id));
        }
        let _reloading = ReloadingMark {
            reloading: &self.reloading_models,
            model_id: &resolved_model_id,
        };
        if self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?
            .contains_key(&resolved_model_id)
        {
            return Err(InferenceRsError::ModelAlreadyLoaded(
                resolved_model_id.clone(),
            ));
        }
        let unloaded_state = self
            .unloaded_models
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?
            .get(&resolved_model_id)
            .cloned()
            .ok_or_else(|| InferenceRsError::ModelNotFound(resolved_model_id.clone()))?;
        self.do_reload_model(&resolved_model_id, unloaded_state)
            .await
    }

    /// Internal method to perform the actual model reload
    async fn do_reload_model(
        &self,
        model_id: &str,
        unloaded_state: UnloadedModelState,
    ) -> Result<(), InferenceRsError> {
        info!("Reloading model: {}", model_id);

        let loader_config = &unloaded_state.loader_config;

        let loader = loader_config
            .build_loader(unloaded_state.engine_config.no_kv_cache)
            .map_err(|e| InferenceRsError::ReloadFailed(format!("Failed to build loader: {e}")))?;
        let prefix_cache_capacity = if unloaded_state.engine_config.no_prefix_cache {
            0
        } else {
            unloaded_state.engine_config.prefix_cache_n
        };
        let pipeline = loader_config
            .load(&*loader, MtpRuntimeConfig::new(prefix_cache_capacity))
            .await
            .map_err(|e| InferenceRsError::ReloadFailed(format!("Failed to load model: {e}")))?;
        let realized_cache_config = pipeline.lock().await.get_metadata().cache_config.clone();
        let mut scheduler_config = unloaded_state.scheduler_config;
        scheduler_config
            .refresh_paged_cache_config(realized_cache_config)
            .map_err(|e| {
                InferenceRsError::ReloadFailed(format!(
                    "Failed to refresh scheduler cache configuration: {e}"
                ))
            })?;

        // Create the reboot state
        let reboot_state = RebootState {
            pipeline: pipeline.clone(),
            method: scheduler_config,
            engine_config: unloaded_state.engine_config,
            mcp_client_config: unloaded_state.mcp_client_config.clone(),
            loader_config: Some(unloaded_state.loader_config.clone()),
        };

        let engine_instance = Self::create_engine_instance(reboot_state)
            .map_err(|e| InferenceRsError::ReloadFailed(format!("Failed to create engine: {e}")))?;

        // Add to engines map
        {
            let mut engines = self
                .engines
                .write()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            engines.insert(model_id.to_string(), engine_instance);
        }
        // Unloading the last model cleared the default; without this, requests naming no model fail after reload.
        {
            let mut default_lock = self
                .default_engine_id
                .write()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            if default_lock.is_none() {
                *default_lock = Some(model_id.to_string());
            }
        }

        // Remove from unloaded map
        {
            let mut unloaded = self
                .unloaded_models
                .write()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            unloaded.remove(model_id);
        }

        info!("Model {} reloaded successfully", model_id);
        Ok(())
    }

    /// Synchronous version of reload_model for use in non-async contexts.
    ///
    /// This method handles different runtime contexts appropriately:
    /// - If called from a multi-threaded tokio runtime, uses `block_in_place`
    /// - If called from a single-threaded runtime, returns an error (use `reload_model()` instead)
    /// - If called outside any runtime, creates a temporary runtime
    pub fn reload_model_blocking(&self, model_id: &str) -> Result<(), InferenceRsError> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
                    Err(InferenceRsError::ReloadFailed(
                        "Cannot reload model blocking from single-threaded runtime. Use reload_model() instead.".to_string()
                    ))
                } else {
                    tokio::task::block_in_place(|| handle.block_on(self.reload_model(model_id)))
                }
            }
            Err(_) => {
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    InferenceRsError::ReloadFailed(format!("Failed to create runtime: {e}"))
                })?;
                rt.block_on(self.reload_model(model_id))
            }
        }
    }

    /// Check if a model is currently loaded (as opposed to unloaded)
    pub fn is_model_loaded(&self, model_id: &str) -> Result<bool, InferenceRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        Ok(engines.contains_key(&resolved_model_id))
    }

    /// Get the status of a model, or None if not found
    pub fn get_model_status(
        &self,
        model_id: &str,
    ) -> Result<Option<ModelStatus>, InferenceRsError> {
        let resolved_model_id = self.resolve_alias(model_id)?;
        // Check if reloading
        {
            let reloading = self
                .reloading_models
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            if reloading.contains(&resolved_model_id) {
                return Ok(Some(ModelStatus::Reloading));
            }
        }

        // Check if loaded
        {
            let engines = self
                .engines
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            if engines.contains_key(&resolved_model_id) {
                return Ok(Some(ModelStatus::Loaded));
            }
        }

        // Check if unloaded
        {
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?;
            if unloaded.contains_key(&resolved_model_id) {
                return Ok(Some(ModelStatus::Unloaded));
            }
        }

        Ok(None)
    }

    /// List all models with their status
    pub fn list_models_with_status(&self) -> Result<Vec<(String, ModelStatus)>, InferenceRsError> {
        let mut result = Vec::new();

        // Get reloading models
        let reloading = self
            .reloading_models
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        for model_id in reloading.iter() {
            result.push((model_id.clone(), ModelStatus::Reloading));
        }
        drop(reloading);

        // Get loaded models
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        for model_id in engines.keys() {
            result.push((model_id.clone(), ModelStatus::Loaded));
        }
        drop(engines);

        // Get unloaded models
        let unloaded = self
            .unloaded_models
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        for model_id in unloaded.keys() {
            // Skip if already in reloading
            if !result.iter().any(|(id, _)| id == model_id) {
                result.push((model_id.clone(), ModelStatus::Unloaded));
            }
        }

        Ok(result)
    }
}
