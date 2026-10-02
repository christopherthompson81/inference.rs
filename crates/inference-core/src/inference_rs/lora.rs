use futures::future::BoxFuture;

use super::*;

impl InferenceRs {
    fn lora_runtime_now(
        &self,
        model_id: Option<&str>,
    ) -> Result<(String, Arc<DynamicLoraRuntime>), InferenceRsError> {
        let resolved_model_id = self.resolve_alias_or_default(model_id)?;
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        let engine = engines
            .get(&resolved_model_id)
            .ok_or_else(|| InferenceRsError::ModelNotFound(resolved_model_id.clone()))?;
        let runtime =
            engine
                .adapter_runtime
                .clone()
                .ok_or_else(|| LoraAdapterError::RuntimeUnavailable {
                    model_id: resolved_model_id.clone(),
                })?;
        Ok((resolved_model_id, runtime))
    }

    async fn lora_runtime(
        &self,
        model_id: Option<&str>,
    ) -> Result<(String, Arc<DynamicLoraRuntime>), InferenceRsError> {
        self.lora_runtime_now(model_id)
    }

    async fn ensure_lora_runtime_current(
        &self,
        model_id: &str,
        expected: &Arc<DynamicLoraRuntime>,
    ) -> Result<(), InferenceRsError> {
        let (_, current) = self.lora_runtime(Some(model_id)).await?;
        if !Arc::ptr_eq(&current, expected) {
            return Err(LoraAdapterError::RuntimeChanged {
                model_id: model_id.to_string(),
            }
            .into());
        }
        Ok(())
    }

    fn log_lora_load(model_id: &str, info: &LoraAdapterInfo, policy: LoraAdapterLoadPolicy) {
        info!(
            model_id,
            alias = %info.alias,
            generation = %info.generation,
            rank = info.rank,
            bytes = info.bytes,
            ?policy,
            "LoRA adapter published"
        );
    }

    fn log_lora_unload(model_id: &str, info: &LoraAdapterInfo) {
        info!(
            model_id,
            alias = %info.alias,
            generation = %info.generation,
            rank = info.rank,
            bytes = info.bytes,
            "LoRA adapter alias removed"
        );
    }

    /// Load a local LoRA adapter directory when its alias is not already registered.
    /// Once admitted to the blocking loader, the operation completes even if this future is dropped.
    pub async fn load_lora_adapter(
        &self,
        model_id: Option<&str>,
        alias: impl Into<String>,
        adapter_dir: impl Into<PathBuf>,
    ) -> Result<LoraAdapterInfo, InferenceRsError> {
        self.load_lora_adapter_with_policy(
            model_id,
            alias,
            adapter_dir,
            LoraAdapterLoadPolicy::Create,
        )
        .await
    }

    /// Load a local LoRA adapter directory using an atomic publication policy.
    pub fn load_lora_adapter_with_policy<'a>(
        &'a self,
        model_id: Option<&'a str>,
        alias: impl Into<String>,
        adapter_dir: impl Into<PathBuf>,
        policy: LoraAdapterLoadPolicy,
    ) -> BoxFuture<'a, Result<LoraAdapterInfo, InferenceRsError>> {
        Box::pin(self.load_lora_adapter_with_policy_inner(
            model_id,
            alias.into(),
            adapter_dir.into(),
            policy,
        ))
    }

    async fn load_lora_adapter_with_policy_inner(
        &self,
        model_id: Option<&str>,
        alias: String,
        adapter_dir: PathBuf,
        policy: LoraAdapterLoadPolicy,
    ) -> Result<LoraAdapterInfo, InferenceRsError> {
        let (resolved_model_id, runtime) = self.lora_runtime(model_id).await?;
        if !runtime.supports_live_updates() {
            return Err(LoraAdapterError::TensorParallelUnsupported {
                model_id: resolved_model_id,
            }
            .into());
        }
        let expected = runtime.clone();
        let permit = DynamicLoraRuntime::try_acquire_load_permit()?;
        let info = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            runtime.load_from_directory_with_policy(alias, adapter_dir, policy)
        })
        .await
        .map_err(LoraAdapterError::Task)?
        .map_err(InferenceRsError::from)?;
        self.ensure_lora_runtime_current(&resolved_model_id, &expected)
            .await?;
        Self::log_lora_load(&resolved_model_id, &info, policy);
        Ok(info)
    }

    /// Load already-open LoRA files using an atomic publication policy.
    pub fn load_lora_adapter_files_with_policy<'a>(
        &'a self,
        model_id: Option<&'a str>,
        alias: impl Into<String>,
        files: LoraAdapterFiles,
        policy: LoraAdapterLoadPolicy,
    ) -> BoxFuture<'a, Result<LoraAdapterInfo, InferenceRsError>> {
        Box::pin(self.load_lora_adapter_files_with_policy_inner(
            model_id,
            alias.into(),
            files,
            policy,
        ))
    }

    async fn load_lora_adapter_files_with_policy_inner(
        &self,
        model_id: Option<&str>,
        alias: String,
        files: LoraAdapterFiles,
        policy: LoraAdapterLoadPolicy,
    ) -> Result<LoraAdapterInfo, InferenceRsError> {
        let (resolved_model_id, runtime) = self.lora_runtime(model_id).await?;
        if !runtime.supports_live_updates() {
            return Err(LoraAdapterError::TensorParallelUnsupported {
                model_id: resolved_model_id,
            }
            .into());
        }
        let expected = runtime.clone();
        let permit = DynamicLoraRuntime::try_acquire_load_permit()?;
        let info = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            runtime.load_from_files_with_policy(alias, files, policy)
        })
        .await
        .map_err(LoraAdapterError::Task)?
        .map_err(InferenceRsError::from)?;
        self.ensure_lora_runtime_current(&resolved_model_id, &expected)
            .await?;
        Self::log_lora_load(&resolved_model_id, &info, policy);
        Ok(info)
    }

    /// Unregister an adapter alias while allowing admitted requests to finish.
    pub fn unload_lora_adapter<'a>(
        &'a self,
        model_id: Option<&'a str>,
        alias: &'a str,
    ) -> BoxFuture<'a, Result<LoraAdapterInfo, InferenceRsError>> {
        Box::pin(self.unload_lora_adapter_inner(model_id, alias))
    }

    async fn unload_lora_adapter_inner(
        &self,
        model_id: Option<&str>,
        alias: &str,
    ) -> Result<LoraAdapterInfo, InferenceRsError> {
        self.unload_lora_adapter_if_generation(model_id, alias, None)
            .await
    }

    /// Unregister an alias only if it still points at the expected generation.
    pub fn unload_lora_adapter_if_generation<'a>(
        &'a self,
        model_id: Option<&'a str>,
        alias: &'a str,
        expected_generation: Option<AdapterGenerationId>,
    ) -> BoxFuture<'a, Result<LoraAdapterInfo, InferenceRsError>> {
        Box::pin(self.unload_lora_adapter_if_generation_inner(model_id, alias, expected_generation))
    }

    async fn unload_lora_adapter_if_generation_inner(
        &self,
        model_id: Option<&str>,
        alias: &str,
        expected_generation: Option<AdapterGenerationId>,
    ) -> Result<LoraAdapterInfo, InferenceRsError> {
        let (resolved_model_id, runtime) = self.lora_runtime(model_id).await?;
        if !runtime.supports_live_updates() {
            return Err(LoraAdapterError::TensorParallelUnsupported {
                model_id: resolved_model_id,
            }
            .into());
        }
        let alias = alias.to_string();
        let expected = runtime.clone();
        let info = tokio::task::spawn_blocking(move || {
            runtime.unload_if_generation(&alias, expected_generation)
        })
        .await
        .map_err(LoraAdapterError::Task)?
        .map_err(InferenceRsError::from)?;
        self.ensure_lora_runtime_current(&resolved_model_id, &expected)
            .await?;
        Self::log_lora_unload(&resolved_model_id, &info);
        Ok(info)
    }

    /// List loaded adapter aliases for a model.
    pub fn list_lora_adapters<'a>(
        &'a self,
        model_id: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Vec<LoraAdapterInfo>, InferenceRsError>> {
        Box::pin(self.list_lora_adapters_inner(model_id))
    }

    async fn list_lora_adapters_inner(
        &self,
        model_id: Option<&str>,
    ) -> Result<Vec<LoraAdapterInfo>, InferenceRsError> {
        let (_, runtime) = self.lora_runtime(model_id).await?;
        Ok(runtime.list())
    }

    /// Return loaded aliases and complete resident-generation capacity usage.
    pub fn lora_adapter_status<'a>(
        &'a self,
        model_id: Option<&'a str>,
    ) -> BoxFuture<'a, Result<LoraRuntimeStatus, InferenceRsError>> {
        Box::pin(self.lora_adapter_status_inner(model_id))
    }

    async fn lora_adapter_status_inner(
        &self,
        model_id: Option<&str>,
    ) -> Result<LoraRuntimeStatus, InferenceRsError> {
        let (_, runtime) = self.lora_runtime(model_id).await?;
        Ok(runtime.status())
    }

    /// List every loaded adapter together with its owning base model.
    pub fn list_lora_adapter_routes(&self) -> Result<Vec<LoraAdapterRoute>, InferenceRsError> {
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        let mut routes = Vec::new();
        for (model_id, engine) in engines.iter() {
            if let Some(runtime) = &engine.adapter_runtime {
                routes.extend(runtime.list().into_iter().map(|adapter| LoraAdapterRoute {
                    model_id: model_id.clone(),
                    adapter,
                }));
            }
        }
        routes.sort_by(|left, right| {
            (&left.model_id, &left.adapter.alias).cmp(&(&right.model_id, &right.adapter.alias))
        });
        Ok(routes)
    }
}
