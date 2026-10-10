using InferenceRs.Native;

namespace InferenceRs;

/// <summary>A media buffer a request names by position: an image, audio or video URL of <c>media://0</c> is the first.</summary>
public sealed record MediaAttachment(byte[] Data, string? MimeType = null);

/// <summary>One file of a skill upload: its path within the skill (<c>SKILL.md</c>, <c>scripts/run.py</c>) and bytes.</summary>
public sealed record SkillFile(string Path, byte[] Data);

/// <summary>
/// A loaded model serving OpenAI-style requests. Requests and responses are the JSON the HTTP server accepts and
/// returns; failures throw <see cref="InferenceException"/> with the protocol's error JSON as its detail.
/// </summary>
/// <remarks>Calls block. An engine may be used from several threads at once.</remarks>
public sealed unsafe class InferenceEngine : IDisposable
{
    private readonly EngineHandle _handle;

    private InferenceEngine(EngineHandle handle) => _handle = handle;

    /// <summary>The ABI version the native library implements, as (major &lt;&lt; 16) | (minor &lt;&lt; 8) | patch.</summary>
    public static uint AbiVersion => NativeMethods.inference_abi_version();

    /// <summary>The native library's build identification.</summary>
    public static string BuildVersion => Utf8.ToString(NativeMethods.inference_build_version());

    /// <summary>Loads an engine from its JSON spec (see <c>inference_engine_load</c> in inference.h).</summary>
    public static InferenceEngine Load(string specJson, HostCallbacks? callbacks = null)
    {
        NativeMethods.EnsureAbi();
        using var spec = new PinnedBytes(specJson);
        if (callbacks is null)
        {
            var status = NativeMethods.inference_engine_load(spec.Pointer, spec.Length, out var engine);
            InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_engine_load));
            return new InferenceEngine(new EngineHandle(engine, []));
        }
        using var pinned = new HostCallbackBridge.Pinned(callbacks);
        var loaded = NativeMethods.inference_engine_load_with_callbacks(
            spec.Pointer, spec.Length, pinned.Native, out var withCallbacks);
        InferenceException.ThrowIfFailed(loaded, nameof(NativeMethods.inference_engine_load_with_callbacks));
        return new InferenceEngine(new EngineHandle(withCallbacks, pinned.TakeIds()));
    }

    /// <summary>The same engine acting for <paramref name="owner"/>: what it stores is that owner's, and it reaches no one else's.</summary>
    /// <remarks>Dispose it like any engine; this one keeps its callbacks until every engine made from it is disposed.</remarks>
    public InferenceEngine ForOwner(string owner)
    {
        using var engine = Borrow();
        using var name = new PinnedBytes(owner);
        var status = NativeMethods.inference_engine_for_owner(engine.Handle, name.Pointer, name.Length, out var scoped);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_engine_for_owner));
        return new InferenceEngine(new EngineHandle(scoped, _handle));
    }

    /// <summary>Makes <paramref name="processor"/> selectable by name in a request's <c>logits_processors</c>, on every engine sharing this one.</summary>
    /// <remarks>The result keeps the engine open until disposed, which unregisters it; requests still running that named it then fail.</remarks>
    public IDisposable RegisterLogitsProcessor(string name, LogitsProcessor processor)
    {
        using var engine = Borrow();
        using var bytes = new PinnedBytes(name);
        var id = HostCallbackRegistry.Add(processor);
        var status = NativeMethods.inference_engine_register_logits_processor(
            engine.Handle, bytes.Pointer, bytes.Length, &HostCallbackBridge.Logits, id);
        if (status != InferenceStatus.Ok) HostCallbackRegistry.Remove([id]);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_engine_register_logits_processor));
        // The borrow still holds the engine, so the registration's own reference cannot fail.
        return new HostRegistration(_handle, name, id, &NativeMethods.inference_engine_unregister_logits_processor);
    }

    /// <summary>Registers <paramref name="tool"/> after load; a chat request offers it by naming it in <c>host_tools</c>.</summary>
    /// <remarks>The result keeps the engine open until disposed, which unregisters the tool; requests still running that named it then fail it.</remarks>
    public IDisposable RegisterTool(HostTool tool)
    {
        var named = System.Text.Json.Nodes.JsonNode.Parse(tool.DefinitionJson)?["function"]?["name"];
        var name = named is System.Text.Json.Nodes.JsonValue value && value.TryGetValue<string>(out var text)
            ? text
            : throw new ArgumentException("the definition has no function.name", nameof(tool));
        using var engine = Borrow();
        using var definition = new PinnedBytes(tool.DefinitionJson);
        var id = HostCallbackRegistry.Add(tool.Handler);
        var native = new NativeHostTool
        {
            Definition = definition.Pointer,
            DefinitionLen = definition.Length,
            Callback = &HostCallbackBridge.Tool,
            UserData = id,
        };
        var status = NativeMethods.inference_engine_register_tool(engine.Handle, &native);
        if (status != InferenceStatus.Ok) HostCallbackRegistry.Remove([id]);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_engine_register_tool));
        return new HostRegistration(_handle, name, id, &NativeMethods.inference_engine_unregister_tool);
    }

    public string Chat(string requestJson, IReadOnlyList<MediaAttachment>? media = null)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        using var attachments = Media(media);
        var status = NativeMethods.inference_chat_with_media(
            engine.Handle, request.Pointer, request.Length, attachments.Pointer, attachments.Count, out var response);
        return Text(status, response, nameof(NativeMethods.inference_chat_with_media));
    }

    public EngineStream ChatStream(string requestJson, IReadOnlyList<MediaAttachment>? media = null)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        using var attachments = Media(media);
        var status = NativeMethods.inference_chat_stream_open_with_media(
            engine.Handle, request.Pointer, request.Length, attachments.Pointer, attachments.Count, out var stream);
        return Stream(status, stream, nameof(NativeMethods.inference_chat_stream_open_with_media));
    }

    public string Completion(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_completion(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_completion));
    }

    public EngineStream CompletionStream(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_completion_stream_open(
            engine.Handle, request.Pointer, request.Length, out var stream);
        return Stream(status, stream, nameof(NativeMethods.inference_completion_stream_open));
    }

    public string Embeddings(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_embeddings(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_embeddings));
    }

    /// <summary>An Anthropic Messages request; failures carry the Anthropic error envelope.</summary>
    public string AnthropicMessages(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_anthropic_messages(
            engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_anthropic_messages));
    }

    /// <summary>The prompt tokens an Anthropic Messages request would use: {"input_tokens"}.</summary>
    public string AnthropicCountTokens(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_anthropic_count_tokens(
            engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_anthropic_count_tokens));
    }

    public EngineStream AnthropicMessagesStream(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_anthropic_messages_stream_open(
            engine.Handle, request.Pointer, request.Length, out var stream);
        return Stream(status, stream, nameof(NativeMethods.inference_anthropic_messages_stream_open));
    }

    public string CreateResponse(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_responses_create(
            engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_responses_create));
    }

    public EngineStream ResponseStream(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_responses_stream_open(
            engine.Handle, request.Pointer, request.Length, out var stream);
        return Stream(status, stream, nameof(NativeMethods.inference_responses_stream_open));
    }

    public string GetResponse(string responseId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(responseId);
        var status = NativeMethods.inference_responses_get(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_responses_get));
    }

    public string DeleteResponse(string responseId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(responseId);
        var status = NativeMethods.inference_responses_delete(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_responses_delete));
    }

    public string CancelResponse(string responseId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(responseId);
        var status = NativeMethods.inference_responses_cancel(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_responses_cancel));
    }

    public string ListModels()
    {
        using var engine = Borrow();
        var status = NativeMethods.inference_models_list(engine.Handle, out var response);
        return Text(status, response, nameof(NativeMethods.inference_models_list));
    }

    /// <summary>Whether a request naming the model in {"model_id"} would be routed: {"model_id", "served"}.</summary>
    public string ModelServed(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_served(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_served));
    }

    /// <summary>The tools the engine's MCP servers give the default model.</summary>
    public string ListMcpTools()
    {
        using var engine = Borrow();
        var status = NativeMethods.inference_mcp_tools_list(engine.Handle, out var response);
        return Text(status, response, nameof(NativeMethods.inference_mcp_tools_list));
    }

    /// <summary>Loads another model into the running engine; the request is one entry of the spec's &quot;models&quot;.</summary>
    public string AddModel(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_add(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_add));
    }

    /// <summary>Stops serving a model and frees it; the request is {&quot;model_id&quot;}.</summary>
    public string RemoveModel(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_remove(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_remove));
    }

    /// <summary>Makes a served model the default; the request is {&quot;model_id&quot;}.</summary>
    public string SetDefaultModel(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_set_default(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_set_default));
    }

    /// <summary>Lets requests name a served model by another id; the request is {&quot;alias&quot;, &quot;model_id&quot;}.</summary>
    public string AddModelAlias(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_alias(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_alias));
    }

    public string UnloadModel(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_unload(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_unload));
    }

    public string ReloadModel(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_reload(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_reload));
    }

    public string ModelStatus(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_status(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_status));
    }

    /// <summary>Requantizes a model that loaded with ISQ; answers once it is requantized.</summary>
    public string ReIsq(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_re_isq(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_re_isq));
    }

    /// <summary>Starts collecting activation statistics from the requests the engine serves.</summary>
    public string CalibrationStart(string requestJson = "{}")
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_calibration_start(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_calibration_start));
    }

    /// <summary>Each loaded model's cumulative prefix- and encoder-cache counters; diff two readings for a span.</summary>
    public string CacheStats()
    {
        using var engine = Borrow();
        var status = NativeMethods.inference_models_cache_stats(engine.Handle, out var response);
        return Text(status, response, nameof(NativeMethods.inference_models_cache_stats));
    }

    /// <summary>Each loaded model's cumulative speculative decoding counters; zero without a proposer.</summary>
    public string SpeculativeStats()
    {
        using var engine = Borrow();
        var status = NativeMethods.inference_models_speculative_stats(engine.Handle, out var response);
        return Text(status, response, nameof(NativeMethods.inference_models_speculative_stats));
    }

    public string CalibrationStatus(string requestJson = "{}")
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_calibration_status(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_calibration_status));
    }

    /// <summary>Requantizes from the collected statistics; returns the status as it stood before.</summary>
    public string CalibrationApply(string requestJson = "{}")
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_calibration_apply(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_calibration_apply));
    }

    public string ListSessions()
    {
        using var engine = Borrow();
        var status = NativeMethods.inference_sessions_list(engine.Handle, out var response);
        return Text(status, response, nameof(NativeMethods.inference_sessions_list));
    }

    public string GetSession(string sessionId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(sessionId);
        var status = NativeMethods.inference_session_get(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_session_get));
    }

    /// <summary>Branches a session into a new one the engine names; the request is {"num_turns"}, the answer {"id"}.</summary>
    public string ForkSession(string sessionId, string requestJson)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(sessionId);
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_session_fork(
            engine.Handle, id.Pointer, id.Length, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_session_fork));
    }

    /// <summary>Imports a session under <paramref name="sessionId"/>, replacing any session there.</summary>
    public string PutSession(string sessionId, string sessionJson)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(sessionId);
        using var session = new PinnedBytes(sessionJson);
        var status = NativeMethods.inference_session_put(
            engine.Handle, id.Pointer, id.Length, session.Pointer, session.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_session_put));
    }

    public string DeleteSession(string sessionId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(sessionId);
        var status = NativeMethods.inference_session_delete(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_session_delete));
    }

    /// <summary>Scores a prompt: each token's log-probability, and its row-major logits when asked for.</summary>
    public (string Scores, float[]? Logits) PromptLogits(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_prompt_logits(
            engine.Handle, request.Pointer, request.Length, out var response, out var blob);
        var scores = Text(status, response, nameof(NativeMethods.inference_prompt_logits));
        if (blob == IntPtr.Zero) return (scores, null);
        var bytes = Owned.TakeBlob(blob).Data;
        var logits = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, logits, 0, bytes.Length);
        return (scores, logits);
    }

    public string Tokenize(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_tokenize(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_tokenize));
    }

    /// <summary>Tokenizes a chat completion request as the model's chat template renders it, to <c>{"tokens"}</c>.</summary>
    public string TokenizeChat(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_tokenize_chat(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_tokenize_chat));
    }

    public string Detokenize(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_detokenize(engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_detokenize));
    }

    public string ListLoraAdapters(string requestJson = "{}")
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_lora_adapters_list(
            engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_lora_adapters_list));
    }

    public string LoadLoraAdapter(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_lora_adapter_load(
            engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_lora_adapter_load));
    }

    public string UnloadLoraAdapter(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_lora_adapter_unload(
            engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_lora_adapter_unload));
    }

    public string ImageGeneration(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_image_generation(
            engine.Handle, request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_image_generation));
    }

    /// <summary>Speaks text; the blob's MIME type carries the sample rate and channel count.</summary>
    public Blob SpeechGeneration(string requestJson)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_speech_generation(engine.Handle, request.Pointer, request.Length, out var blob);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_speech_generation));
        return Owned.TakeBlob(blob);
    }

    /// <summary>Transcribes encoded audio (WAV, MP3, FLAC, ...); the blob's MIME type names the response format.</summary>
    public Blob Transcription(string requestJson, byte[] audio)
    {
        using var engine = Borrow();
        using var request = new PinnedBytes(requestJson);
        using var bytes = new PinnedBytes(audio);
        var status = NativeMethods.inference_transcription(
            engine.Handle, request.Pointer, request.Length, bytes.Pointer, bytes.Length, out var blob);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_transcription));
        return Owned.TakeBlob(blob);
    }

    /// <summary>Answers the approval an <c>agentic_tool_approval_required</c> stream event named.</summary>
    public string ResolveApproval(string approvalId, string decisionJson)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(approvalId);
        using var decision = new PinnedBytes(decisionJson);
        var status = NativeMethods.inference_approval_resolve(
            engine.Handle, id.Pointer, id.Length, decision.Pointer, decision.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_approval_resolve));
    }

    public string UploadFile(byte[] data, string filename, string purpose, string? mimeType = null)
    {
        using var engine = Borrow();
        using var bytes = new PinnedBytes(data);
        var status = NativeMethods.inference_file_upload(
            engine.Handle, bytes.Pointer, bytes.Length, filename, mimeType, purpose, out var response);
        return Text(status, response, nameof(NativeMethods.inference_file_upload));
    }

    public string ListFiles()
    {
        using var engine = Borrow();
        var status = NativeMethods.inference_files_list(engine.Handle, out var response);
        return Text(status, response, nameof(NativeMethods.inference_files_list));
    }

    public string GetFile(string fileId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(fileId);
        var status = NativeMethods.inference_file_get(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_file_get));
    }

    public string DeleteFile(string fileId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(fileId);
        var status = NativeMethods.inference_file_delete(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_file_delete));
    }

    public Blob FileContent(string fileId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(fileId);
        var status = NativeMethods.inference_file_content(engine.Handle, id.Pointer, id.Length, out var blob);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_file_content));
        return Owned.TakeBlob(blob);
    }

    /// <summary>The files a Responses container (a code-running session) produced.</summary>
    public string ListContainerFiles(string containerId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(containerId);
        var status = NativeMethods.inference_container_files_list(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_container_files_list));
    }

    public string GetContainerFile(string containerId, string fileId)
    {
        using var engine = Borrow();
        using var container = new PinnedBytes(containerId);
        using var file = new PinnedBytes(fileId);
        var status = NativeMethods.inference_container_file_get(
            engine.Handle, container.Pointer, container.Length, file.Pointer, file.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_container_file_get));
    }

    public Blob ContainerFileContent(string containerId, string fileId)
    {
        using var engine = Borrow();
        using var container = new PinnedBytes(containerId);
        using var file = new PinnedBytes(fileId);
        var status = NativeMethods.inference_container_file_content(
            engine.Handle, container.Pointer, container.Length, file.Pointer, file.Length, out var blob);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_container_file_content));
        return Owned.TakeBlob(blob);
    }

    public string UploadSkill(IReadOnlyList<SkillFile> files)
    {
        using var engine = Borrow();
        using var native = Skill(files);
        var status = NativeMethods.inference_skill_upload(engine.Handle, native.Pointer, native.Count, out var response);
        return Text(status, response, nameof(NativeMethods.inference_skill_upload));
    }

    public string UploadSkillVersion(string skillId, IReadOnlyList<SkillFile> files)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(skillId);
        using var native = Skill(files);
        var status = NativeMethods.inference_skill_version_upload(
            engine.Handle, id.Pointer, id.Length, native.Pointer, native.Count, out var response);
        return Text(status, response, nameof(NativeMethods.inference_skill_version_upload));
    }

    public string ListSkills()
    {
        using var engine = Borrow();
        var status = NativeMethods.inference_skills_list(engine.Handle, out var response);
        return Text(status, response, nameof(NativeMethods.inference_skills_list));
    }

    public string ListSkillVersions(string skillId)
    {
        using var engine = Borrow();
        using var id = new PinnedBytes(skillId);
        var status = NativeMethods.inference_skill_versions_list(engine.Handle, id.Pointer, id.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_skill_versions_list));
    }

    /// <summary>Host, device and build information; needs no engine.</summary>
    public static string SystemInfo()
    {
        NativeMethods.EnsureAbi();
        var status = NativeMethods.inference_system_info(out var response);
        return Text(status, response, nameof(NativeMethods.inference_system_info));
    }

    /// <summary>Environment diagnostics; needs no engine.</summary>
    public static string SystemDoctor()
    {
        NativeMethods.EnsureAbi();
        var status = NativeMethods.inference_system_doctor(out var response);
        return Text(status, response, nameof(NativeMethods.inference_system_doctor));
    }

    /// <summary>The quantization and settings that fit a model on this machine, found without loading it.</summary>
    public static string TuneModel(string requestJson)
    {
        NativeMethods.EnsureAbi();
        using var request = new PinnedBytes(requestJson);
        var status = NativeMethods.inference_model_tune(request.Pointer, request.Length, out var response);
        return Text(status, response, nameof(NativeMethods.inference_model_tune));
    }

    // Dispose ends the engine for its callers even while open streams keep the native engine alive.
    private volatile bool _disposed;

    public void Dispose()
    {
        _disposed = true;
        _handle.Dispose();
    }

    private Lease Borrow()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new(_handle);
    }

    private static string Text(InferenceStatus status, IntPtr response, string operation)
    {
        InferenceException.ThrowIfFailed(status, operation);
        return Owned.TakeString(response);
    }

    private EngineStream Stream(InferenceStatus status, IntPtr stream, string operation)
    {
        InferenceException.ThrowIfFailed(status, operation);
        return new EngineStream(new StreamHandle(stream, _handle));
    }

    private static NativeArray<NativeMedia> Media(IReadOnlyList<MediaAttachment>? media)
    {
        media ??= [];
        var native = new NativeArray<NativeMedia>(media.Count);
        try
        {
            for (var index = 0; index < media.Count; index++)
            {
                native.Pointer[index] = new NativeMedia
                {
                    Data = native.Bytes(media[index].Data),
                    Len = (nuint)media[index].Data.Length,
                    MimeType = native.Text(media[index].MimeType),
                };
            }
            return native;
        }
        catch
        {
            native.Dispose();
            throw;
        }
    }

    private static NativeArray<NativeSkillFile> Skill(IReadOnlyList<SkillFile> files)
    {
        var native = new NativeArray<NativeSkillFile>(files.Count);
        try
        {
            for (var index = 0; index < files.Count; index++)
            {
                native.Pointer[index] = new NativeSkillFile
                {
                    Path = native.Text(files[index].Path),
                    Data = native.Bytes(files[index].Data),
                    Len = (nuint)files[index].Data.Length,
                };
            }
            return native;
        }
        catch
        {
            native.Dispose();
            throw;
        }
    }
}
