using System.Runtime.InteropServices;

namespace InferenceRs.Native;

/// <summary>One-to-one P/Invoke declarations for <c>inference.h</c>; ownership and errors belong to the wrappers.</summary>
/// <remarks>Returned <c>const char *</c> values are borrowed, so they are <see cref="IntPtr"/>, never marshalled strings.</remarks>
internal static unsafe partial class NativeMethods
{
    internal const string Library = "inference_ffi";

    /// <summary>The ABI these declarations mirror; while it is 0.0.x any other version may differ anywhere.</summary>
    internal const uint AbiVersion = (0 << 16) | (0 << 8) | 20;

    /// <summary>Refuses a library built for another ABI, before any call into it could misread its memory.</summary>
    internal static void EnsureAbi()
    {
        var actual = inference_abi_version();
        if (actual == AbiVersion) return;
        throw new InvalidOperationException(
            $"libinference_ffi implements ABI {Describe(actual)}; these bindings need {Describe(AbiVersion)}");
    }

    private static string Describe(uint version) => $"{version >> 16}.{(version >> 8) & 0xff}.{version & 0xff}";

    static NativeMethods() => NativeLibraryResolver.Install();

    // Versioning and errors.

    [LibraryImport(Library)]
    internal static partial uint inference_abi_version();

    [LibraryImport(Library)]
    internal static partial IntPtr inference_build_version();

    [LibraryImport(Library)]
    internal static partial IntPtr inference_last_error();

    [LibraryImport(Library)]
    internal static partial IntPtr inference_status_string(InferenceStatus status);

    // Layout detection.

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial InferenceStatus inference_layout_model_load(
        string modelDir, NativeBackendConfig* backend, out IntPtr outModel);

    [LibraryImport(Library)]
    internal static partial void inference_layout_model_free(IntPtr model);

    [LibraryImport(Library)]
    internal static partial nuint inference_layout_model_label_count(IntPtr model);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_layout_model_label(IntPtr model, nuint index, out IntPtr outLabel);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_layout_detect(
        IntPtr model, NativeImage* image, float threshold, out IntPtr outResult);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_layout_detect_batch(
        IntPtr model, NativeImage* images, nuint count, float threshold, IntPtr* outResults);

    [LibraryImport(Library)]
    internal static partial void inference_layout_result_free(IntPtr result);

    [LibraryImport(Library)]
    internal static partial nuint inference_layout_result_count(IntPtr result);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_layout_result_detection(
        IntPtr result, nuint index, out int outClassId, out IntPtr outLabel, out float outScore, float* outBbox);

    // Engine lifetime and host callbacks.

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_engine_load(byte* spec, nuint specLen, out IntPtr outEngine);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_engine_load_with_callbacks(
        byte* spec, nuint specLen, NativeHostCallbacks* callbacks, out IntPtr outEngine);

    [LibraryImport(Library)]
    internal static partial void inference_engine_free(IntPtr engine);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_engine_for_owner(
        IntPtr engine, byte* owner, nuint ownerLen, out IntPtr outEngine);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_engine_register_logits_processor(
        IntPtr engine, byte* name, nuint nameLen,
        delegate* unmanaged[Cdecl]<IntPtr, float*, nuint, uint*, nuint, int> callback, nint userData);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_engine_unregister_logits_processor(
        IntPtr engine, byte* name, nuint nameLen);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_engine_register_tool(IntPtr engine, NativeHostTool* tool);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_engine_unregister_tool(IntPtr engine, byte* name, nuint nameLen);

    [LibraryImport(Library)]
    internal static partial void inference_callback_result_set(IntPtr result, byte* data, nuint len);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void inference_callback_result_fail(IntPtr result, string? message);

    // Requests: engine, request JSON in, an owned string or stream out.

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_chat(IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_chat_with_media(
        IntPtr engine, byte* request, nuint requestLen, NativeMedia* media, nuint mediaCount, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_chat_stream_open(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outStream);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_chat_stream_open_with_media(
        IntPtr engine, byte* request, nuint requestLen, NativeMedia* media, nuint mediaCount, out IntPtr outStream);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_completion(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_completion_stream_open(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outStream);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_embeddings(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_anthropic_messages(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_anthropic_messages_stream_open(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outStream);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_responses_create(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_responses_stream_open(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outStream);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_responses_get(IntPtr engine, byte* id, nuint idLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_responses_delete(IntPtr engine, byte* id, nuint idLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_responses_cancel(IntPtr engine, byte* id, nuint idLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_models_list(IntPtr engine, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_add(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_remove(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_set_default(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_alias(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_served(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_mcp_tools_list(IntPtr engine, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_anthropic_count_tokens(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_unload(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_reload(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_status(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_lora_adapters_list(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_lora_adapter_load(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_lora_adapter_unload(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_image_generation(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_speech_generation(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outBlob);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_approval_resolve(
        IntPtr engine, byte* approvalId, nuint approvalIdLen, byte* request, nuint requestLen, out IntPtr outResponse);

    // Requantization, online calibration, sessions and tokenization.

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_re_isq(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_calibration_start(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_calibration_status(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_models_cache_stats(IntPtr engine, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_calibration_apply(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_sessions_list(IntPtr engine, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_session_get(IntPtr engine, byte* id, nuint idLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_session_put(
        IntPtr engine, byte* id, nuint idLen, byte* session, nuint sessionLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_session_fork(
        IntPtr engine, byte* id, nuint idLen, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_session_delete(
        IntPtr engine, byte* id, nuint idLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_tokenize(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_prompt_logits(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse, out IntPtr outBlob);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_detokenize(
        IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_tokenize_chat(IntPtr engine, byte* request, nuint requestLen, out IntPtr outResponse);

    // Files, skills and system reports.

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial InferenceStatus inference_file_upload(
        IntPtr engine, byte* data, nuint len, string filename, string? mimeType, string purpose, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_files_list(IntPtr engine, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_file_get(IntPtr engine, byte* id, nuint idLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_file_delete(IntPtr engine, byte* id, nuint idLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_file_content(IntPtr engine, byte* id, nuint idLen, out IntPtr outBlob);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_container_files_list(
        IntPtr engine, byte* containerId, nuint containerIdLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_container_file_get(
        IntPtr engine, byte* containerId, nuint containerIdLen, byte* fileId, nuint fileIdLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_container_file_content(
        IntPtr engine, byte* containerId, nuint containerIdLen, byte* fileId, nuint fileIdLen, out IntPtr outBlob);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_skill_upload(
        IntPtr engine, NativeSkillFile* files, nuint fileCount, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_skill_version_upload(
        IntPtr engine, byte* skillId, nuint skillIdLen, NativeSkillFile* files, nuint fileCount, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_skills_list(IntPtr engine, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_skill_versions_list(
        IntPtr engine, byte* skillId, nuint skillIdLen, out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_system_info(out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_system_doctor(out IntPtr outResponse);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_model_tune(byte* request, nuint requestLen, out IntPtr outResponse);

    // Streams and owned results.

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_stream_next(IntPtr stream, long timeoutMs, out IntPtr outEvent, out int outDone);

    [LibraryImport(Library)]
    internal static partial InferenceStatus inference_stream_cancel(IntPtr stream);

    [LibraryImport(Library)]
    internal static partial void inference_stream_free(IntPtr stream);

    [LibraryImport(Library)]
    internal static partial IntPtr inference_blob_data(IntPtr blob);

    [LibraryImport(Library)]
    internal static partial nuint inference_blob_len(IntPtr blob);

    [LibraryImport(Library)]
    internal static partial IntPtr inference_blob_mime_type(IntPtr blob);

    [LibraryImport(Library)]
    internal static partial void inference_blob_free(IntPtr blob);

    [LibraryImport(Library)]
    internal static partial IntPtr inference_string_data(IntPtr value);

    [LibraryImport(Library)]
    internal static partial nuint inference_string_len(IntPtr value);

    [LibraryImport(Library)]
    internal static partial void inference_string_free(IntPtr value);
}
