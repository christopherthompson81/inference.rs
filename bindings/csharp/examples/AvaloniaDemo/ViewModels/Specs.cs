using System.Text.Json.Nodes;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>The loads each tab runs, built from a slot's settings on the UI thread.</summary>
public static class Specs
{
    /// <summary>A chat model: auto-detected from a Hugging Face id or directory, or a GGUF file with its mmproj.</summary>
    /// <remarks>Uses the settings' MTP choice, for checkpoints with a built-in draft head.</remarks>
    public static SlotLoader Chat => slot => ChatModel(slot, mtp: true);

    /// <summary>A document reader: loaded as a chat model, without speculative decoding.</summary>
    public static SlotLoader Ocr => slot => ChatModel(slot, mtp: false);

    private static Func<Action<LoadProgress>, IDisposable> ChatModel(ModelSlotViewModel slot, bool mtp)
    {
        var settings = SettingsStore.Current;
        var source = slot.Source.Trim();
        if (!slot.IsGguf)
        {
            var run = new JsonObject { ["model_id"] = source };
            if (settings.HfCacheOrNull is { } runCache) run["hf_cache_path"] = runCache;
            return Engine(new JsonObject { ["Run"] = run }, slot, mtp);
        }
        var directory = Path.GetDirectoryName(Path.GetFullPath(source))!;
        var gguf = new JsonObject
        {
            ["quantized_model_id"] = directory,
            ["quantized_filename"] = Path.GetFileName(source),
        };
        if (slot.MmprojPath.Trim() is { Length: > 0 } mmproj)
        {
            // the spec names the projector by file name, inside the GGUF's directory
            if (Path.GetDirectoryName(Path.GetFullPath(mmproj)) != directory)
            {
                throw new ArgumentException($"the mmproj must be in the GGUF's directory, {directory}");
            }
            gguf["mmproj_filename"] = Path.GetFileName(mmproj);
        }
        else
        {
            // a projector beside the GGUF, when there is one
            gguf["mmproj_selection"] = "any";
        }
        if (settings.HfCacheOrNull is { } cache) gguf["hf_cache_path"] = cache;
        return Engine(new JsonObject { ["GGUF"] = gguf }, slot, mtp);
    }

    public static SlotLoader Diffusion(Func<string> arch) => slot => Engine(
        new JsonObject { ["DiffusionPlain"] = new JsonObject { ["model_id"] = slot.Source.Trim(), ["arch"] = arch() } },
        slot);

    public static SlotLoader Speech => slot => Engine(
        new JsonObject { ["Speech"] = new JsonObject { ["model_id"] = slot.Source.Trim() } }, slot);

    public static SlotLoader Transcription => slot => Engine(
        new JsonObject { ["Transcription"] = new JsonObject { ["model_id"] = slot.Source.Trim() } }, slot);

    public static SlotLoader Diarization => slot => Engine(
        new JsonObject { ["Diarization"] = new JsonObject { ["model_id"] = slot.Source.Trim() } }, slot);

    /// <summary>The layout detector, on the slot's device: <c>auto</c> is the first GPU, else the CPU.</summary>
    public static SlotLoader Layout => slot =>
    {
        var path = slot.Source.Trim();
        var device = slot.DeviceOrNull ?? Host.Accelerators.FirstOrDefault() ?? Host.CpuDevice;
        var backend = device.Split(':') is [var kind, var ordinal] ? new Backend(kind, int.Parse(ordinal)) : new Backend(device);
        return _ => LayoutModel.Load(path, backend);
    };

    private static Func<Action<LoadProgress>, IDisposable> Engine(JsonObject model, ModelSlotViewModel slot, bool mtp = false)
    {
        var settings = SettingsStore.Current;
        var runtime = new JsonObject();
        if (slot.DeviceOrNull is { } device) runtime["device"] = device;
        if (slot.HasContext) runtime["paged_cache"] = new JsonObject { ["context_len"] = (int)slot.ContextLength };
        if (settings.TokenSource is { } token) runtime["token_source"] = token;
        if (mtp && settings.Mtp)
        {
            var drafts = new JsonObject();
            if (settings.MtpDrafts > 0) drafts["n_predict"] = (int)settings.MtpDrafts;
            runtime["mtp"] = drafts;
        }
        var spec = new JsonObject { ["model"] = model, ["runtime"] = runtime }.ToJsonString();
        return progress => InferenceEngine.Load(spec, new HostCallbacks { Progress = progress });
    }
}
