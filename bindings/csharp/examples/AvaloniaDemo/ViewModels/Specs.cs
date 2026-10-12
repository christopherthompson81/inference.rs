using System.Text.Json.Nodes;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>The loads each tab runs, built from a slot's settings on the UI thread.</summary>
public static class Specs
{
    /// <summary>A chat model: auto-detected from a Hugging Face id or directory, or a GGUF file with its mmproj.</summary>
    public static SlotLoader Chat => slot =>
    {
        var source = slot.Source.Trim();
        if (!slot.IsGguf) return Engine(new JsonObject { ["Run"] = new JsonObject { ["model_id"] = source } }, slot);
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
        return Engine(new JsonObject { ["GGUF"] = gguf }, slot);
    };

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

    private static Func<Action<LoadProgress>, IDisposable> Engine(JsonObject model, ModelSlotViewModel slot)
    {
        var runtime = new JsonObject();
        if (slot.DeviceOrNull is { } device) runtime["device"] = device;
        if (slot.HasContext) runtime["paged_cache"] = new JsonObject { ["context_len"] = (int)slot.ContextLength };
        var spec = new JsonObject { ["model"] = model, ["runtime"] = runtime }.ToJsonString();
        return progress => InferenceEngine.Load(spec, new HostCallbacks { Progress = progress });
    }
}
