using System.Text.Json;
using System.Text.Json.Nodes;

namespace InferenceRs.AvaloniaDemo.Services;

/// <summary>What the native library reports about this machine, read once: its devices and the full report.</summary>
public static class Host
{
    public const string AutoDevice = "auto";
    public const string CpuDevice = "cpu";
    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    static Host()
    {
        try
        {
            var report = JsonNode.Parse(InferenceEngine.SystemInfo())!;
            Report = report.ToJsonString(Indented);
            Accelerators = report["devices"]!.AsArray()
                .Select(device => (Kind: (string?)device!["kind"], Ordinal: (int?)device["ordinal"]))
                .Where(device => device.Kind is not null and not CpuDevice && device.Ordinal is not null)
                .Select(device => $"{device.Kind}:{device.Ordinal}")
                .ToList();
        }
        catch (Exception error)
        {
            // a missing or mismatched native library: the window still opens and says so
            Error = error.Message;
            Report = "";
            Accelerators = [];
        }
    }

    /// <summary>Why the native library could not be used, when it could not.</summary>
    public static string? Error { get; }

    public static string Report { get; }

    /// <summary>The GPUs the library can run on, as engine specs name them (<c>cuda:0</c>, <c>metal:0</c>).</summary>
    public static IReadOnlyList<string> Accelerators { get; }

    /// <summary>The devices a slot offers: the engine's own pick, each GPU, then the CPU.</summary>
    public static IReadOnlyList<string> DeviceOptions => [AutoDevice, .. Accelerators, CpuDevice];
}
