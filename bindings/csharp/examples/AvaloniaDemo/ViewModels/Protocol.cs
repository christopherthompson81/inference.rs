using System.Diagnostics;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>Names the requests share, and how the tabs report failures.</summary>
public static class Protocol
{
    /// <summary>The model a request goes to: an engine's only model.</summary>
    public const string DefaultModel = "default";

    /// <summary>A request names its attached buffers by position.</summary>
    public static string MediaUrl(int index) => $"media://{index}";

    /// <summary>A failure as a status line: the ABI's status and message, or any other exception's message.</summary>
    public static string Describe(Exception error) => error is InferenceException inference
        ? $"{inference.Status}: {inference.Message}"
        : error.Message;

    public static string Seconds(Stopwatch clock) => $"{clock.Elapsed.TotalSeconds:0.0} s";
}
