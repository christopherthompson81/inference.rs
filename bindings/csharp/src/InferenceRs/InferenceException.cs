using System.Text.Json;
using InferenceRs.Native;

namespace InferenceRs;

/// <summary>A native call returned a non-OK status.</summary>
public sealed class InferenceException : Exception
{
    internal InferenceException(InferenceStatus status, string detail, string operation)
        : base(detail.Length > 0 ? $"{operation}: {detail} ({Describe(status)})" : $"{operation}: {Describe(status)}")
    {
        Status = status;
        Detail = detail;
        Operation = operation;
    }

    /// <summary>The status the native call returned.</summary>
    public InferenceStatus Status { get; }

    /// <summary>The native detail: for engine calls, the protocol's error JSON; empty when the call left none.</summary>
    public string Detail { get; }

    /// <summary>The ABI entry point that failed.</summary>
    public string Operation { get; }

    /// <summary>The error JSON's <c>code</c> (OpenAI envelope) or <c>type</c> (Anthropic envelope), if it has one.</summary>
    public string? Code
    {
        get
        {
            try
            {
                using var document = JsonDocument.Parse(Detail);
                var root = document.RootElement;
                if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("error", out var error)) return null;
                if (error.ValueKind != JsonValueKind.Object) return null;
                if (error.TryGetProperty("code", out var code) && code.ValueKind == JsonValueKind.String)
                    return code.GetString();
                return error.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String
                    ? type.GetString()
                    : null;
            }
            catch (JsonException)
            {
                return null;
            }
        }
    }

    private static string Describe(InferenceStatus status) =>
        Utf8.ToString(NativeMethods.inference_status_string(status));

    /// <summary>Throws unless OK, reading the thread-local detail on the thread that made the call.</summary>
    internal static void ThrowIfFailed(InferenceStatus status, string operation)
    {
        if (status == InferenceStatus.Ok) return;
        throw new InferenceException(status, Utf8.ToString(NativeMethods.inference_last_error()), operation);
    }
}
