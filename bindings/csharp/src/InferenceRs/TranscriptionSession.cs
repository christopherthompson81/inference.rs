using System.Text.Json;
using InferenceRs.Native;

namespace InferenceRs;

/// <summary>A streamed transcription on a cache-aware streaming recogniser: push 16-bit mono PCM as it arrives, then
/// finish for the whole transcript. Disposing an unfinished session closes it.</summary>
/// <remarks>Calls are serialized, so threads may share a session. An open session keeps its engine loaded.</remarks>
public sealed unsafe class TranscriptionSession : IDisposable
{
    private readonly TranscriptionSessionHandle _session;
    // the native session takes one call at a time; an audio thread and a caller may share it
    private readonly object _gate = new();

    internal TranscriptionSession(TranscriptionSessionHandle session) => _session = session;

    /// <summary>Pushes little-endian 16-bit mono PCM at the session's rate.</summary>
    /// <returns>The text the audio so far completes; often empty for a short piece.</returns>
    public string Push(byte[] pcm16)
    {
        lock (_gate)
        {
            return PushLocked(pcm16);
        }
    }

    private string PushLocked(byte[] pcm16)
    {
        using var session = new Lease(_session);
        using var audio = new PinnedBytes(pcm16);
        var status = NativeMethods.inference_transcription_session_push(
            session.Handle, audio.Pointer, audio.Length, out var response);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_transcription_session_push));
        return Delta(Owned.TakeString(response));
    }

    /// <summary>Ends the session: the text its last audio adds, and the whole transcript in its response format.</summary>
    public (string Delta, Blob Transcript) Finish()
    {
        lock (_gate)
        {
            return FinishLocked();
        }
    }

    private (string Delta, Blob Transcript) FinishLocked()
    {
        using var session = new Lease(_session);
        var status = NativeMethods.inference_transcription_session_finish(session.Handle, out var delta, out var blob);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_transcription_session_finish));
        return (Delta(Owned.TakeString(delta)), Owned.TakeBlob(blob));
    }

    public void Dispose() => _session.Dispose();

    private static string Delta(string json)
    {
        using var document = JsonDocument.Parse(json);
        return document.RootElement.GetProperty("delta").GetString() ?? string.Empty;
    }
}
