using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using InferenceRs.Native;

namespace InferenceRs;

/// <summary>One stream event: its protocol name (<c>chunk</c>, <c>message_start</c>, <c>response.completed</c>, ...) and data.</summary>
/// <remarks>A failed request ends with an <c>error</c> event whose data is the protocol's error JSON.</remarks>
public sealed record StreamEvent(string Name, JsonElement Data)
{
    internal static StreamEvent Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        return new StreamEvent(root.GetProperty("event").GetString() ?? string.Empty, root.GetProperty("data").Clone());
    }
}

/// <summary>A streaming request. Enumerate it for its events; disposing it abandons the request.</summary>
/// <remarks>Use a stream from one thread at a time. An open stream keeps its engine loaded.</remarks>
public sealed class EngineStream : IEnumerable<StreamEvent>, IDisposable
{
    private readonly StreamHandle _stream;

    internal EngineStream(StreamHandle stream) => _stream = stream;

    /// <summary>Waits up to <paramref name="timeout"/> for the next event; null or infinite waits until one arrives.</summary>
    /// <returns>False on a timeout, or once the stream has ended (then <see cref="IsDone"/> is true).</returns>
    public bool TryNext(TimeSpan? timeout, [NotNullWhen(true)] out StreamEvent? streamEvent)
    {
        streamEvent = null;
        if (IsDone) return false;
        var timeoutMs = TimeoutMs(timeout);
        using var stream = new Lease(_stream);
        var status = NativeMethods.inference_stream_next(stream.Handle, timeoutMs, out var eventHandle, out var done);
        InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_stream_next));
        if (done != 0)
        {
            IsDone = true;
            return false;
        }
        if (eventHandle == IntPtr.Zero) return false;
        streamEvent = StreamEvent.Parse(Owned.TakeString(eventHandle));
        return true;
    }

    /// <summary>Whether the stream has ended.</summary>
    public bool IsDone { get; private set; }

    public IEnumerator<StreamEvent> GetEnumerator()
    {
        while (TryNext(null, out var streamEvent)) yield return streamEvent;
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    public void Dispose() => _stream.Dispose();

    // Rounded up, so a sub-millisecond wait still waits rather than polling.
    private static long TimeoutMs(TimeSpan? timeout)
    {
        if (timeout is null || timeout == Timeout.InfiniteTimeSpan) return -1;
        ArgumentOutOfRangeException.ThrowIfLessThan(timeout.Value, TimeSpan.Zero, nameof(timeout));
        return (long)Math.Ceiling(timeout.Value.TotalMilliseconds);
    }
}
