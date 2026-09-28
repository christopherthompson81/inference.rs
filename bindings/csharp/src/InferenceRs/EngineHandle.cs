using System.Runtime.InteropServices;
using InferenceRs.Native;

namespace InferenceRs;

/// <summary>The native engine and the ids of its host callbacks.</summary>
/// <remarks>Calls and open streams add references, so the engine is freed only after the last of them ends.</remarks>
internal sealed class EngineHandle : SafeHandle
{
    private readonly nint[] _callbacks;

    internal EngineHandle(IntPtr engine, nint[] callbacks)
        : base(IntPtr.Zero, ownsHandle: true)
    {
        SetHandle(engine);
        _callbacks = callbacks;
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.inference_engine_free(handle);
        HostCallbackRegistry.Remove(_callbacks);
        return true;
    }
}

/// <summary>A native stream holding a reference on its engine, released when the stream is freed.</summary>
internal sealed class StreamHandle : SafeHandle
{
    private readonly EngineHandle _engine;

    /// <summary>Takes a new reference on <paramref name="engine"/>; frees <paramref name="stream"/> if it cannot.</summary>
    internal StreamHandle(IntPtr stream, EngineHandle engine)
        : base(IntPtr.Zero, ownsHandle: true)
    {
        var added = false;
        try
        {
            engine.DangerousAddRef(ref added);
        }
        finally
        {
            // Disposed meanwhile: the stream must not outlive the engine's callback registrations.
            if (!added) NativeMethods.inference_stream_free(stream);
        }
        SetHandle(stream);
        _engine = engine;
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.inference_stream_free(handle);
        _engine.DangerousRelease();
        return true;
    }
}

/// <summary>A native layout model.</summary>
internal sealed class LayoutHandle : SafeHandle
{
    internal LayoutHandle(IntPtr model)
        : base(IntPtr.Zero, ownsHandle: true) => SetHandle(model);

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.inference_layout_model_free(handle);
        return true;
    }
}

/// <summary>Keeps a handle alive for one call, even if another thread disposes it meanwhile.</summary>
internal readonly struct Lease : IDisposable
{
    private readonly SafeHandle _handle;

    internal Lease(SafeHandle handle)
    {
        var added = false;
        handle.DangerousAddRef(ref added);
        _handle = handle;
    }

    internal IntPtr Handle => _handle.DangerousGetHandle();

    public void Dispose() => _handle.DangerousRelease();
}
