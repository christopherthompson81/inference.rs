namespace InferenceRs;

/// <summary>A CUDA device the engine holds for GPU work of the caller's own: nothing loads there until it is
/// disposed.</summary>
public sealed class DeviceLease : IDisposable
{
    private readonly DeviceLeaseHandle _lease;

    internal DeviceLease(DeviceLeaseHandle lease) => _lease = lease;

    /// <summary>Releases the device.</summary>
    public void Dispose() => _lease.Dispose();
}
