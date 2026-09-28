using System.Runtime.InteropServices;
using System.Text;

namespace InferenceRs.Native;

/// <summary>Bytes pinned for one call; the pointer is never NULL, even for an empty input, as the ABI requires.</summary>
internal sealed unsafe class PinnedBytes : IDisposable
{
    private static readonly byte[] Empty = new byte[1];

    private GCHandle _pin;

    internal PinnedBytes(string value) : this(Encoding.UTF8.GetBytes(value)) { }

    /// <summary>Pins the caller's array in place rather than copying it.</summary>
    internal PinnedBytes(byte[] value)
    {
        _pin = GCHandle.Alloc(value.Length == 0 ? Empty : value, GCHandleType.Pinned);
        Pointer = (byte*)_pin.AddrOfPinnedObject();
        Length = (nuint)value.Length;
    }

    internal byte* Pointer { get; }

    internal nuint Length { get; }

    public void Dispose()
    {
        if (_pin.IsAllocated) _pin.Free();
    }
}

/// <summary>An array of native structs, with the buffers and strings they point at, alive for one call.</summary>
internal sealed unsafe class NativeArray<T> : IDisposable
    where T : unmanaged
{
    private readonly List<IntPtr> _allocations = [];
    private readonly List<PinnedBytes> _pins = [];

    internal NativeArray(int count)
    {
        Count = (nuint)count;
        if (count == 0) return;
        Pointer = (T*)Marshal.AllocHGlobal(sizeof(T) * count);
        _allocations.Add((IntPtr)Pointer);
    }

    /// <summary>NULL when empty, which the ABI accepts with a count of 0.</summary>
    internal T* Pointer { get; }

    internal nuint Count { get; }

    internal byte* Bytes(byte[] data)
    {
        var pin = new PinnedBytes(data);
        _pins.Add(pin);
        return pin.Pointer;
    }

    internal IntPtr Text(string? value)
    {
        var text = Utf8.Allocate(value);
        _allocations.Add(text);
        return text;
    }

    public void Dispose()
    {
        foreach (var pin in _pins) pin.Dispose();
        foreach (var allocation in _allocations) Utf8.Free(allocation);
        _pins.Clear();
        _allocations.Clear();
    }
}
