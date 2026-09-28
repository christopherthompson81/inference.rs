using System.Runtime.InteropServices;
using System.Text;

namespace InferenceRs.Native;

/// <summary>UTF-8 conversions at the ABI boundary.</summary>
internal static unsafe class Utf8
{
    /// <summary>Copies a borrowed NUL-terminated string; the ABI owns every <c>const char *</c> it returns.</summary>
    internal static string ToString(IntPtr value) =>
        value == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(value) ?? string.Empty;

    /// <summary>Decodes a pointer and length the ABI passed in, such as a callback's arguments.</summary>
    internal static string ToString(byte* data, nuint len) =>
        data == null ? string.Empty : Encoding.UTF8.GetString(data, checked((int)len));

    /// <summary>A NUL-terminated copy for a struct field, released with <see cref="Free"/> after the call.</summary>
    internal static IntPtr Allocate(string? value)
    {
        if (value is null) return IntPtr.Zero;
        var bytes = Encoding.UTF8.GetBytes(value);
        var buffer = Marshal.AllocHGlobal(bytes.Length + 1);
        Marshal.Copy(bytes, 0, buffer, bytes.Length);
        Marshal.WriteByte(buffer, bytes.Length, 0);
        return buffer;
    }

    internal static void Free(IntPtr value)
    {
        if (value != IntPtr.Zero) Marshal.FreeHGlobal(value);
    }
}
