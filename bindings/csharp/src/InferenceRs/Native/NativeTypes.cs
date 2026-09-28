using System.Runtime.InteropServices;

namespace InferenceRs.Native;

/// <summary>Mirrors <c>inference_status</c>.</summary>
public enum InferenceStatus
{
    Ok = 0,
    InvalidArgument = 1,
    LoadFailed = 2,
    Runtime = 3,
    OutOfRange = 4,
    NotAvailable = 5,
    Internal = 6,
    InvalidRequest = 7,
    Unavailable = 8,
    NotFound = 9,
}

/// <summary>Mirrors <c>inference_pixel_format</c>.</summary>
public enum PixelFormat
{
    Rgb8 = 0,
    Bgr8 = 1,
    Rgba8 = 2,
    Bgra8 = 3,
    Gray8 = 4,
}

/// <summary>Mirrors <c>inference_backend_config</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeBackendConfig
{
    public IntPtr Backend;
    public int Device;
    public int Threads;
}

/// <summary>Mirrors <c>inference_image</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeImage
{
    public byte* Pixels;
    public uint Width;
    public uint Height;
    public uint Stride;
    public int Format;
}

/// <summary>Mirrors <c>inference_media</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeMedia
{
    public byte* Data;
    public nuint Len;
    public IntPtr MimeType;
}

/// <summary>Mirrors <c>inference_skill_file</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSkillFile
{
    public IntPtr Path;
    public byte* Data;
    public nuint Len;
}

/// <summary>Mirrors <c>inference_host_tool</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeHostTool
{
    public byte* Definition;
    public nuint DefinitionLen;
    public delegate* unmanaged[Cdecl]<IntPtr, byte*, byte*, nuint, byte*, nuint, IntPtr, void> Callback;
    public IntPtr UserData;
}

/// <summary>Mirrors <c>inference_host_callbacks</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeHostCallbacks
{
    public NativeHostTool* Tools;
    public nuint ToolCount;
    public delegate* unmanaged[Cdecl]<IntPtr, byte*, nuint, IntPtr, void> Search;
    public IntPtr SearchUserData;
}
