using System.Runtime.InteropServices;

namespace InferenceRs.Native;

/// <summary>One-to-one P/Invoke declarations for <c>inference.h</c>, generated into NativeMethods.g.cs; ownership and errors belong to the wrappers.</summary>
/// <remarks>Returned <c>const char *</c> values are borrowed, so they are <see cref="IntPtr"/>, never marshalled strings.</remarks>
internal static unsafe partial class NativeMethods
{
    internal const string Library = "inference_ffi";

    /// <summary>Refuses a library built for another ABI, before any call into it could misread its memory.</summary>
    internal static void EnsureAbi()
    {
        var actual = inference_abi_version();
        if (actual == AbiVersion) return;
        throw new InvalidOperationException(
            $"libinference_ffi implements ABI {Describe(actual)}; these bindings need {Describe(AbiVersion)}");
    }

    private static string Describe(uint version) => $"{version >> 16}.{(version >> 8) & 0xff}.{version & 0xff}";

    static NativeMethods() => NativeLibraryResolver.Install();
}
