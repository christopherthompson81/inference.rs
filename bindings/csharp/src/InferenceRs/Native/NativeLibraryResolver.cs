using System.Runtime.InteropServices;

namespace InferenceRs.Native;

/// <summary>Finds libinference_ffi, which cargo builds into target/ rather than next to the managed assembly.</summary>
internal static class NativeLibraryResolver
{
    private const string DirectoryVariable = "INFERENCE_NATIVE_DIR";

    // Release first: a consumer that built it has the fast one; tests fall back to the dev build they just made.
    private static readonly string[] Profiles = ["release", "debug"];

    /// <summary>Runs from <see cref="NativeMethods"/>'s static constructor, on first interop use.</summary>
    internal static void Install() =>
        NativeLibrary.SetDllImportResolver(typeof(NativeLibraryResolver).Assembly, Resolve);

    private static IntPtr Resolve(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != NativeMethods.Library) return IntPtr.Zero;
        if (TryLoadFromSearchDirectories(out var loaded)) return loaded;
        return NativeLibrary.TryLoad(libraryName, assembly, searchPath, out var found) ? found : IntPtr.Zero;
    }

    /// <summary>The library from INFERENCE_NATIVE_DIR or the checkout's target/, without the platform search.</summary>
    internal static bool TryLoadFromSearchDirectories(out IntPtr loaded)
    {
        foreach (var directory in SearchDirectories())
        {
            var path = Path.Combine(directory, FileName());
            if (File.Exists(path) && NativeLibrary.TryLoad(path, out loaded)) return true;
        }
        loaded = IntPtr.Zero;
        return false;
    }

    /// <summary>INFERENCE_NATIVE_DIR, then the target/ of the checkout these bindings sit in.</summary>
    private static IEnumerable<string> SearchDirectories()
    {
        var configured = Environment.GetEnvironmentVariable(DirectoryVariable);
        if (!string.IsNullOrEmpty(configured)) yield return configured;

        // Walking up is what finds the checkout's target/ from a test binary several directories deep.
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            foreach (var profile in Profiles) yield return Path.Combine(dir.FullName, "target", profile);
        }
    }

    private static string FileName()
    {
        if (OperatingSystem.IsWindows()) return "inference_ffi.dll";
        if (OperatingSystem.IsMacOS()) return "libinference_ffi.dylib";
        return "libinference_ffi.so";
    }
}
