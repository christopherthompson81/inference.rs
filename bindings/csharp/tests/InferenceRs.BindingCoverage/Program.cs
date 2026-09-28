using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace InferenceRs.BindingCoverage;

/// <summary>
/// Checks every entry point inference.h declares is bound in NativeMethods, and nothing is bound that it does not
/// declare. The Rust header test checks the library against the header; this checks the bindings against it too.
/// Exit codes follow CTest: 0 pass, 1 fail, 77 skip.
/// </summary>
internal static class Program
{
    private const string HeaderVariable = "INFERENCE_HEADER";
    private const string HeaderPath = "crates/inference-ffi/include/inference.h";

    private static int Main(string[] args)
    {
        var header = args.Length > 0 ? args[0] : Environment.GetEnvironmentVariable(HeaderVariable) ?? FindHeader();
        if (header is null)
        {
            Console.WriteLine($"inference.h not found; set {HeaderVariable} or pass its path. Skipping.");
            return 77;
        }
        Console.WriteLine($"header: {header}");

        var declared = DeclaredEntryPoints(File.ReadAllText(header));
        var bound = BoundEntryPoints();
        var missing = declared.Keys.Except(bound.Keys).Order(StringComparer.Ordinal).ToList();
        var extra = bound.Keys.Except(declared.Keys).Order(StringComparer.Ordinal).ToList();
        foreach (var name in missing) Console.Error.WriteLine($"  NOT BOUND: {name}");
        foreach (var name in extra) Console.Error.WriteLine($"  NOT IN HEADER: {name}");
        var arity = declared.Where(entry => bound.TryGetValue(entry.Key, out var count) && count != entry.Value)
            .Select(entry => $"{entry.Key}: header {entry.Value}, binding {bound[entry.Key]}")
            .ToList();
        foreach (var mismatch in arity) Console.Error.WriteLine($"  PARAMETER COUNT: {mismatch}");
        var unexported = Unexported(bound.Keys);
        foreach (var name in unexported) Console.Error.WriteLine($"  NOT EXPORTED BY THE LIBRARY: {name}");
        if (missing.Count > 0 || extra.Count > 0 || arity.Count > 0 || unexported.Count > 0) return 1;

        var headerAbi = HeaderAbiVersion(File.ReadAllText(header));
        var boundAbi = (uint)NativeAbi.GetValue(null)!;
        if (headerAbi != boundAbi)
        {
            Console.Error.WriteLine($"  the header is ABI {headerAbi:x6}, the bindings expect {boundAbi:x6}");
            return 1;
        }

        Console.WriteLine($"every declared entry point is bound and exported ({declared.Count}), at the header's ABI version");
        return 0;
    }

    private static FieldInfo NativeAbi =>
        NativeMethodsType.GetField("AbiVersion", BindingFlags.Static | BindingFlags.NonPublic)!;

    private static uint HeaderAbiVersion(string header)
    {
        uint Part(string name) =>
            uint.Parse(Regex.Match(header, $@"#define INFERENCE_ABI_VERSION_{name} (\d+)").Groups[1].Value);
        return (Part("MAJOR") << 16) | (Part("MINOR") << 8) | Part("PATCH");
    }

    private static string? FindHeader()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var path = Path.Combine(dir.FullName, HeaderPath);
            if (File.Exists(path)) return path;
        }
        return null;
    }

    /// <summary>Entry points after INFERENCE_API and their parameter counts; comments are stripped first.</summary>
    private static Dictionary<string, int> DeclaredEntryPoints(string header)
    {
        var code = Regex.Replace(header, @"/\*.*?\*/", "", RegexOptions.Singleline);
        return Regex.Matches(code, @"INFERENCE_API\s+[A-Za-z_][A-Za-z0-9_ \*]*?\b(inference_[a-z0-9_]+)\s*\(([^)]*)\)")
            .ToDictionary(
                match => match.Groups[1].Value,
                match => match.Groups[2].Value.Trim() is "void" or "" ? 0 : match.Groups[2].Value.Split(',').Length,
                StringComparer.Ordinal);
    }

    /// <summary>Methods of the compiled NativeMethods that import from the library, and their parameter counts.</summary>
    private static Dictionary<string, int> BoundEntryPoints() =>
        NativeMethodsType
            .GetMethods(BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
            .Where(method => method.Name.StartsWith("inference_", StringComparison.Ordinal))
            .Where(method => method.GetCustomAttribute<LibraryImportAttribute>() is not null)
            .ToDictionary(method => method.Name, method => method.GetParameters().Length, StringComparer.Ordinal);

    /// <summary>Bound names the built library does not export; none are checked when the library is not built.</summary>
    private static List<string> Unexported(IEnumerable<string> bound)
    {
        if (!InferenceRs.Native.NativeLibraryResolver.TryLoadFromSearchDirectories(out var handle))
        {
            Console.WriteLine("libinference_ffi not found; skipping the export check.");
            return [];
        }
        return bound.Where(name => !NativeLibrary.TryGetExport(handle, name, out _)).Order(StringComparer.Ordinal).ToList();
    }

    private static Type NativeMethodsType =>
        typeof(InferenceEngine).Assembly.GetType("InferenceRs.Native.NativeMethods", throwOnError: true)!;
}
