using System.Text.Json;
using System.Text.Json.Serialization;

namespace InferenceRs.AvaloniaDemo.Services;

/// <summary>A slot's model and options, remembered between sessions.</summary>
public sealed class SlotState
{
    public string Source { get; set; } = "";

    public string Device { get; set; } = Host.AutoDevice;

    public decimal ContextLength { get; set; }

    public string MmprojPath { get; set; } = "";
}

/// <summary>What the demo remembers: load and sampling defaults, model folders, Hugging Face access and the session.</summary>
public sealed class AppSettings
{
    // a KV cache sized for this much context, not the 90% of free memory an unsized one takes, so several models fit
    public const decimal DefaultContextLength = 8192;

    public string Device { get; set; } = Host.AutoDevice;

    public decimal ContextLength { get; set; } = DefaultContextLength;

    /// <summary>Speculative decoding with the head built into the checkpoint, for chat models that have one.</summary>
    public bool Mtp { get; set; }

    /// <summary>Draft tokens per step; 0 keeps the model's default.</summary>
    public decimal MtpDrafts { get; set; }

    public decimal Temperature { get; set; } = 0.7m;

    public decimal TopP { get; set; } = 0.95m;

    /// <summary>0 samples from the whole vocabulary.</summary>
    public decimal TopK { get; set; }

    public decimal MinP { get; set; }

    public decimal MaxTokens { get; set; } = 1024;

    public bool EnableThinking { get; set; } = true;

    public string SystemPrompt { get; set; } = "";

    /// <summary>Folders the model manager searches.</summary>
    public List<string> ModelFolders { get; set; } = [];

    /// <summary>The Hugging Face cache to use; empty takes the environment's.</summary>
    public string HfCachePath { get; set; } = "";

    /// <summary>A token for gated repositories; empty uses the one <c>hf auth login</c> saved.</summary>
    public string HfToken { get; set; } = "";

    public bool RestoreSession { get; set; } = true;

    /// <summary>Loads the restored models at startup, not only fills in their slots.</summary>
    public bool LoadOnStartup { get; set; }

    /// <summary>Each slot's last model, by the slot's title.</summary>
    public Dictionary<string, SlotState> Slots { get; set; } = [];

    /// <summary>The token source a spec or download request names, when a token is set.</summary>
    [JsonIgnore]
    public string? TokenSource => HfToken.Trim() is { Length: > 0 } token ? $"literal:{token}" : null;

    [JsonIgnore]
    public string? HfCacheOrNull => HfCachePath.Trim() is { Length: > 0 } path ? path : null;
}

/// <summary>Reads and writes the settings file in the user's config directory.</summary>
public static class SettingsStore
{
    private const string FolderName = "inference-rs-demo";
    private const string FileName = "settings.json";
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    // without Create, a config directory that does not exist yet comes back empty and the path turns relative
    public static string FilePath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData, Environment.SpecialFolderOption.Create),
        FolderName,
        FileName);

    /// <summary>The settings in force: those the file held at startup, or the last ones saved.</summary>
    public static AppSettings Current { get; private set; } = Load();

    private static AppSettings Load()
    {
        try
        {
            return File.Exists(FilePath)
                ? JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(FilePath), Options) ?? new AppSettings()
                : new AppSettings();
        }
        catch (Exception error) when (error is IOException or JsonException or UnauthorizedAccessException)
        {
            // an unreadable file falls back to the defaults rather than keeping the window from opening
            return new AppSettings();
        }
    }

    public static void Save(AppSettings settings)
    {
        Current = settings;
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        File.WriteAllText(FilePath, JsonSerializer.Serialize(settings, Options));
        // it can hold a Hugging Face token
        if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(FilePath, UnixFileMode.UserRead | UnixFileMode.UserWrite);
    }

    /// <summary>A deep copy to edit, so a cancelled settings window changes nothing.</summary>
    public static AppSettings Copy(AppSettings settings) =>
        JsonSerializer.Deserialize<AppSettings>(JsonSerializer.Serialize(settings, Options), Options)!;
}
