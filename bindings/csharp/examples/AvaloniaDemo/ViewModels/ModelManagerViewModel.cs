using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>The tab slot a model can be put in.</summary>
public enum ModelTarget
{
    Chat,
    Documents,
    Speech,
    Transcription,
    Diarization,
    Images,
}

/// <summary>One model the scan found: what it is, where it is, and which slots take it.</summary>
public sealed class ModelEntryViewModel
{
    private const double BytesPerGb = 1e9;
    private const string HfCacheSource = "hf_cache";

    public ModelEntryViewModel(JsonElement model)
    {
        Path = Text(model, "path") ?? "";
        RepoId = Text(model, "repo_id");
        InHfCache = Text(model, "source") == HfCacheSource;
        Format = Text(model, "format") ?? "";
        Kind = Text(model, "kind");
        Loader = Text(model, "loader");
        Architecture = Text(model, "architecture");
        Quantization = Text(model, "quantization");
        SizeLabel = Text(model, "size_label");
        BaseModel = Text(model, "base_model");
        Error = Text(model, "error");
        ContextLength = model.TryGetProperty("context_length", out var context) && context.ValueKind == JsonValueKind.Number
            ? context.GetInt64()
            : null;
        SizeBytes = model.GetProperty("size_bytes").GetInt64();
        Files = Strings(model, "files");
        Projectors = Strings(model, "projectors");
        var file = System.IO.Path.GetFileName(Path.TrimEnd(System.IO.Path.DirectorySeparatorChar));
        Name = Format == "gguf" || Format == "nemo" ? file : RepoId ?? Text(model, "name") ?? file;
        Targets = Kind switch
        {
            "text" => [ModelTarget.Chat],
            "multimodal" => [ModelTarget.Chat, ModelTarget.Documents],
            "speech" => [ModelTarget.Speech],
            "transcription" => [ModelTarget.Transcription],
            "diarization" => [ModelTarget.Diarization],
            "image_generation" => [ModelTarget.Images],
            _ => [],
        };
    }

    public string Path { get; }

    public string? RepoId { get; }

    public bool InHfCache { get; }

    public string Format { get; }

    public string? Kind { get; }

    public string? Loader { get; }

    public string? Architecture { get; }

    public string? Quantization { get; }

    public string? SizeLabel { get; }

    public string? BaseModel { get; }

    public long? ContextLength { get; }

    public long SizeBytes { get; }

    public IReadOnlyList<string> Files { get; }

    public IReadOnlyList<string> Projectors { get; }

    public string? Error { get; }

    public string Name { get; }

    public IReadOnlyList<ModelTarget> Targets { get; }

    public bool IsSupported => Kind is not null;

    public bool IsProjector => Kind == "projector";

    public string KindLabel => Kind switch
    {
        null => "Unsupported",
        "multimodal" => "Multimodal",
        "image_generation" => "Image generation",
        "voice_activity" => "Voice activity",
        _ => char.ToUpperInvariant(Kind[0]) + Kind[1..],
    };

    public string SizeText => $"{SizeBytes / BytesPerGb:0.0} GB";

    public string Where => InHfCache ? $"HF cache ({RepoId})" : System.IO.Path.GetDirectoryName(Path) ?? "";

    public string Details => string.Join(", ", new[] { Architecture, Loader, SizeLabel, Quantization }.Where(part => part is { Length: > 0 }));

    /// <summary>What a slot loads: the repository id for a cached checkpoint, the file or directory otherwise.</summary>
    public string LoadSource => InHfCache && Format is not ("gguf" or "nemo") && RepoId is { } repo ? repo : Path;

    /// <summary>Cache repositories delete through the bindings; a local GGUF or .nemo is a file to remove.</summary>
    public bool CanDelete => InHfCache ? RepoId is not null : Format is "gguf" or "nemo";

    public bool HasProjectors => Projectors.Count > 0;

    public string ProjectorText => string.Join(Environment.NewLine, Projectors);

    public bool HasError => Error is { Length: > 0 };

    public bool HasBaseModel => BaseModel is { Length: > 0 };

    public bool HasContext => ContextLength is not null;

    public bool Matches(string filter) =>
        filter.Length == 0
        || new[] { Name, Path, Architecture, Quantization, BaseModel, RepoId, KindLabel }
            .Any(text => text?.Contains(filter, StringComparison.OrdinalIgnoreCase) == true);

    private static string? Text(JsonElement model, string name) =>
        model.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static IReadOnlyList<string> Strings(JsonElement model, string name) =>
        model.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Array
            ? [.. value.EnumerateArray().Select(item => item.GetString() ?? "")]
            : [];
}

/// <summary>The models in the settings' folders and the Hugging Face cache: browse, put in a tab, delete, download.</summary>
public sealed partial class ModelManagerViewModel : ObservableObject, IDisposable
{
    private const double BytesPerMb = 1e6;

    private readonly Action<ModelTarget, string, string> _assign;
    private readonly List<ModelEntryViewModel> _all = [];
    private EngineStream? _download;

    /// <param name="assign">Puts a model (its source and mmproj) in the target's slot and shows that tab.</param>
    public ModelManagerViewModel(Action<ModelTarget, string, string> assign) => _assign = assign;

    public ObservableCollection<ModelEntryViewModel> Entries { get; } = [];

    [ObservableProperty]
    public partial string Filter { get; set; } = "";

    partial void OnFilterChanged(string value) => ApplyFilter();

    [ObservableProperty]
    public partial bool ShowUnsupported { get; set; }

    partial void OnShowUnsupportedChanged(bool value) => ApplyFilter();

    [ObservableProperty]
    public partial bool ShowProjectors { get; set; }

    partial void OnShowProjectorsChanged(bool value) => ApplyFilter();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelection))]
    [NotifyCanExecuteChangedFor(nameof(DeleteCommand))]
    [NotifyCanExecuteChangedFor(nameof(UseCommand))]
    public partial ModelEntryViewModel? Selected { get; set; }

    partial void OnSelectedChanged(ModelEntryViewModel? value) => IsConfirmingDelete = false;

    public bool HasSelection => Selected is not null;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RefreshCommand))]
    public partial bool IsScanning { get; set; }

    [ObservableProperty]
    public partial string Status { get; set; } = "";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    public partial string Error { get; set; } = "";

    public bool HasError => Error.Length > 0;

    [ObservableProperty]
    public partial bool IsConfirmingDelete { get; set; }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DownloadCommand))]
    public partial string DownloadRepo { get; set; } = "";

    /// <summary>A GGUF quantization to pick from the repository (<c>Q4_K_M</c>, <c>4</c>); empty takes everything.</summary>
    [ObservableProperty]
    public partial string DownloadQuant { get; set; } = "";

    /// <summary>Particular files, comma separated; empty takes the quant's files or the whole repository.</summary>
    [ObservableProperty]
    public partial string DownloadFiles { get; set; } = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DownloadCommand))]
    [NotifyCanExecuteChangedFor(nameof(CancelDownloadCommand))]
    public partial bool IsDownloading { get; set; }

    [ObservableProperty]
    public partial double DownloadPercent { get; set; }

    [ObservableProperty]
    public partial string DownloadStatus { get; set; } = "";

    private bool CanRefresh() => !IsScanning;

    [RelayCommand(CanExecute = nameof(CanRefresh))]
    private async Task RefreshAsync()
    {
        var settings = SettingsStore.Current;
        var request = new JsonObject
        {
            ["folders"] = new JsonArray([.. settings.ModelFolders.Select(folder => JsonValue.Create(folder))]),
            ["hf_cache"] = true,
        };
        if (settings.HfCacheOrNull is { } cache) request["hf_cache_path"] = cache;
        IsScanning = true;
        Error = "";
        Status = "Scanning";
        var clock = Stopwatch.StartNew();
        try
        {
            var json = await Task.Run(() => InferenceEngine.ScanModels(request.ToJsonString()));
            using var scan = JsonDocument.Parse(json);
            var selected = Selected?.Path;
            _all.Clear();
            _all.AddRange(scan.RootElement.GetProperty("models").EnumerateArray().Select(model => new ModelEntryViewModel(model)));
            ApplyFilter();
            Selected = Entries.FirstOrDefault(entry => entry.Path == selected);
            var problems = scan.RootElement.GetProperty("errors").EnumerateArray()
                .Select(error => $"{error.GetProperty("path").GetString()}: {error.GetProperty("error").GetString()}")
                .ToList();
            Error = string.Join(Environment.NewLine, problems);
            var folders = settings.ModelFolders.Count;
            Status = $"{_all.Count} models in {folders} folder{(folders == 1 ? "" : "s")} and the Hugging Face cache, "
                + $"scanned in {Protocol.Seconds(clock)}";
        }
        catch (Exception error)
        {
            Error = Protocol.Describe(error);
            Status = "";
        }
        finally
        {
            IsScanning = false;
        }
    }

    private void ApplyFilter()
    {
        var filter = Filter.Trim();
        Entries.Clear();
        foreach (var entry in _all)
        {
            if (!ShowUnsupported && !entry.IsSupported) continue;
            if (!ShowProjectors && entry.IsProjector) continue;
            if (entry.Matches(filter)) Entries.Add(entry);
        }
    }

    private bool CanUse(ModelTarget target) => Selected?.Targets.Contains(target) == true;

    [RelayCommand(CanExecute = nameof(CanUse))]
    private void Use(ModelTarget target)
    {
        var entry = Selected!;
        _assign(target, entry.LoadSource, target == ModelTarget.Chat || target == ModelTarget.Documents
            ? entry.Projectors.FirstOrDefault() ?? ""
            : "");
        Status = $"Put {entry.Name} in the {target} tab; load it there.";
    }

    private bool CanDelete() => Selected?.CanDelete == true;

    [RelayCommand(CanExecute = nameof(CanDelete))]
    private void Delete() => IsConfirmingDelete = true;

    [RelayCommand]
    private void KeepModel() => IsConfirmingDelete = false;

    [RelayCommand]
    private async Task ConfirmDeleteAsync()
    {
        var entry = Selected;
        IsConfirmingDelete = false;
        if (entry is null) return;
        try
        {
            if (entry.InHfCache)
            {
                var request = new JsonObject { ["repo_id"] = entry.RepoId };
                if (SettingsStore.Current.HfCacheOrNull is { } cache) request["hf_cache_path"] = cache;
                await Task.Run(() => InferenceEngine.DeleteCachedRepo(request.ToJsonString()));
            }
            else
            {
                foreach (var file in entry.Files.Count > 0 ? entry.Files : [entry.Path]) File.Delete(file);
            }
            Selected = null;
            await RefreshAsync();
        }
        catch (Exception error)
        {
            Error = Protocol.Describe(error);
        }
    }

    private bool CanDownload() => !IsDownloading && DownloadRepo.Trim().Length > 0;

    [RelayCommand(CanExecute = nameof(CanDownload))]
    private async Task DownloadAsync()
    {
        var settings = SettingsStore.Current;
        var request = new JsonObject { ["repo_id"] = DownloadRepo.Trim() };
        if (DownloadQuant.Trim() is { Length: > 0 } quant) request["quant"] = quant;
        var files = DownloadFiles.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (files.Length > 0) request["files"] = new JsonArray([.. files.Select(file => JsonValue.Create(file))]);
        if (settings.TokenSource is { } token) request["token_source"] = token;
        if (settings.HfCacheOrNull is { } cache) request["hf_cache_path"] = cache;
        Error = "";
        DownloadPercent = 0;
        DownloadStatus = "Starting";
        IsDownloading = true;
        try
        {
            _download = InferenceEngine.DownloadModel(request.ToJsonString());
            var outcome = await Task.Run(() => Follow(_download));
            DownloadStatus = outcome;
        }
        catch (Exception error)
        {
            Error = Protocol.Describe(error);
            DownloadStatus = "";
        }
        finally
        {
            _download?.Dispose();
            _download = null;
            IsDownloading = false;
        }
        await RefreshAsync();
    }

    // runs on a worker thread; returns the line to leave in the status
    private string Follow(EngineStream download)
    {
        var count = 0;
        var started = new List<string>();
        foreach (var streamEvent in download)
        {
            switch (streamEvent.Name)
            {
                case "files":
                    count = streamEvent.Data.GetArrayLength();
                    break;
                case "progress":
                    var file = streamEvent.Data.GetProperty("file").GetString() ?? "";
                    if (!started.Contains(file)) started.Add(file);
                    var done = streamEvent.Data.GetProperty("done").GetDouble();
                    var total = streamEvent.Data.GetProperty("total").GetDouble();
                    // each finished file is a whole share of the bar, the current one its fraction
                    var percent = 100.0 * (started.Count - 1 + (total > 0 ? done / total : 1)) / Math.Max(count, 1);
                    var status = $"{file}: {done / BytesPerMb:0} of {total / BytesPerMb:0} MB (file {started.Count} of {count})";
                    Dispatcher.UIThread.Post(() =>
                    {
                        DownloadPercent = percent;
                        DownloadStatus = status;
                    });
                    break;
                case "done":
                    return $"Downloaded {count} file{(count == 1 ? "" : "s")} of {streamEvent.Data.GetProperty("repo_id").GetString()}";
                case "canceled":
                    return "Cancelled; a partly downloaded file resumes next time";
                case "error":
                    var message = streamEvent.Data.TryGetProperty("error", out var error)
                        && error.TryGetProperty("message", out var text)
                        ? text.GetString()
                        : streamEvent.Data.ToString();
                    Dispatcher.UIThread.Post(() => Error = message ?? "");
                    return "Failed";
            }
        }
        return "";
    }

    private bool CanCancelDownload() => IsDownloading;

    [RelayCommand(CanExecute = nameof(CanCancelDownload))]
    private void CancelDownload() => _download?.Cancel();

    public void Dispose() => _download?.Cancel();
}
