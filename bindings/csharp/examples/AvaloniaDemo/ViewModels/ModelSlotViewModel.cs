using System.Diagnostics;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>Reads a slot's settings on the UI thread into the load a worker thread then runs.</summary>
public delegate Func<Action<LoadProgress>, IDisposable> SlotLoader(ModelSlotViewModel slot);

/// <summary>One model a tab runs: where it comes from, the device, and its load progress and errors.</summary>
public sealed partial class ModelSlotViewModel : ObservableObject, IDisposable
{
    private const double BytesPerMiB = 1024 * 1024;

    // one load at a time: progress reaches every listener in the process, so each bar shows only its own load
    private static readonly SemaphoreSlim LoadGate = new(1, 1);

    private readonly SlotLoader _loader;
    private readonly IFileService _files;
    private IDisposable? _handle;

    public ModelSlotViewModel(string title, string hint, SlotLoader loader, IFileService files, bool hasContext = false)
    {
        Title = title;
        Hint = hint;
        HasContext = hasContext;
        _loader = loader;
        _files = files;
    }

    public string Title { get; }

    /// <summary>An example source, shown while the box is empty.</summary>
    public string Hint { get; }

    /// <summary>Whether the model keeps a KV cache, whose context length the slot then sets.</summary>
    public bool HasContext { get; }

    public IReadOnlyList<string> Devices { get; } = Host.DeviceOptions;

    /// <summary>Raised on the UI thread when a model loads or unloads.</summary>
    public event EventHandler? Changed;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadCommand))]
    [NotifyPropertyChangedFor(nameof(IsGguf))]
    [NotifyPropertyChangedFor(nameof(TakesMmproj))]
    public partial string Source { get; set; } = "";

    [ObservableProperty]
    public partial string Device { get; set; } = DefaultDevice();

    [ObservableProperty]
    public partial decimal ContextLength { get; set; } = SettingsStore.Current.ContextLength;

    /// <summary>The multimodal projector of a GGUF model, in the GGUF's directory.</summary>
    [ObservableProperty]
    public partial string MmprojPath { get; set; } = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadCommand))]
    [NotifyCanExecuteChangedFor(nameof(UnloadCommand))]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    public partial bool IsLoading { get; set; }

    /// <summary>Set while the tab runs the model, which must then stay loaded.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadCommand))]
    [NotifyCanExecuteChangedFor(nameof(UnloadCommand))]
    [NotifyPropertyChangedFor(nameof(IsIdle))]
    public partial bool IsInUse { get; set; }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(UnloadCommand))]
    public partial bool IsLoaded { get; set; }

    [ObservableProperty]
    public partial string Status { get; set; } = "Not loaded";

    [ObservableProperty]
    public partial string Stage { get; set; } = "";

    [ObservableProperty]
    public partial double Percent { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    public partial string Error { get; set; } = "";

    public bool HasError => Error.Length > 0;

    /// <summary>Whether the slot's settings may change: not while it loads or its tab runs it.</summary>
    public bool IsIdle => !IsLoading && !IsInUse;

    public bool IsGguf => Source.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);

    /// <summary>A chat GGUF, whose images and audio need the model's multimodal projector.</summary>
    public bool TakesMmproj => HasContext && IsGguf;

    /// <summary>The loaded model, when it is an engine.</summary>
    public InferenceEngine? Engine => _handle as InferenceEngine;

    /// <summary>The loaded model, when it is a layout detector.</summary>
    public LayoutModel? Layout => _handle as LayoutModel;

    /// <summary>The device, or null to let the engine pick.</summary>
    public string? DeviceOrNull => Device == Host.AutoDevice ? null : Device;

    private bool CanLoad() => IsIdle && Source.Trim().Length > 0;

    // the settings' device, unless this machine no longer has it
    private static string DefaultDevice() =>
        Host.DeviceOptions.Contains(SettingsStore.Current.Device) ? SettingsStore.Current.Device : Host.AutoDevice;

    /// <summary>The slot's model and options, to remember for the next session.</summary>
    public SlotState Capture() => new()
    {
        Source = Source,
        Device = Device,
        ContextLength = ContextLength,
        MmprojPath = MmprojPath,
    };

    /// <summary>Fills the slot in from a remembered session; loading it is up to the caller.</summary>
    public void Restore(SlotState state)
    {
        Source = state.Source;
        if (Host.DeviceOptions.Contains(state.Device)) Device = state.Device;
        if (state.ContextLength > 0) ContextLength = state.ContextLength;
        MmprojPath = state.MmprojPath;
    }

    /// <summary>Puts a model the model manager picked into the slot, unloading what it held.</summary>
    public void Assign(string source, string mmprojPath)
    {
        if (!IsIdle) return;
        Unload();
        Error = "";
        Source = source;
        MmprojPath = mmprojPath;
    }

    [RelayCommand(CanExecute = nameof(CanLoad))]
    private async Task LoadAsync()
    {
        Unload();
        Error = "";
        Percent = 0;
        Func<Action<LoadProgress>, IDisposable> load;
        try
        {
            load = _loader(this);
        }
        catch (Exception error)
        {
            Error = Protocol.Describe(error);
            Status = "Not loaded";
            return;
        }
        IsLoading = true;
        Status = "Waiting for another model to finish loading";
        await LoadGate.WaitAsync();
        Status = "Loading";
        Stage = "Starting";
        var clock = Stopwatch.StartNew();
        try
        {
            _handle = await Task.Run(() => load(OnProgress));
            IsLoaded = true;
            Status = $"Loaded {Source.Trim()} in {Protocol.Seconds(clock)}";
        }
        catch (Exception error)
        {
            Error = Protocol.Describe(error);
            Status = "Load failed";
        }
        finally
        {
            LoadGate.Release();
            IsLoading = false;
            Stage = "";
            Changed?.Invoke(this, EventArgs.Empty);
        }
    }

    private bool CanUnload() => IsLoaded && IsIdle;

    [RelayCommand(CanExecute = nameof(CanUnload))]
    private void Unload()
    {
        if (_handle is null) return;
        _handle.Dispose();
        _handle = null;
        IsLoaded = false;
        Status = "Not loaded";
        Changed?.Invoke(this, EventArgs.Empty);
    }

    [RelayCommand]
    private async Task BrowseFileAsync()
    {
        if (await _files.OpenPathAsync($"{Title}: model file", FileTypes.Gguf, FileTypes.All) is { } path) Source = path;
    }

    [RelayCommand]
    private async Task BrowseFolderAsync()
    {
        if (await _files.OpenFolderAsync($"{Title}: model directory") is { } path) Source = path;
    }

    [RelayCommand]
    private async Task BrowseMmprojAsync()
    {
        if (await _files.OpenPathAsync("Multimodal projector (mmproj GGUF)", FileTypes.Gguf) is { } path) MmprojPath = path;
    }

    // load threads call this; the UI shows the stage reported last
    private void OnProgress(LoadProgress progress) => Dispatcher.UIThread.Post(() =>
    {
        if (!IsLoading) return;
        Percent = progress.Total == 0 ? 100 : 100.0 * progress.Done / progress.Total;
        Stage = progress.Unit == "bytes"
            ? $"{progress.Stage}: {progress.Done / BytesPerMiB:0} of {progress.Total / BytesPerMiB:0} MiB"
            : $"{progress.Stage}: {progress.Done} of {progress.Total}";
    });

    public void Dispose() => Unload();
}
