using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>Text to image with the diffusion models: FLUX and HiDream-O1.</summary>
public sealed partial class ImageViewModel : ObservableObject, IDisposable
{
    // what the model picks itself, for the step and guidance boxes
    private const decimal ModelDefault = 0;
    private const decimal RandomSeed = -1;

    private readonly IFileService _files;
    private byte[]? _image;

    public ImageViewModel(IFileService files)
    {
        _files = files;
        Slot = new ModelSlotViewModel(
            "Image generation model",
            "HiDream-ai/HiDream-O1-Image-Dev, black-forest-labs/FLUX.1-schnell, or a .gguf",
            Specs.Diffusion(() => Arch),
            files);
        Slot.Changed += (_, _) => GenerateCommand.NotifyCanExecuteChanged();
    }

    public ModelSlotViewModel Slot { get; }

    /// <summary>The diffusion architectures the engine loads, by their spec names.</summary>
    public ObservableCollection<string> Archs { get; } = ["hidream-o1-dev", "hidream-o1", "flux", "flux-offloaded"];

    [ObservableProperty]
    public partial string Arch { get; set; } = "hidream-o1-dev";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(GenerateCommand))]
    public partial string Prompt { get; set; } = "A lighthouse on a rocky coast at dusk, oil painting";

    [ObservableProperty]
    public partial decimal Width { get; set; } = 1024;

    [ObservableProperty]
    public partial decimal Height { get; set; } = 1024;

    [ObservableProperty]
    public partial decimal Steps { get; set; } = ModelDefault;

    [ObservableProperty]
    public partial decimal Guidance { get; set; } = ModelDefault;

    [ObservableProperty]
    public partial decimal Seed { get; set; } = RandomSeed;

    [ObservableProperty]
    public partial Bitmap? Image { get; set; }

    partial void OnImageChanged(Bitmap? oldValue, Bitmap? newValue) => oldValue?.Dispose();

    [ObservableProperty]
    public partial string Status { get; set; } = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(GenerateCommand))]
    [NotifyCanExecuteChangedFor(nameof(SaveCommand))]
    public partial bool IsBusy { get; set; }

    partial void OnIsBusyChanged(bool value) => Slot.IsInUse = value;

    private bool CanGenerate() => Slot.IsLoaded && Prompt.Trim().Length > 0 && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanGenerate))]
    private async Task GenerateAsync()
    {
        var engine = Slot.Engine!;
        var request = new JsonObject
        {
            ["model"] = Protocol.DefaultModel,
            ["prompt"] = Prompt.Trim(),
            ["width"] = (int)Width,
            ["height"] = (int)Height,
            ["response_format"] = "b64_json",
        };
        if (Steps > ModelDefault) request["steps"] = (int)Steps;
        if (Guidance > ModelDefault) request["guidance_scale"] = (double)Guidance;
        if (Seed > RandomSeed) request["seed"] = (long)Seed;
        IsBusy = true;
        Status = "Generating";
        try
        {
            var clock = Stopwatch.StartNew();
            _image = await Task.Run(() =>
            {
                var response = JsonDocument.Parse(engine.ImageGeneration(request.ToJsonString())).RootElement;
                return Convert.FromBase64String(response.GetProperty("data")[0].GetProperty("b64_json").GetString()!);
            });
            using var stream = new MemoryStream(_image);
            Image = new Bitmap(stream);
            Status = $"{Image.PixelSize.Width}x{Image.PixelSize.Height} in {Protocol.Seconds(clock)}";
        }
        catch (Exception error)
        {
            _image = null;
            Status = Protocol.Describe(error);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanSave() => _image is not null && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanSave))]
    private async Task SaveAsync()
    {
        try
        {
            await _files.SaveAsync("Save image", "image.png", _image!);
        }
        catch (Exception error)
        {
            Status = Protocol.Describe(error);
        }
    }

    public void Dispose()
    {
        Image = null;
        Slot.Dispose();
    }
}
