using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>Text to image with the diffusion models (FLUX and HiDream-O1), and HiDream-O1's edits: one reference image
/// to edit, or several as subjects for a new scene.</summary>
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
        References.CollectionChanged += (_, _) =>
        {
            OnPropertyChanged(nameof(IsEditing));
            OnPropertyChanged(nameof(HasOneReference));
            OnPropertyChanged(nameof(UsesSize));
        };
    }

    public ModelSlotViewModel Slot { get; }

    /// <summary>The diffusion architectures the engine loads, by their spec names.</summary>
    public ObservableCollection<string> Archs { get; } = ["hidream-o1-dev", "hidream-o1", "flux", "flux-offloaded"];

    [ObservableProperty]
    public partial string Arch { get; set; } = "hidream-o1-dev";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(GenerateCommand))]
    public partial string Prompt { get; set; } = "A lighthouse on a rocky coast at dusk, oil painting";

    /// <summary>The images an edit conditions on; with none, the prompt alone generates.</summary>
    public ObservableCollection<AttachmentViewModel> References { get; } = [];

    public bool IsEditing => References.Count > 0;

    public bool HasOneReference => References.Count == 1;

    /// <summary>With one reference, the output can take the reference's own size instead of Width and Height.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(UsesSize))]
    public partial bool MatchReference { get; set; } = true;

    public bool UsesSize => References.Count != 1 || !MatchReference;

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
    [NotifyCanExecuteChangedFor(nameof(AddReferenceCommand))]
    [NotifyCanExecuteChangedFor(nameof(RemoveReferenceCommand))]
    public partial bool IsBusy { get; set; }

    partial void OnIsBusyChanged(bool value) => Slot.IsInUse = value;

    private bool IsIdle() => !IsBusy;

    [RelayCommand(CanExecute = nameof(IsIdle))]
    private async Task AddReferenceAsync()
    {
        try
        {
            if (await _files.OpenAsync("Reference image", FileTypes.Images) is { } file)
            {
                References.Add(new AttachmentViewModel(file.Name, file.Data, FileTypes.MimeOf(file.Name)));
            }
        }
        catch (Exception error)
        {
            Status = Protocol.Describe(error);
        }
    }

    [RelayCommand(CanExecute = nameof(IsIdle))]
    private void RemoveReference(AttachmentViewModel reference)
    {
        References.Remove(reference);
        reference.Dispose();
    }

    private bool CanGenerate() => Slot.IsLoaded && Prompt.Trim().Length > 0 && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanGenerate))]
    private async Task GenerateAsync()
    {
        var engine = Slot.Engine!;
        var request = new JsonObject
        {
            ["model"] = Protocol.DefaultModel,
            ["prompt"] = Prompt.Trim(),
            ["response_format"] = "b64_json",
        };
        if (UsesSize)
        {
            request["width"] = (int)Width;
            request["height"] = (int)Height;
        }
        // the request names each reference by its position among the attached buffers
        var media = References.Select(r => new MediaAttachment(r.Data, r.MimeType)).ToList();
        if (media.Count > 0)
        {
            request["images"] = new JsonArray(media.Select((_, i) =>
                (JsonNode)new JsonObject { ["image_url"] = Protocol.MediaUrl(i) }).ToArray());
        }
        if (Steps > ModelDefault) request["steps"] = (int)Steps;
        if (Guidance > ModelDefault) request["guidance_scale"] = (double)Guidance;
        if (Seed > RandomSeed) request["seed"] = (long)Seed;
        IsBusy = true;
        Status = IsEditing ? "Editing" : "Generating";
        try
        {
            var clock = Stopwatch.StartNew();
            _image = await Task.Run(() =>
            {
                var json = media.Count > 0
                    ? engine.ImageEdit(request.ToJsonString(), media)
                    : engine.ImageGeneration(request.ToJsonString());
                var response = JsonDocument.Parse(json).RootElement;
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
        foreach (var reference in References) reference.Dispose();
        Image = null;
        Slot.Dispose();
    }
}
