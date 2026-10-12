using System.Diagnostics;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>Text to speech: generate a WAV, play it with the system player, or save it.</summary>
public sealed partial class SpeechViewModel : ObservableObject, IDisposable
{
    private const string AudioFormat = "wav";

    private readonly IFileService _files;
    private byte[]? _audio;

    public SpeechViewModel(IFileService files)
    {
        _files = files;
        Slot = new ModelSlotViewModel(
            "Speech synthesis model",
            "hexgrad/Kokoro-82M, or a Dia model",
            Specs.Speech,
            files);
        Slot.Changed += (_, _) => GenerateCommand.NotifyCanExecuteChanged();
    }

    public ModelSlotViewModel Slot { get; }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(GenerateCommand))]
    public partial string Text { get; set; } = "Hello! This sentence was spoken by inference.rs, from C#.";

    /// <summary>A voice the model ships (Kokoro: af_heart, bm_george, ...; comma-separated voices blend).</summary>
    [ObservableProperty]
    public partial string Voice { get; set; } = "af_heart";

    [ObservableProperty]
    public partial decimal Speed { get; set; } = 1.0m;

    [ObservableProperty]
    public partial string Status { get; set; } = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(GenerateCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(SaveCommand))]
    public partial bool IsBusy { get; set; }

    partial void OnIsBusyChanged(bool value) => Slot.IsInUse = value;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(SaveCommand))]
    public partial bool HasAudio { get; set; }

    private bool CanGenerate() => Slot.IsLoaded && Text.Trim().Length > 0 && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanGenerate))]
    private async Task GenerateAsync()
    {
        var engine = Slot.Engine!;
        var request = new JsonObject
        {
            ["model"] = Protocol.DefaultModel,
            ["input"] = Text.Trim(),
            ["response_format"] = AudioFormat,
            ["speed"] = (double)Speed,
        };
        if (Voice.Trim() is { Length: > 0 } voice) request["voice"] = voice;
        IsBusy = true;
        Status = "Generating";
        try
        {
            var clock = Stopwatch.StartNew();
            _audio = (await Task.Run(() => engine.SpeechGeneration(request.ToJsonString()))).Data;
            HasAudio = true;
            Status = $"{_audio.Length / 1024} KiB of audio in {Protocol.Seconds(clock)}";
        }
        catch (Exception error)
        {
            Status = Protocol.Describe(error);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool HasResult() => HasAudio && !IsBusy;

    [RelayCommand(CanExecute = nameof(HasResult))]
    private async Task PlayAsync()
    {
        // a fresh name each time: a player may still hold the last file open
        try
        {
            await _files.OpenWithSystemAsync($"inference-rs-speech-{Guid.NewGuid():N}.{AudioFormat}", _audio!);
        }
        catch (Exception error)
        {
            Status = Protocol.Describe(error);
        }
    }

    [RelayCommand(CanExecute = nameof(HasResult))]
    private async Task SaveAsync()
    {
        try
        {
            await _files.SaveAsync("Save speech", $"speech.{AudioFormat}", _audio!);
        }
        catch (Exception error)
        {
            Status = Protocol.Describe(error);
        }
    }

    public void Dispose() => Slot.Dispose();
}
