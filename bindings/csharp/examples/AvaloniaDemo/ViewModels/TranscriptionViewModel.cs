using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>A transcript segment, with its speaker when a diarization model ran.</summary>
public sealed record TranscriptSegment(double Start, double End, string Text, string Speaker)
{
    public string Span => $"{Start,7:0.00} - {End,7:0.00}";
}

/// <summary>Speech to text for an audio file, with segment timestamps and optional speaker labels.</summary>
public sealed partial class TranscriptionViewModel : ObservableObject, IDisposable
{
    private readonly IFileService _files;
    private PickedFile? _audio;

    public TranscriptionViewModel(IFileService files)
    {
        _files = files;
        Slot = new ModelSlotViewModel(
            "Speech recognition model",
            "nvidia/parakeet-tdt-0.6b-v3, or a .nemo / directory / .gguf",
            Specs.Transcription,
            files);
        Diarizer = new ModelSlotViewModel(
            "Speaker diarization model (optional)",
            "nvidia/diar_streaming_sortformer_4spk-v2.1",
            Specs.Diarization,
            files);
        Slot.Changed += (_, _) => TranscribeCommand.NotifyCanExecuteChanged();
    }

    public ModelSlotViewModel Slot { get; }

    public ModelSlotViewModel Diarizer { get; }

    public ObservableCollection<TranscriptSegment> Segments { get; } = [];

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(TranscribeCommand))]
    public partial string AudioName { get; set; } = "No audio chosen";

    [ObservableProperty]
    public partial string Language { get; set; } = "";

    [ObservableProperty]
    public partial string Transcript { get; set; } = "";

    [ObservableProperty]
    public partial string Status { get; set; } = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(TranscribeCommand))]
    public partial bool IsBusy { get; set; }

    partial void OnIsBusyChanged(bool value) => Slot.IsInUse = Diarizer.IsInUse = value;

    [RelayCommand]
    private async Task ChooseAudioAsync()
    {
        if (await _files.OpenAsync("Audio to transcribe", FileTypes.Audio, FileTypes.All) is { } file) UseAudio(file);
    }

    /// <summary>Transcribes <paramref name="file"/> next.</summary>
    public void UseAudio(PickedFile file)
    {
        _audio = file;
        AudioName = file.Name;
    }

    private bool CanTranscribe() => Slot.IsLoaded && _audio is not null && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanTranscribe))]
    private async Task TranscribeAsync()
    {
        var engine = Slot.Engine!;
        var diarizer = Diarizer.Engine;
        var audio = _audio!.Data;
        var request = new JsonObject { ["model"] = Protocol.DefaultModel, ["response_format"] = "verbose_json" };
        if (Language.Trim() is { Length: > 0 } language) request["language"] = language;
        IsBusy = true;
        Status = "Transcribing";
        Transcript = "";
        Segments.Clear();
        try
        {
            var clock = Stopwatch.StartNew();
            var (text, segments) = await Task.Run(() =>
            {
                var transcript = JsonDocument.Parse(engine.Transcription(request.ToJsonString(), audio).Data).RootElement;
                var turns = diarizer is null ? [] : Turns(diarizer.Diarization("""{"response_format": "json"}""", audio));
                return (transcript.GetProperty("text").GetString() ?? "", Label(transcript, turns));
            });
            Transcript = text;
            foreach (var segment in segments) Segments.Add(segment);
            Status = $"Done in {Protocol.Seconds(clock)}";
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

    private static List<(double Start, double End, string Speaker)> Turns(Blob diarization)
    {
        return JsonDocument.Parse(diarization.Data).RootElement.GetProperty("segments").EnumerateArray()
            .Select(turn => (
                turn.GetProperty("start").GetDouble(),
                turn.GetProperty("end").GetDouble(),
                $"Speaker {turn.GetProperty("speaker")}"))
            .ToList();
    }

    // each transcript segment takes the speaker whose turns overlap it most
    private static List<TranscriptSegment> Label(JsonElement transcript, List<(double Start, double End, string Speaker)> turns)
    {
        return transcript.GetProperty("segments").EnumerateArray().Select(segment =>
        {
            var start = segment.GetProperty("start").GetDouble();
            var end = segment.GetProperty("end").GetDouble();
            var speaker = turns
                .GroupBy(turn => turn.Speaker)
                .Select(turn => (turn.Key, Overlap: turn.Sum(t => Math.Max(0, Math.Min(end, t.End) - Math.Max(start, t.Start)))))
                .Where(turn => turn.Overlap > 0)
                .OrderByDescending(turn => turn.Overlap)
                .Select(turn => turn.Key)
                .FirstOrDefault() ?? "";
            return new TranscriptSegment(start, end, segment.GetProperty("text").GetString()?.Trim() ?? "", speaker);
        }).ToList();
    }

    public void Dispose()
    {
        Slot.Dispose();
        Diarizer.Dispose();
    }
}
