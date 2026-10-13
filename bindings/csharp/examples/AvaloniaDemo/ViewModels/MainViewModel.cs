using CommunityToolkit.Mvvm.ComponentModel;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>A tab per modality, each with the models it loads.</summary>
public sealed partial class MainViewModel : ObservableObject, IDisposable
{
    private const int ChatTab = 0;
    private const int TranscriptionTab = 1;
    private const int SpeechTab = 2;
    private const int ImagesTab = 3;
    private const int DocumentsTab = 4;

    public MainViewModel(IFileService files)
    {
        Chat = new ChatViewModel(files);
        Transcription = new TranscriptionViewModel(files);
        Speech = new SpeechViewModel(files);
        Image = new ImageViewModel(files);
        Document = new DocumentViewModel(files);
        Engine = new EngineViewModel(Chat.Slot);
        RestoreSession();
    }

    /// <summary>Why the native library could not be used, shown over every tab.</summary>
    public string? HostError => Host.Error;

    public bool HasHostError => Host.Error is not null;

    public ChatViewModel Chat { get; }

    public TranscriptionViewModel Transcription { get; }

    public SpeechViewModel Speech { get; }

    public ImageViewModel Image { get; }

    public DocumentViewModel Document { get; }

    public EngineViewModel Engine { get; }

    [ObservableProperty]
    public partial int SelectedTab { get; set; }

    private IEnumerable<ModelSlotViewModel> Slots =>
    [
        Chat.Slot, Transcription.Slot, Transcription.Diarizer, Speech.Slot, Image.Slot, Document.LayoutSlot, Document.OcrSlot,
    ];

    public SettingsViewModel CreateSettings(IFileService files) => new(SettingsStore.Current, files);

    public ModelManagerViewModel CreateModelManager() => new(Assign);

    /// <summary>Takes newly saved settings into the open tabs: the chat panel's sampling.</summary>
    public void ApplySettings(AppSettings settings)
    {
        Chat.Temperature = settings.Temperature;
        Chat.TopP = settings.TopP;
        Chat.TopK = settings.TopK;
        Chat.MinP = settings.MinP;
        Chat.MaxTokens = settings.MaxTokens;
        Chat.EnableThinking = settings.EnableThinking;
        Chat.SystemPrompt = settings.SystemPrompt;
    }

    private void Assign(ModelTarget target, string source, string mmprojPath)
    {
        var (slot, tab) = target switch
        {
            ModelTarget.Chat => (Chat.Slot, ChatTab),
            ModelTarget.Documents => (Document.OcrSlot, DocumentsTab),
            ModelTarget.Layout => (Document.LayoutSlot, DocumentsTab),
            ModelTarget.Speech => (Speech.Slot, SpeechTab),
            ModelTarget.Transcription => (Transcription.Slot, TranscriptionTab),
            ModelTarget.Diarization => (Transcription.Diarizer, TranscriptionTab),
            _ => (Image.Slot, ImagesTab),
        };
        slot.Assign(source, mmprojPath);
        SelectedTab = tab;
    }

    private void RestoreSession()
    {
        var settings = SettingsStore.Current;
        if (!settings.RestoreSession) return;
        foreach (var slot in Slots)
        {
            if (!settings.Slots.TryGetValue(slot.Title, out var state)) continue;
            slot.Restore(state);
            // the slots' shared gate loads them one at a time
            if (settings.LoadOnStartup && slot.LoadCommand.CanExecute(null)) slot.LoadCommand.Execute(null);
        }
    }

    private void SaveSession()
    {
        var settings = SettingsStore.Current;
        if (!settings.RestoreSession) return;
        settings.Slots = Slots.Where(slot => slot.Source.Trim().Length > 0).ToDictionary(slot => slot.Title, slot => slot.Capture());
        try
        {
            SettingsStore.Save(settings);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // a read-only config directory loses the session, not the shutdown
        }
    }

    public void Dispose()
    {
        SaveSession();
        Chat.Dispose();
        Transcription.Dispose();
        Speech.Dispose();
        Image.Dispose();
        Document.Dispose();
    }
}
