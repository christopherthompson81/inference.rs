using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>A tab per modality, each with the models it loads.</summary>
public sealed class MainViewModel : IDisposable
{
    public MainViewModel(IFileService files)
    {
        Chat = new ChatViewModel(files);
        Transcription = new TranscriptionViewModel(files);
        Speech = new SpeechViewModel(files);
        Image = new ImageViewModel(files);
        Document = new DocumentViewModel(files);
        Engine = new EngineViewModel(Chat.Slot);
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

    public void Dispose()
    {
        Chat.Dispose();
        Transcription.Dispose();
        Speech.Dispose();
        Image.Dispose();
        Document.Dispose();
    }
}
