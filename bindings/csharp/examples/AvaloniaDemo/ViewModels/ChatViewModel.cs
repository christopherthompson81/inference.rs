using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>An image or audio clip attached to a message.</summary>
public sealed partial class AttachmentViewModel(string name, byte[] data, string mimeType) : ObservableObject, IDisposable
{
    private const int ThumbnailWidth = 160;

    public string Name { get; } = name;

    public byte[] Data { get; } = data;

    public string MimeType { get; } = mimeType;

    public bool IsImage => MimeType.StartsWith("image/", StringComparison.Ordinal);

    /// <summary>The thumbnail of an image; null for audio or an image Avalonia cannot decode.</summary>
    public Bitmap? Thumbnail { get; } = mimeType.StartsWith("image/", StringComparison.Ordinal) ? TryDecode(data) : null;

    private static Bitmap? TryDecode(byte[] data)
    {
        try
        {
            using var stream = new MemoryStream(data);
            return Bitmap.DecodeToWidth(stream, ThumbnailWidth);
        }
        catch (Exception)
        {
            return null;
        }
    }

    public void Dispose() => Thumbnail?.Dispose();
}

public sealed partial class ChatMessageViewModel(string role) : ObservableObject
{
    public string Role { get; } = role;

    public bool IsUser => Role == "user";

    public ObservableCollection<AttachmentViewModel> Attachments { get; } = [];

    [ObservableProperty]
    public partial string Content { get; set; } = "";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasReasoning))]
    public partial string Reasoning { get; set; } = "";

    public bool HasReasoning => Reasoning.Length > 0;

    [ObservableProperty]
    public partial bool IsStreaming { get; set; }

    /// <summary>Why the reply stopped short; shown beside it, never sent back as the model's words.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    public partial string Error { get; set; } = "";

    public bool HasError => Error.Length > 0;

    [ObservableProperty]
    public partial string Stats { get; set; } = "";
}

/// <summary>Streaming chat with sampling settings, cancellation, reasoning and media attachments.</summary>
public sealed partial class ChatViewModel : ObservableObject, IDisposable
{
    private readonly IFileService _files;
    private EngineStream? _stream;

    public ChatViewModel(IFileService files)
    {
        _files = files;
        Slot = new ModelSlotViewModel(
            "Chat model",
            "Qwen/Qwen3-0.6B, a model directory, or a .gguf file",
            Specs.Chat,
            files,
            hasContext: true);
        Slot.Changed += (_, _) => SendCommand.NotifyCanExecuteChanged();
        Attachments.CollectionChanged += (_, _) => SendCommand.NotifyCanExecuteChanged();
    }

    public ModelSlotViewModel Slot { get; }

    public ObservableCollection<ChatMessageViewModel> Messages { get; } = [];

    public ObservableCollection<AttachmentViewModel> Attachments { get; } = [];

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    public partial string Input { get; set; } = "";

    [ObservableProperty]
    public partial string SystemPrompt { get; set; } = "";

    // NumericUpDown binds decimal only
    [ObservableProperty]
    public partial decimal Temperature { get; set; } = 0.7m;

    [ObservableProperty]
    public partial decimal TopP { get; set; } = 0.95m;

    [ObservableProperty]
    public partial decimal MaxTokens { get; set; } = 1024;

    [ObservableProperty]
    public partial bool EnableThinking { get; set; } = true;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    [NotifyCanExecuteChangedFor(nameof(ClearCommand))]
    public partial bool IsGenerating { get; set; }

    partial void OnIsGeneratingChanged(bool value) => Slot.IsInUse = value;

    private bool CanSend() => Slot.IsLoaded && !IsGenerating && (Input.Trim().Length > 0 || Attachments.Count > 0);

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        var engine = Slot.Engine!;
        var user = new ChatMessageViewModel("user") { Content = Input.Trim() };
        foreach (var attachment in Attachments) user.Attachments.Add(attachment);
        Messages.Add(user);
        Input = "";
        Attachments.Clear();
        var (request, media) = BuildRequest();
        var reply = new ChatMessageViewModel("assistant") { IsStreaming = true };
        Messages.Add(reply);
        IsGenerating = true;
        var clock = Stopwatch.StartNew();
        try
        {
            _stream = engine.ChatStream(request, media);
            var tokens = await Task.Run(() => Drain(_stream, reply));
            reply.Stats = tokens > 0 ? $"{tokens} tokens, {tokens / clock.Elapsed.TotalSeconds:0.0} tok/s" : "";
        }
        catch (Exception error)
        {
            reply.Error = Protocol.Describe(error);
        }
        finally
        {
            reply.IsStreaming = false;
            _stream?.Dispose();
            _stream = null;
            IsGenerating = false;
        }
    }

    // runs on a worker thread; returns the completion tokens the final usage reports
    private static int Drain(EngineStream stream, ChatMessageViewModel reply)
    {
        var tokens = 0;
        foreach (var streamEvent in stream)
        {
            if (streamEvent.Name == "error")
            {
                var text = streamEvent.Data.ToString();
                Dispatcher.UIThread.Post(() => reply.Error = text);
                continue;
            }
            if (streamEvent.Name != "chunk") continue;
            var chunk = streamEvent.Data;
            if (chunk.TryGetProperty("usage", out var usage) && usage.ValueKind == JsonValueKind.Object
                && usage.TryGetProperty("completion_tokens", out var completion))
            {
                tokens = completion.GetInt32();
            }
            if (!chunk.TryGetProperty("choices", out var choices) || choices.GetArrayLength() == 0) continue;
            if (!choices[0].TryGetProperty("delta", out var delta)) continue;
            var content = Text(delta, "content");
            var reasoning = Text(delta, "reasoning_content");
            if (content.Length == 0 && reasoning.Length == 0) continue;
            Dispatcher.UIThread.Post(() =>
            {
                reply.Content += content;
                reply.Reasoning += reasoning;
            });
        }
        return tokens;
    }

    private static string Text(JsonElement delta, string name) =>
        delta.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString()! : "";

    // the whole conversation, with every attachment numbered across it as `media://N`
    private (string Request, List<MediaAttachment> Media) BuildRequest()
    {
        var media = new List<MediaAttachment>();
        var messages = new JsonArray();
        if (SystemPrompt.Trim() is { Length: > 0 } system)
        {
            messages.Add(new JsonObject { ["role"] = "system", ["content"] = system });
        }
        foreach (var message in Messages)
        {
            if (!message.IsUser)
            {
                messages.Add(new JsonObject { ["role"] = "assistant", ["content"] = message.Content });
                continue;
            }
            var parts = new JsonArray();
            foreach (var attachment in message.Attachments)
            {
                var url = Protocol.MediaUrl(media.Count);
                media.Add(new MediaAttachment(attachment.Data, attachment.MimeType));
                var kind = attachment.IsImage ? "image_url" : "audio_url";
                parts.Add(new JsonObject { ["type"] = kind, [kind] = new JsonObject { ["url"] = url } });
            }
            parts.Add(new JsonObject { ["type"] = "text", ["text"] = message.Content });
            messages.Add(new JsonObject { ["role"] = "user", ["content"] = parts });
        }
        var request = new JsonObject
        {
            ["model"] = Protocol.DefaultModel,
            ["messages"] = messages,
            ["stream"] = true,
            ["temperature"] = (double)Temperature,
            ["top_p"] = (double)TopP,
            ["max_tokens"] = (int)MaxTokens,
            ["enable_thinking"] = EnableThinking,
        };
        return (request.ToJsonString(), media);
    }

    private bool CanStop() => IsGenerating;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop() => _stream?.Cancel();

    private bool CanClear() => !IsGenerating;

    [RelayCommand(CanExecute = nameof(CanClear))]
    private void Clear()
    {
        foreach (var attachment in Messages.SelectMany(message => message.Attachments).Concat(Attachments))
        {
            attachment.Dispose();
        }
        Messages.Clear();
        Attachments.Clear();
    }

    [RelayCommand]
    private async Task AttachAsync()
    {
        if (await _files.OpenAsync("Attach an image or audio clip", FileTypes.Images, FileTypes.Audio) is { } file)
        {
            Attachments.Add(new AttachmentViewModel(file.Name, file.Data, FileTypes.MimeOf(file.Name)));
        }
    }

    [RelayCommand]
    private void RemoveAttachment(AttachmentViewModel attachment)
    {
        Attachments.Remove(attachment);
        attachment.Dispose();
    }

    public void Dispose()
    {
        _stream?.Cancel();
        Clear();
        Slot.Dispose();
    }
}
