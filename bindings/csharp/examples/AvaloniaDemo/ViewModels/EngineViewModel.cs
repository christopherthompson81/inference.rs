using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>One card of the chat engine's model list.</summary>
/// <param name="canOperate">A served model of its own: not the <c>default</c> alias, nor a LoRA adapter on one.</param>
public sealed class ServedModelViewModel(string id, bool isDefault, string status, bool canOperate)
{
    // the alias card carries no status of its own
    private const string AliasStatus = "alias";

    public string Id { get; } = id;

    public bool IsDefault { get; } = isDefault;

    public string Status { get; } = status.Length > 0 ? status : AliasStatus;

    public bool CanOperate { get; } = canOperate;
}

/// <summary>The chat engine's models (unload and reload) and its metrics, and the host's system report.</summary>
public sealed partial class EngineViewModel : ObservableObject
{
    private const string NoEngine = "Load a chat model to see its engine";
    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    private readonly ModelSlotViewModel _chat;

    public EngineViewModel(ModelSlotViewModel chat)
    {
        _chat = chat;
        _chat.Changed += (_, _) =>
        {
            if (RefreshCommand.CanExecute(null)) RefreshCommand.Execute(null);
        };
    }

    public ObservableCollection<ServedModelViewModel> Models { get; } = [];

    public string SystemReport => Host.Report;

    [ObservableProperty]
    public partial string CacheStats { get; set; } = "";

    [ObservableProperty]
    public partial string SpeculativeStats { get; set; } = "";

    [ObservableProperty]
    public partial string Status { get; set; } = NoEngine;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RefreshCommand))]
    [NotifyCanExecuteChangedFor(nameof(UnloadCommand))]
    [NotifyCanExecuteChangedFor(nameof(ReloadCommand))]
    public partial bool IsBusy { get; set; }

    private bool IsIdle() => !IsBusy;

    [RelayCommand(CanExecute = nameof(IsIdle))]
    private async Task RefreshAsync()
    {
        IsBusy = true;
        try
        {
            await Refresh();
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanOperate(ServedModelViewModel? model) => model is { CanOperate: true } && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanOperate))]
    private Task UnloadAsync(ServedModelViewModel model) =>
        Operate(model, engine => engine.UnloadModel(Request(model.Id)));

    [RelayCommand(CanExecute = nameof(CanOperate))]
    private Task ReloadAsync(ServedModelViewModel model) =>
        Operate(model, engine => engine.ReloadModel(Request(model.Id)));

    private async Task Operate(ServedModelViewModel model, Func<InferenceEngine, string> operation)
    {
        if (_chat.Engine is not { } engine) return;
        IsBusy = true;
        Status = $"Working on {model.Id}";
        try
        {
            await Task.Run(() => operation(engine));
            await Refresh();
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

    private async Task Refresh()
    {
        Models.Clear();
        if (_chat.Engine is not { } engine)
        {
            CacheStats = SpeculativeStats = "";
            Status = NoEngine;
            return;
        }
        try
        {
            var (models, cache, speculative) = await Task.Run(() =>
            {
                var cards = JsonDocument.Parse(engine.ListModels()).RootElement.GetProperty("data").EnumerateArray()
                    .Select(card => new ServedModelViewModel(
                        card.GetProperty("id").GetString()!,
                        card.TryGetProperty("default", out var isDefault) && isDefault.GetBoolean(),
                        card.TryGetProperty("status", out var status) ? status.GetString()! : "",
                        card.TryGetProperty("status", out _) && !card.TryGetProperty("parent", out _)))
                    .ToList();
                return (cards, Pretty(engine.CacheStats()), Pretty(engine.SpeculativeStats()));
            });
            foreach (var model in models) Models.Add(model);
            CacheStats = cache;
            SpeculativeStats = speculative;
            Status = $"Updated {DateTime.Now:T}";
        }
        catch (Exception error)
        {
            Status = Protocol.Describe(error);
        }
    }

    private static string Request(string id) => new JsonObject { ["model_id"] = id }.ToJsonString();

    private static string Pretty(string json) => JsonNode.Parse(json)!.ToJsonString(Indented);
}
