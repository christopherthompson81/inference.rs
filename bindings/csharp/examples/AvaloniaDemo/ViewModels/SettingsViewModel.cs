using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>An edited copy of the settings; saving writes it and makes it current, cancelling drops it.</summary>
public sealed partial class SettingsViewModel : ObservableObject
{
    private readonly IFileService _files;

    public SettingsViewModel(AppSettings current, IFileService files)
    {
        _files = files;
        Settings = SettingsStore.Copy(current);
        Folders = new ObservableCollection<string>(Settings.ModelFolders);
        HfCachePath = Settings.HfCachePath;
    }

    public AppSettings Settings { get; }

    public ObservableCollection<string> Folders { get; }

    public IReadOnlyList<string> Devices { get; } = Host.DeviceOptions;

    public string SettingsFile => SettingsStore.FilePath;

    [ObservableProperty]
    public partial string HfCachePath { get; set; }

    [RelayCommand]
    private async Task AddFolderAsync()
    {
        if (await _files.OpenFolderAsync("A folder of models") is { } folder && !Folders.Contains(folder)) Folders.Add(folder);
    }

    [RelayCommand]
    private void RemoveFolder(string folder) => Folders.Remove(folder);

    [RelayCommand]
    private async Task BrowseHfCacheAsync()
    {
        if (await _files.OpenFolderAsync("Hugging Face cache (the hub directory)") is { } folder) HfCachePath = folder;
    }

    /// <summary>Writes the edited settings and returns them, now current.</summary>
    public AppSettings Save()
    {
        Settings.ModelFolders = [.. Folders];
        Settings.HfCachePath = HfCachePath.Trim();
        SettingsStore.Save(Settings);
        return Settings;
    }
}
