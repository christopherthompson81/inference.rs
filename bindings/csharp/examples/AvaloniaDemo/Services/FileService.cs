using Avalonia.Controls;
using Avalonia.Platform.Storage;

namespace InferenceRs.AvaloniaDemo.Services;

/// <summary>A file the user picked: its name, its local path when it has one, and its bytes.</summary>
public sealed record PickedFile(string Name, string? Path, byte[] Data);

/// <summary>The dialogs the view models need, so they never touch a control.</summary>
public interface IFileService
{
    Task<PickedFile?> OpenAsync(string title, params FilePickerFileType[] types);

    Task<string?> OpenPathAsync(string title, params FilePickerFileType[] types);

    Task<string?> OpenFolderAsync(string title);

    Task<bool> SaveAsync(string title, string suggestedName, byte[] data);

    /// <summary>Opens the bytes with the system's default application, from a temporary file.</summary>
    Task OpenWithSystemAsync(string fileName, byte[] data);
}

public static class FileTypes
{
    public static readonly FilePickerFileType Images = new("Images")
    {
        Patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"],
        MimeTypes = ["image/png", "image/jpeg", "image/webp", "image/bmp"],
    };

    public static readonly FilePickerFileType Audio = new("Audio")
    {
        Patterns = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"],
        MimeTypes = ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/mp4"],
    };

    public static readonly FilePickerFileType Gguf = new("GGUF") { Patterns = ["*.gguf"] };

    public static readonly FilePickerFileType All = new("All files") { Patterns = ["*"] };

    /// <summary>The MIME type for a media file, from its extension.</summary>
    public static string MimeOf(string name) => System.IO.Path.GetExtension(name).ToLowerInvariant() switch
    {
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".webp" => "image/webp",
        ".bmp" => "image/bmp",
        ".wav" => "audio/wav",
        ".mp3" => "audio/mpeg",
        ".flac" => "audio/flac",
        ".ogg" => "audio/ogg",
        ".m4a" => "audio/mp4",
        _ => "application/octet-stream",
    };
}

public sealed class FileService(TopLevel topLevel) : IFileService
{
    public async Task<PickedFile?> OpenAsync(string title, params FilePickerFileType[] types)
    {
        var files = await topLevel.StorageProvider.OpenFilePickerAsync(
            new FilePickerOpenOptions { Title = title, AllowMultiple = false, FileTypeFilter = types });
        if (files.Count == 0) return null;
        await using var stream = await files[0].OpenReadAsync();
        using var buffer = new MemoryStream();
        await stream.CopyToAsync(buffer);
        return new PickedFile(files[0].Name, files[0].TryGetLocalPath(), buffer.ToArray());
    }

    public async Task<string?> OpenPathAsync(string title, params FilePickerFileType[] types)
    {
        var files = await topLevel.StorageProvider.OpenFilePickerAsync(
            new FilePickerOpenOptions { Title = title, AllowMultiple = false, FileTypeFilter = types });
        return files.Count == 0 ? null : files[0].TryGetLocalPath();
    }

    public async Task<string?> OpenFolderAsync(string title)
    {
        var folders = await topLevel.StorageProvider.OpenFolderPickerAsync(
            new FolderPickerOpenOptions { Title = title, AllowMultiple = false });
        return folders.Count == 0 ? null : folders[0].TryGetLocalPath();
    }

    public async Task<bool> SaveAsync(string title, string suggestedName, byte[] data)
    {
        var file = await topLevel.StorageProvider.SaveFilePickerAsync(
            new FilePickerSaveOptions { Title = title, SuggestedFileName = suggestedName });
        if (file is null) return false;
        await using var stream = await file.OpenWriteAsync();
        await stream.WriteAsync(data);
        return true;
    }

    public async Task OpenWithSystemAsync(string fileName, byte[] data)
    {
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), fileName);
        await File.WriteAllBytesAsync(path, data);
        await topLevel.Launcher.LaunchFileInfoAsync(new FileInfo(path));
    }
}
