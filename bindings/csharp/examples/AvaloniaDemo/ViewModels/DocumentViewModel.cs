using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InferenceRs.AvaloniaDemo.Services;
using LayoutPixels = InferenceRs.Native.PixelFormat;

namespace InferenceRs.AvaloniaDemo.ViewModels;

/// <summary>A detected layout region, in source-image pixels, with the text read from it.</summary>
/// <param name="badgeFontSize">The size of the region's number badge, which sits left of its box.</param>
public sealed partial class RegionViewModel(int order, LayoutDetection detection, double badgeFontSize) : ObservableObject
{
    private const double BadgeWidthPerFontSize = 1.8;

    public int Order { get; } = order;

    public string Label { get; } = detection.Label;

    public float Score { get; } = detection.Score;

    public double X { get; } = detection.Box[0];

    public double Y { get; } = detection.Box[1];

    public double Right { get; } = detection.Box[2];

    public double Bottom { get; } = detection.Box[3];

    public double Width => Math.Max(1, Right - X);

    public double Height => Math.Max(1, Bottom - Y);

    public string Caption => $"{Order}. {Label} ({Score:0.00})";

    public double BadgeFontSize { get; } = badgeFontSize;

    public double BadgeX => X - BadgeFontSize * BadgeWidthPerFontSize;

    [ObservableProperty]
    public partial string Text { get; set; } = "";
}

/// <summary>Document OCR: layout detection, then each region read by an OCR vision-language model.</summary>
public sealed partial class DocumentViewModel : ObservableObject, IDisposable
{
    private const int BytesPerPixel = 4;
    private const int AlphaOffset = 3;
    private const byte Opaque = 255;
    // region badges scale with the page, so they read the same on a phone photo and a 300 dpi scan
    private const double BadgePageFraction = 1.0 / 40;
    private const double BadgeGutterPerFontSize = 2;
    private const int OcrMaxTokens = 2048;
    private static readonly Vector Dpi = new(96, 96);

    // PaddleOCR-VL's task prompts by PP-DocLayoutV3 label; every other text kind takes the text prompt
    private static readonly Dictionary<string, string> TaskPrompts = new()
    {
        ["table"] = "Table Recognition:",
        ["display_formula"] = "Formula Recognition:",
        ["inline_formula"] = "Formula Recognition:",
        ["chart"] = "Chart Recognition:",
    };
    // the PP-DocLayoutV3 labels of regions with no text to read
    private static readonly HashSet<string> Pictures = ["image", "header_image", "footer_image", "seal"];

    private readonly IFileService _files;
    private byte[]? _pagePixels;

    public DocumentViewModel(IFileService files)
    {
        _files = files;
        LayoutSlot = new ModelSlotViewModel(
            "Layout model (PP-DocLayoutV3)",
            "PaddlePaddle/PP-DocLayoutV3_safetensors directory, or its .gguf",
            Specs.Layout,
            files);
        OcrSlot = new ModelSlotViewModel(
            "OCR model",
            "PaddlePaddle/PaddleOCR-VL-1.5, or a .gguf with its mmproj",
            Specs.Ocr,
            files,
            hasContext: true);
        LayoutSlot.Changed += (_, _) => DetectCommand.NotifyCanExecuteChanged();
        OcrSlot.Changed += (_, _) => ReadCommand.NotifyCanExecuteChanged();
    }

    public ModelSlotViewModel LayoutSlot { get; }

    public ModelSlotViewModel OcrSlot { get; }

    public ObservableCollection<RegionViewModel> Regions { get; } = [];

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DetectCommand))]
    [NotifyCanExecuteChangedFor(nameof(ReadCommand))]
    public partial Bitmap? Page { get; set; }

    partial void OnPageChanged(Bitmap? oldValue, Bitmap? newValue) => oldValue?.Dispose();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(BadgeGutter))]
    public partial double PageWidth { get; set; }

    private double BadgeFontSize => Math.Max(1, PageWidth * BadgePageFraction);

    /// <summary>Room left of the page for the badges of regions at its edge.</summary>
    public Thickness BadgeGutter => new(BadgeFontSize * BadgeGutterPerFontSize, 0, 0, 0);

    [ObservableProperty]
    public partial double PageHeight { get; set; }

    /// <summary>The prompt for text regions, and for the whole page when nothing was detected.</summary>
    [ObservableProperty]
    public partial string TextPrompt { get; set; } = "OCR:";

    [ObservableProperty]
    public partial string Text { get; set; } = "";

    [ObservableProperty]
    public partial string Status { get; set; } = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ChoosePageCommand))]
    [NotifyCanExecuteChangedFor(nameof(DetectCommand))]
    [NotifyCanExecuteChangedFor(nameof(ReadCommand))]
    public partial bool IsBusy { get; set; }

    partial void OnIsBusyChanged(bool value) => LayoutSlot.IsInUse = OcrSlot.IsInUse = value;

    private bool CanChoosePage() => !IsBusy;

    [RelayCommand(CanExecute = nameof(CanChoosePage))]
    private async Task ChoosePageAsync()
    {
        try
        {
            if (await _files.OpenAsync("Page image", FileTypes.Images) is { } file) UsePage(file);
        }
        catch (Exception error)
        {
            Status = Protocol.Describe(error);
        }
    }

    /// <summary>Shows <paramref name="file"/> as the page to detect and read.</summary>
    public void UsePage(PickedFile file)
    {
        using var stream = new MemoryStream(file.Data);
        var page = new Bitmap(stream);
        _pagePixels = Pixels(page);
        Page = page;
        PageWidth = page.PixelSize.Width;
        PageHeight = page.PixelSize.Height;
        Regions.Clear();
        Text = "";
        Status = $"{file.Name}: {page.PixelSize.Width}x{page.PixelSize.Height}";
    }

    // the page as tightly packed opaque BGRA rows, as the layout model and the crops take it
    private static byte[] Pixels(Bitmap page)
    {
        var size = page.PixelSize;
        var stride = size.Width * BytesPerPixel;
        var pixels = new byte[stride * size.Height];
        // the framebuffer overload converts from whatever the file decoded to (gray, 1-bit, RGBA)
        using var converted = new WriteableBitmap(size, Dpi, PixelFormat.Bgra8888, AlphaFormat.Premul);
        using (var buffer = converted.Lock())
        {
            page.CopyPixels(buffer);
            for (var row = 0; row < size.Height; row++)
            {
                Marshal.Copy(buffer.Address + row * buffer.RowBytes, pixels, row * stride, stride);
            }
        }
        // premultiplied pixels over white, so a transparent page reads as paper, not black
        for (var offset = 0; offset < pixels.Length; offset += BytesPerPixel)
        {
            var clear = Opaque - pixels[offset + AlphaOffset];
            for (var channel = 0; channel < AlphaOffset; channel++) pixels[offset + channel] += (byte)clear;
            pixels[offset + AlphaOffset] = Opaque;
        }
        return pixels;
    }

    private bool CanDetect() => LayoutSlot.IsLoaded && Page is not null && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanDetect))]
    private async Task DetectAsync()
    {
        var layout = LayoutSlot.Layout!;
        var image = new LayoutImage(_pagePixels!, (uint)PageWidth, (uint)PageHeight, LayoutPixels.Bgra8);
        IsBusy = true;
        Status = "Detecting layout";
        try
        {
            var clock = Stopwatch.StartNew();
            var detections = await Task.Run(() => layout.Detect(image));
            Regions.Clear();
            for (var index = 0; index < detections.Count; index++)
            {
                Regions.Add(new RegionViewModel(index + 1, detections[index], BadgeFontSize));
            }
            Status = $"{detections.Count} regions in {Protocol.Seconds(clock)}";
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

    private bool CanRead() => OcrSlot.IsLoaded && Page is not null && !IsBusy;

    [RelayCommand(CanExecute = nameof(CanRead))]
    private async Task ReadAsync()
    {
        var engine = OcrSlot.Engine!;
        var (pixels, width, height) = (_pagePixels!, (int)PageWidth, (int)PageHeight);
        List<(RegionViewModel? Region, PixelRect Bounds, string Prompt)> jobs = Regions.Count == 0
            ? [(null, new PixelRect(0, 0, width, height), TextPrompt)]
            : Regions.Where(region => !Pictures.Contains(region.Label))
                .Select(region => ((RegionViewModel?)region, Bounds(region, width, height),
                    TaskPrompts.GetValueOrDefault(region.Label, TextPrompt)))
                .ToList();
        IsBusy = true;
        Text = "";
        var text = new StringBuilder();
        try
        {
            var clock = Stopwatch.StartNew();
            for (var index = 0; index < jobs.Count; index++)
            {
                var (region, bounds, prompt) = jobs[index];
                Status = $"Reading {index + 1} of {jobs.Count}";
                var read = await Task.Run(() => Ocr(engine, Png(pixels, width, bounds), prompt));
                if (region is not null) region.Text = read;
                text.AppendLine(read).AppendLine();
                Text = text.ToString();
            }
            Status = $"Read {jobs.Count} regions in {Protocol.Seconds(clock)}";
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

    // the region's box, clamped to the page
    private static PixelRect Bounds(RegionViewModel region, int width, int height)
    {
        var left = Math.Clamp((int)Math.Floor(region.X), 0, width - 1);
        var top = Math.Clamp((int)Math.Floor(region.Y), 0, height - 1);
        var right = Math.Clamp((int)Math.Ceiling(region.Right), left + 1, width);
        var bottom = Math.Clamp((int)Math.Ceiling(region.Bottom), top + 1, height);
        return new PixelRect(left, top, right - left, bottom - top);
    }

    // `rect` of the page, as a PNG
    private static byte[] Png(byte[] pixels, int pageWidth, PixelRect rect)
    {
        var stride = pageWidth * BytesPerPixel;
        using var crop = new WriteableBitmap(rect.Size, Dpi, PixelFormat.Bgra8888, AlphaFormat.Premul);
        using (var target = crop.Lock())
        {
            for (var row = 0; row < rect.Height; row++)
            {
                Marshal.Copy(
                    pixels,
                    (rect.Y + row) * stride + rect.X * BytesPerPixel,
                    target.Address + row * target.RowBytes,
                    rect.Width * BytesPerPixel);
            }
        }
        using var stream = new MemoryStream();
        crop.Save(stream, new PngBitmapEncoderOptions());
        return stream.ToArray();
    }

    private static string Ocr(InferenceEngine engine, byte[] png, string prompt)
    {
        var request = new JsonObject
        {
            ["model"] = Protocol.DefaultModel,
            ["messages"] = new JsonArray(new JsonObject
            {
                ["role"] = "user",
                ["content"] = new JsonArray(
                    new JsonObject { ["type"] = "image_url", ["image_url"] = new JsonObject { ["url"] = Protocol.MediaUrl(0) } },
                    new JsonObject { ["type"] = "text", ["text"] = prompt }),
            }),
            ["max_tokens"] = OcrMaxTokens,
            ["temperature"] = 0.0,
        };
        var reply = JsonDocument.Parse(engine.Chat(request.ToJsonString(), [new MediaAttachment(png, "image/png")]));
        return reply.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()?.Trim() ?? "";
    }

    public void Dispose()
    {
        Page = null;
        LayoutSlot.Dispose();
        OcrSlot.Dispose();
    }
}
