using InferenceRs.Native;

namespace InferenceRs;

/// <summary>Where a model runs: <c>cpu</c> (the default), <c>cuda</c> or <c>metal</c>.</summary>
/// <param name="Threads">CPU worker threads; 0 is the process default.</param>
public sealed record Backend(string? Name = null, int Device = 0, int Threads = 0);

/// <summary>One detected layout region, in reading order.</summary>
/// <param name="Box">x1, y1, x2, y2 in source-image pixels.</param>
/// <param name="Polygon">The region's outline as (x, y) vertices in source-image pixels; the box's corners when the
/// mask gives no outline.</param>
public sealed record LayoutDetection(int ClassId, string Label, float Score, float[] Box, (float X, float Y)[] Polygon);

/// <summary>An 8-bit image in caller memory, copied during detection.</summary>
/// <param name="Stride">Bytes per row; 0 means tightly packed.</param>
public sealed record LayoutImage(byte[] Pixels, uint Width, uint Height, PixelFormat Format, uint Stride = 0);

/// <summary>A PP-DocLayoutV3 document layout detector.</summary>
public sealed unsafe class LayoutModel : IDisposable
{
    /// <summary>Pass as a threshold to use the model's default (0.5).</summary>
    public const float DefaultThreshold = -1.0f;

    private const int BoxFloats = 4;

    private readonly LayoutHandle _model;

    private LayoutModel(IntPtr model) => _model = new LayoutHandle(model);

    /// <summary>Loads an HF-format directory (config.json, preprocessor_config.json, model.safetensors).</summary>
    public static LayoutModel Load(string modelDir, Backend? backend = null)
    {
        NativeMethods.EnsureAbi();
        backend ??= new Backend();
        var name = Utf8.Allocate(backend.Name);
        try
        {
            var config = new NativeBackendConfig { Backend = name, Device = backend.Device, Threads = backend.Threads };
            var status = NativeMethods.inference_layout_model_load(modelDir, &config, out var model);
            InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_layout_model_load));
            return new LayoutModel(model);
        }
        finally
        {
            Utf8.Free(name);
        }
    }

    /// <summary>The class labels, indexed by class id.</summary>
    public IReadOnlyList<string> Labels
    {
        get
        {
            using var model = new Lease(_model);
            var count = (int)NativeMethods.inference_layout_model_label_count(model.Handle);
            var labels = new string[count];
            for (var index = 0; index < count; index++)
            {
                var status = NativeMethods.inference_layout_model_label(model.Handle, (nuint)index, out var label);
                InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_layout_model_label));
                labels[index] = Utf8.ToString(label);
            }
            return labels;
        }
    }

    public IReadOnlyList<LayoutDetection> Detect(LayoutImage image, float threshold = DefaultThreshold) =>
        DetectBatch([image], threshold)[0];

    /// <summary>Detects on every image in one batched forward.</summary>
    public IReadOnlyList<IReadOnlyList<LayoutDetection>> DetectBatch(
        IReadOnlyList<LayoutImage> images, float threshold = DefaultThreshold)
    {
        if (images.Count == 0) return [];
        using var model = new Lease(_model);
        using var native = new NativeArray<NativeImage>(images.Count);
        for (var index = 0; index < images.Count; index++)
        {
            var image = images[index];
            native.Pointer[index] = new NativeImage
            {
                Pixels = native.Bytes(image.Pixels),
                Width = image.Width,
                Height = image.Height,
                Stride = image.Stride,
                Format = (int)image.Format,
            };
        }
        var results = new IntPtr[images.Count];
        fixed (IntPtr* outResults = results)
        {
            var status = NativeMethods.inference_layout_detect_batch(
                model.Handle, native.Pointer, native.Count, threshold, outResults);
            InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_layout_detect_batch));
        }
        try
        {
            return results.Select(Read).ToArray();
        }
        finally
        {
            foreach (var result in results) NativeMethods.inference_layout_result_free(result);
        }
    }

    public void Dispose() => _model.Dispose();

    private static IReadOnlyList<LayoutDetection> Read(IntPtr result)
    {
        var count = (int)NativeMethods.inference_layout_result_count(result);
        var detections = new LayoutDetection[count];
        for (var index = 0; index < count; index++)
        {
            var box = new float[BoxFloats];
            fixed (float* outBox = box)
            {
                var status = NativeMethods.inference_layout_result_detection(
                    result, (nuint)index, out var classId, out var label, out var score, outBox);
                InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_layout_result_detection));
                status = NativeMethods.inference_layout_result_polygon(
                    result, (nuint)index, out var points, out var pointCount);
                InferenceException.ThrowIfFailed(status, nameof(NativeMethods.inference_layout_result_polygon));
                var polygon = new (float X, float Y)[(int)pointCount];
                var coordinates = (float*)points;
                for (var vertex = 0; vertex < polygon.Length; vertex++)
                {
                    polygon[vertex] = (coordinates[2 * vertex], coordinates[2 * vertex + 1]);
                }
                detections[index] = new LayoutDetection(classId, Utf8.ToString(label), score, box, polygon);
            }
        }
        return detections;
    }
}
