using System.Runtime.InteropServices;
using System.Text;
using InferenceRs.Native;

namespace InferenceRs;

/// <summary>Bytes the engine returned with their MIME type: generated speech, or a file's content.</summary>
public sealed record Blob(byte[] Data, string MimeType);

/// <summary>Copies and frees the owned handles the ABI hands out.</summary>
internal static unsafe class Owned
{
    internal static string TakeString(IntPtr value)
    {
        try
        {
            var data = NativeMethods.inference_string_data(value);
            var len = checked((int)NativeMethods.inference_string_len(value));
            return Encoding.UTF8.GetString((byte*)data, len);
        }
        finally
        {
            NativeMethods.inference_string_free(value);
        }
    }

    internal static Blob TakeBlob(IntPtr blob)
    {
        try
        {
            var len = checked((int)NativeMethods.inference_blob_len(blob));
            var data = new byte[len];
            if (len > 0) Marshal.Copy(NativeMethods.inference_blob_data(blob), data, 0, len);
            return new Blob(data, Utf8.ToString(NativeMethods.inference_blob_mime_type(blob)));
        }
        finally
        {
            NativeMethods.inference_blob_free(blob);
        }
    }
}
