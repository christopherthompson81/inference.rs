using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using InferenceRs.Native;

namespace InferenceRs;

/// <summary>A host tool call: the arguments JSON the model passed and where in the agent loop it came from.</summary>
public sealed record HostToolCall(string Name, string ArgumentsJson, string? SessionId, int? Round);

/// <summary>A tool the model may call, answered by this process.</summary>
/// <param name="DefinitionJson">The OpenAI function tool JSON; its <c>function.name</c> is the tool's name.</param>
/// <param name="Handler">Returns what the model sees; an exception reaches the model as a failed tool call.</param>
/// <remarks>Handlers run on engine worker threads, possibly several at once, and must not call back into the engine.</remarks>
public sealed record HostTool(string DefinitionJson, Func<HostToolCall, string> Handler);

/// <summary>Host functions the agent loop calls, fixed when the engine loads.</summary>
public sealed class HostCallbacks
{
    public List<HostTool> Tools { get; } = [];

    /// <summary>Answers a web search with a JSON array of <c>{title, description, url, content}</c>.</summary>
    public Func<string, string>? Search { get; init; }
}

/// <summary>Handlers by the id the native side passes back as user_data.</summary>
/// <remarks>
/// An id, not a GCHandle: requests still finishing may call a callback after the engine is freed, and a removed id
/// fails that call instead of dereferencing a freed handle.
/// </remarks>
internal static class HostCallbackRegistry
{
    private static readonly ConcurrentDictionary<nint, Delegate> Handlers = new();
    private static long next;

    internal static nint Add(Delegate handler)
    {
        var id = (nint)Interlocked.Increment(ref next);
        Handlers[id] = handler;
        return id;
    }

    internal static T? Find<T>(nint id) where T : Delegate => Handlers.TryGetValue(id, out var handler) ? (T)handler : null;

    internal static void Remove(IEnumerable<nint> ids)
    {
        foreach (var id in ids) Handlers.TryRemove(id, out _);
    }
}

/// <summary>The unmanaged entry points the engine calls, and the native struct describing them.</summary>
internal static unsafe class HostCallbackBridge
{
    private const string EngineGone = "the engine that registered this callback has been freed";

    /// <summary>The native description of the handlers; their registrations pass to the engine on success.</summary>
    internal sealed class Pinned : IDisposable
    {
        private readonly NativeArray<NativeHostTool> _tools;
        private readonly NativeArray<NativeHostCallbacks> _callbacks = new(1);
        private List<nint>? _ids = [];

        internal Pinned(HostCallbacks callbacks)
        {
            _tools = new NativeArray<NativeHostTool>(callbacks.Tools.Count);
            try
            {
                for (var index = 0; index < callbacks.Tools.Count; index++)
                {
                    var tool = callbacks.Tools[index];
                    var definition = Encoding.UTF8.GetBytes(tool.DefinitionJson);
                    var id = HostCallbackRegistry.Add(tool.Handler);
                    _ids.Add(id);
                    _tools.Pointer[index] = new NativeHostTool
                    {
                        Definition = _tools.Bytes(definition),
                        DefinitionLen = (nuint)definition.Length,
                        Callback = &Tool,
                        UserData = id,
                    };
                }
                var search = callbacks.Search is { } handler ? HostCallbackRegistry.Add(handler) : 0;
                if (search != 0) _ids.Add(search);
                _callbacks.Pointer[0] = new NativeHostCallbacks
                {
                    Tools = _tools.Pointer,
                    ToolCount = _tools.Count,
                    Search = search == 0 ? null : &Search,
                    SearchUserData = search,
                };
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        internal NativeHostCallbacks* Native => _callbacks.Pointer;

        /// <summary>Hands the registrations to the engine, which removes them when it is freed.</summary>
        internal nint[] TakeIds()
        {
            var ids = _ids!.ToArray();
            _ids = null;
            return ids;
        }

        /// <summary>Frees the load-time copies (the library keeps none), and the registrations unless taken.</summary>
        public void Dispose()
        {
            if (_ids is not null) HostCallbackRegistry.Remove(_ids);
            _ids = null;
            _tools.Dispose();
            _callbacks.Dispose();
        }
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void Tool(
        IntPtr userData, byte* toolName, byte* arguments, nuint argumentsLen, byte* context, nuint contextLen, IntPtr result)
    {
        try
        {
            var handler = HostCallbackRegistry.Find<Func<HostToolCall, string>>(userData);
            if (handler is null)
            {
                Fail(result, EngineGone);
                return;
            }
            var (sessionId, round) = ReadContext(Utf8.ToString(context, contextLen));
            var call = new HostToolCall(
                Utf8.ToString((IntPtr)toolName), Utf8.ToString(arguments, argumentsLen), sessionId, round);
            Answer(result, handler(call));
        }
        catch (Exception exception)
        {
            Fail(result, exception.Message);
        }
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void Search(IntPtr userData, byte* query, nuint queryLen, IntPtr result)
    {
        try
        {
            var handler = HostCallbackRegistry.Find<Func<string, string>>(userData);
            if (handler is null)
            {
                Fail(result, EngineGone);
                return;
            }
            Answer(result, handler(Utf8.ToString(query, queryLen)));
        }
        catch (Exception exception)
        {
            Fail(result, exception.Message);
        }
    }

    private static void Answer(IntPtr result, string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        fixed (byte* data = bytes)
        {
            NativeMethods.inference_callback_result_set(result, data, (nuint)bytes.Length);
        }
    }

    // Nothing may escape an UnmanagedCallersOnly method, so a failure to report the failure reports it without detail.
    private static void Fail(IntPtr result, string message)
    {
        try
        {
            NativeMethods.inference_callback_result_fail(result, message);
        }
        catch
        {
            NativeMethods.inference_callback_result_fail(result, null);
        }
    }

    private static (string? SessionId, int? Round) ReadContext(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        var sessionId = root.TryGetProperty("session_id", out var session) && session.ValueKind == JsonValueKind.String
            ? session.GetString()
            : null;
        int? round = root.TryGetProperty("round", out var value) && value.ValueKind == JsonValueKind.Number
            && value.TryGetInt32(out var parsed) ? parsed : null;
        return (sessionId, round);
    }
}
