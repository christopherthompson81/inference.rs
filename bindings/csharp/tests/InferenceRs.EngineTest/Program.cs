using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using InferenceRs.Native;

namespace InferenceRs.EngineTest;

/// <summary>
/// Drives the engine through the bindings on the tiny random-weight checkpoint, which
/// `cargo run -p inference-ffi --example tiny_checkpoint -- DIR` writes. Exit codes follow CTest: 0, 1, 77 skip.
/// </summary>
internal static class Program
{
    private const string ModelVariable = "INFERENCE_TEST_TINY_CHECKPOINT";
    private const int MaxTokens = 6;
    private const string Prompt = "Reply with the single word: ok";
    private static readonly TimeSpan PollTimeout = TimeSpan.FromSeconds(60);
    private const string ImageFixture = "crates/inference/tests/fixtures/paddleocr_vl/page_00.png";

    private static int failures;

    private static int Main()
    {
        var model = Environment.GetEnvironmentVariable(ModelVariable);
        if (string.IsNullOrEmpty(model))
        {
            Console.WriteLine($"{ModelVariable} is not set; skipping.");
            return 77;
        }
        Check("system info is JSON", JsonNode.Parse(InferenceEngine.SystemInfo()) is JsonObject);

        using (var engine = InferenceEngine.Load(Spec(model)))
        {
            ChatAndStreamAgree(engine);
            MediaAttachmentsMatchDataUrls(engine);
            OtherProtocolsStream(engine);
            ResponsesAreStored(engine);
            ErrorsCarryTheEnvelope(engine);
            FilesRoundTrip(engine);
            SkillsAreStored(engine);
        }
        StreamsOutliveTheirEngine(model);
        HostToolsLoadAndBadOnesAreRefused(model);

        Console.WriteLine(failures == 0 ? "all engine checks passed" : $"{failures} engine checks failed");
        return failures == 0 ? 0 : 1;
    }

    private static string Spec(string model) => new JsonObject
    {
        ["model"] = new JsonObject { ["MultimodalPlain"] = new JsonObject { ["model_id"] = model, ["dtype"] = "f32" } },
        ["runtime"] = new JsonObject { ["device"] = "cpu" },
    }.ToJsonString();

    private static string ChatRequest(bool stream) => new JsonObject
    {
        ["model"] = "default",
        ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = Prompt }),
        ["max_tokens"] = MaxTokens,
        ["temperature"] = 0.0,
        ["top_k"] = 1,
        ["stream"] = stream,
    }.ToJsonString();

    private static void ChatAndStreamAgree(InferenceEngine engine)
    {
        var response = JsonNode.Parse(engine.Chat(ChatRequest(false)))!;
        Check("chat returns a chat.completion", (string?)response["object"] == "chat.completion");
        var text = (string?)response["choices"]![0]!["message"]!["content"] ?? string.Empty;

        var streamed = new StringBuilder();
        using (var stream = engine.ChatStream(ChatRequest(true)))
        {
            foreach (var streamEvent in stream)
            {
                Check("chat streams chunk events", streamEvent.Name == "chunk");
                var delta = streamEvent.Data.GetProperty("choices")[0].GetProperty("delta");
                if (delta.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String)
                    streamed.Append(content.GetString());
            }
            Check("a finished stream stays finished", stream.IsDone && !stream.TryNext(TimeSpan.Zero, out _));
        }
        Check("streamed text equals the blocking reply", streamed.ToString() == text);
    }

    private static string ImageRequest(string url) => new JsonObject
    {
        ["model"] = "default",
        ["messages"] = new JsonArray(new JsonObject
        {
            ["role"] = "user",
            ["content"] = new JsonArray(
                new JsonObject { ["type"] = "image_url", ["image_url"] = new JsonObject { ["url"] = url } },
                new JsonObject { ["type"] = "text", ["text"] = "OCR:" }),
        }),
        ["max_tokens"] = MaxTokens,
        ["temperature"] = 0.0,
        ["top_k"] = 1,
    }.ToJsonString();

    private static void MediaAttachmentsMatchDataUrls(InferenceEngine engine)
    {
        var png = File.ReadAllBytes(FindInCheckout(ImageFixture));
        var byUrl = JsonNode.Parse(engine.Chat(ImageRequest($"data:image/png;base64,{Convert.ToBase64String(png)}")))!;
        var byMedia = JsonNode.Parse(engine.Chat(ImageRequest("media://0"), [new MediaAttachment(png, "image/png")]))!;
        Check("an attached image decodes like the same image as a data URL",
            (string?)byUrl["choices"]![0]!["message"]!["content"] == (string?)byMedia["choices"]![0]!["message"]!["content"]);
    }

    private static void OtherProtocolsStream(InferenceEngine engine)
    {
        var completion = new JsonObject
        {
            ["model"] = "default", ["prompt"] = Prompt, ["max_tokens"] = MaxTokens, ["temperature"] = 0.0, ["top_k"] = 1,
        }.ToJsonString();
        Check("a completion returns text_completion",
            (string?)JsonNode.Parse(engine.Completion(completion))!["object"] == "text_completion");
        using (var stream = engine.CompletionStream(completion))
        {
            Check("a completion stream opens with nothing ready or a chunk",
                !stream.TryNext(TimeSpan.Zero, out var first) || first.Name == "chunk");
            Check("a completion stream streams chunks", stream.All(streamEvent => streamEvent.Name == "chunk"));
        }

        var messages = new JsonObject
        {
            ["model"] = "default",
            ["max_tokens"] = MaxTokens,
            ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = Prompt }),
            ["temperature"] = 0.0,
            ["top_k"] = 1,
        }.ToJsonString();
        Check("Anthropic Messages returns a message",
            (string?)JsonNode.Parse(engine.AnthropicMessages(messages))!["type"] == "message");
        using (var stream = engine.AnthropicMessagesStream(messages))
        {
            var names = new List<string>();
            while (stream.TryNext(PollTimeout, out var streamEvent)) names.Add(streamEvent.Name);
            Check("an Anthropic stream runs message_start to message_stop",
                names.FirstOrDefault() == "message_start" && names.LastOrDefault() == "message_stop");
        }
    }

    private static void ResponsesAreStored(InferenceEngine engine)
    {
        var request = new JsonObject
        {
            ["model"] = "default", ["input"] = Prompt, ["max_output_tokens"] = MaxTokens, ["temperature"] = 0.0, ["top_k"] = 1,
        };
        var created = JsonNode.Parse(engine.CreateResponse(request.ToJsonString()))!;
        var id = (string)created["id"]!;
        Check("a stored response is fetched", (string?)JsonNode.Parse(engine.GetResponse(id))!["id"] == id);
        engine.DeleteResponse(id);
        Check("a deleted response is NotFound", Throws(() => engine.GetResponse(id))?.Status == InferenceStatus.NotFound);
        request["stream"] = true;
        using var stream = engine.ResponseStream(request.ToJsonString());
        Check("a Responses stream ends with response.completed", stream.LastOrDefault()?.Name == "response.completed");

        var models = JsonNode.Parse(engine.ListModels())!["data"]!.AsArray();
        Check("the model list starts with the default alias", (string?)models[0]!["id"] == "default");
        var unknown = Throws(() => engine.ModelStatus("""{"model_id": "no-such-model"}"""));
        Check("an unknown model's status is NotFound", unknown?.Status == InferenceStatus.NotFound);
        var approval = Throws(() => engine.ResolveApproval("never-issued", """{"decision": "approve"}"""));
        Check("an unknown approval is NotFound", approval?.Status == InferenceStatus.NotFound);
    }

    private static void StreamsOutliveTheirEngine(string model)
    {
        var engine = InferenceEngine.Load(Spec(model));
        using var stream = engine.ChatStream(ChatRequest(true));
        engine.Dispose();
        Check("a stream runs to its end after its engine is disposed", stream.Count() > 0 && stream.IsDone);
        var afterDispose = false;
        try
        {
            engine.Chat(ChatRequest(false));
        }
        catch (ObjectDisposedException)
        {
            afterDispose = true;
        }
        Check("a disposed engine refuses calls", afterDispose);
    }

    private static string FindInCheckout(string relative)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var path = Path.Combine(dir.FullName, relative);
            if (File.Exists(path)) return path;
        }
        throw new FileNotFoundException(relative);
    }

    private static void ErrorsCarryTheEnvelope(InferenceEngine engine)
    {
        var unknown = Throws(() => engine.Chat(
            """{"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]}"""));
        Check("an unknown model is NotFound", unknown?.Status == InferenceStatus.NotFound);
        Check("the error code comes from the envelope", unknown?.Code == "model_not_found");
        var malformed = Throws(() => engine.Chat("{not json"));
        Check("malformed JSON is InvalidRequest", malformed?.Status == InferenceStatus.InvalidRequest);
    }

    private static void FilesRoundTrip(InferenceEngine engine)
    {
        var contents = Encoding.UTF8.GetBytes("col_a,col_b\n1,2\n");
        var uploaded = JsonNode.Parse(engine.UploadFile(contents, "table.csv", "user_data", "text/csv"))!;
        var id = (string)uploaded["id"]!;
        var blob = engine.FileContent(id);
        Check("file content round-trips", blob.Data.SequenceEqual(contents) && blob.MimeType == "text/csv");
        engine.DeleteFile(id);
        Check("a deleted file is NotFound", Throws(() => engine.GetFile(id))?.Status == InferenceStatus.NotFound);
        Check("a deleted file's content is NotFound",
            Throws(() => engine.FileContent(id))?.Status == InferenceStatus.NotFound);
        Check("an empty file uploads", JsonNode.Parse(engine.UploadFile([], "empty.txt", "user_data")) is JsonObject);
    }

    private static void SkillsAreStored(InferenceEngine engine)
    {
        var skillMd = "---\nname: csv-summary\ndescription: Summarizes a CSV file.\n---\nRead the file.\n";
        var skill = JsonNode.Parse(engine.UploadSkill([new SkillFile("SKILL.md", Encoding.UTF8.GetBytes(skillMd))]))!;
        Check("a skill uploads", (string?)skill["name"] == "csv-summary");
        var listed = JsonNode.Parse(engine.ListSkills())!["data"]!.AsArray();
        Check("the skill is listed", listed.Count == 1);
        var bad = Throws(() => engine.UploadSkill([new SkillFile("SKILL.md", Encoding.UTF8.GetBytes("no frontmatter"))]));
        Check("a skill without frontmatter is InvalidRequest", bad?.Status == InferenceStatus.InvalidRequest);
    }

    private static void HostToolsLoadAndBadOnesAreRefused(string model)
    {
        var definition = """{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}""";
        var callbacks = new HostCallbacks { Search = query => "[]" };
        callbacks.Tools.Add(new HostTool(definition, call => call.ArgumentsJson));
        using (var engine = InferenceEngine.Load(Spec(model), callbacks))
        {
            Check("chat works with a host tool registered",
                JsonNode.Parse(engine.Chat(ChatRequest(false))) is JsonObject);
        }
        var bad = new HostCallbacks();
        bad.Tools.Add(new HostTool("not json", call => ""));
        var refused = Throws(() => InferenceEngine.Load(Spec(model), bad));
        Check("a malformed tool definition is InvalidArgument", refused?.Status == InferenceStatus.InvalidArgument);
    }

    private static InferenceException? Throws(Action action)
    {
        try
        {
            action();
            return null;
        }
        catch (InferenceException exception)
        {
            return exception;
        }
    }

    private static void Check(string name, bool passed)
    {
        if (passed) return;
        failures++;
        Console.Error.WriteLine($"  FAILED: {name}");
    }
}
