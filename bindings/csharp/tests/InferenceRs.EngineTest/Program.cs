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
    // Enough to outlast the few steps a cancel takes to land; the random weights never stop on their own.
    private const int LongCompletion = 512;
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
        var badTune = Throws(() => InferenceEngine.TuneModel("""{"model_id": "org/model", "dtype": "no-such-dtype"}"""));
        Check("tuning with an unknown dtype is InvalidRequest", badTune?.Status == InferenceStatus.InvalidRequest);

        using (var engine = InferenceEngine.Load(Spec(model)))
        {
            ChatAndStreamAgree(engine);
            MediaAttachmentsMatchDataUrls(engine);
            OtherProtocolsStream(engine);
            ResponsesAreStored(engine);
            ErrorsCarryTheEnvelope(engine);
            FilesRoundTrip(engine);
            OwnersStayApart(engine);
            SkillsAreStored(engine);
            RuntimeOperations(engine);
            ACancelledStreamEndsWithItsUsage(engine);
        }
        ModelsAreManagedAtRuntime(model);
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
        // The token cap ends the run: the random weights never stop on their own.
        Check("a capped Responses stream ends with response.incomplete", stream.LastOrDefault()?.Name == "response.incomplete");

        var models = JsonNode.Parse(engine.ListModels())!["data"]!.AsArray();
        Check("the model list starts with the default alias", (string?)models[0]!["id"] == "default");
        Check("one model card is marked default", models.Count(card => (bool?)card!["default"] == true) == 1);
        Check("the default alias is served", (bool)JsonNode.Parse(engine.ModelServed("""{"model_id": "default"}"""))!["served"]!);
        Check("an unknown model is not served",
            !(bool)JsonNode.Parse(engine.ModelServed("""{"model_id": "no-such-model"}"""))!["served"]!);
        Check("a model without MCP servers lists no MCP tools",
            JsonNode.Parse(engine.ListMcpTools())!["data"]!.AsArray().Count == 0);
        var counted = JsonNode.Parse(engine.AnthropicCountTokens(
            """{"model": "default", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok"}]}"""))!;
        Check("a Messages request counts its prompt tokens", (int)counted["input_tokens"]! > 0);
        var unknown = Throws(() => engine.ModelStatus("""{"model_id": "no-such-model"}"""));
        Check("an unknown model's status is NotFound", unknown?.Status == InferenceStatus.NotFound);
        var approval = Throws(() => engine.ResolveApproval("never-issued", """{"decision": "approve"}"""));
        Check("an unknown approval is NotFound", approval?.Status == InferenceStatus.NotFound);
    }

    private static void RuntimeOperations(InferenceEngine engine)
    {
        var tokens = JsonNode.Parse(engine.Tokenize("""{"text": "Reply with ok"}"""))!["tokens"]!;
        var detokenize = new JsonObject { ["tokens"] = tokens.DeepClone() }.ToJsonString();
        // The tiny tokenizer has no decoder, so its word-boundary markers come back as they are.
        var text = ((string?)JsonNode.Parse(engine.Detokenize(detokenize))!["text"])?.Replace('\u2581', ' ');
        Check("detokenizing tokens gives the text back", text == "Reply with ok");
        var (scored, noLogits) = engine.PromptLogits("""{"prompt": "Reply with ok"}""");
        var logprobs = JsonNode.Parse(scored)!["token_logprobs"]!.AsArray();
        Check("a scored prompt has a log-probability per token", logprobs.Count > 1 && logprobs[0] is null && noLogits is null);
        var withLogits = new JsonObject { ["prompt"] = tokens.DeepClone(), ["output"] = "logits" }.ToJsonString();
        var (scoredAgain, logits) = engine.PromptLogits(withLogits);
        var vocab = (int)JsonNode.Parse(scoredAgain)!["vocab_size"]!;
        Check("a scored prompt's logits are tokens times vocab", logits?.Length == tokens.AsArray().Count * vocab);

        var steps = 0;
        var forced = engine.RegisterLogitsProcessor("cs-forced", (stepLogits, _) =>
        {
            Interlocked.Increment(ref steps);
            stepLogits.Fill(float.NegativeInfinity);
            stepLogits[^1] = 0;
        });
        Check("a logits processor name registers once",
            Throws(() => engine.RegisterLogitsProcessor("cs-forced", (_, _) => { }))?.Status == InferenceStatus.InvalidRequest);
        const string processed = """{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "logits_processors": ["cs-forced"]}""";
        var generated = (int)JsonNode.Parse(engine.Chat(processed))!["usage"]!["completion_tokens"]!;
        Check("a request naming a logits processor runs it each step", steps > 0 && steps == generated);
        forced.Dispose();
        Check("a request naming an unregistered logits processor is InvalidRequest",
            Throws(() => engine.Chat(processed))?.Status == InferenceStatus.InvalidRequest);
        const string lateTool = """{"type": "function", "function": {"name": "cs_late", "parameters": {"type": "object"}}}""";
        var late = engine.RegisterTool(new HostTool(lateTool, _ => "found"));
        Check("a tool registered after load registers its name once",
            Throws(() => engine.RegisterTool(new HostTool(lateTool, _ => "found")))?.Status == InferenceStatus.InvalidRequest);
        late.Dispose();
        Check("a request naming an unregistered tool is InvalidRequest",
            Throws(() => engine.Chat("""{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "host_tools": ["cs_late"]}"""))?.Status
                == InferenceStatus.InvalidRequest);
        var scoped = engine.ForOwner("cs-processor-owner");
        var outliving = scoped.RegisterLogitsProcessor("cs-scoped", (_, _) => { });
        scoped.Dispose();
        outliving.Dispose();
        Check("a registration disposed after its engine still unregisters its name",
            Throws(() => engine.RegisterLogitsProcessor("cs-scoped", (_, _) => { }).Dispose()) is null);

        const string session = """{"messages": [{"role": {"Left": "user"}, "content": {"Left": "hi"}}]}""";
        Check("a session is imported", (string?)JsonNode.Parse(engine.PutSession("cs-session", session))!["id"] == "cs-session");
        Check("an imported session is listed",
            JsonNode.Parse(engine.ListSessions())!["data"]!.AsArray().Any(id => (string?)id == "cs-session"));
        Check("an imported session exports", JsonNode.Parse(engine.GetSession("cs-session"))!["messages"] is JsonArray);
        var fork = (string?)JsonNode.Parse(engine.ForkSession("cs-session", """{"num_turns": 0}"""))!["id"];
        Check("a forked session gets its own id", fork is not null && fork != "cs-session");
        Check("a forked session exports", JsonNode.Parse(engine.GetSession(fork!))!["messages"] is JsonArray);
        engine.DeleteSession(fork!);
        var unknownFork = Throws(() => engine.ForkSession("no-such-session", """{"num_turns": 0}"""));
        Check("forking an unknown session is InvalidRequest", unknownFork?.Status == InferenceStatus.InvalidRequest);
        Check("a session is deleted", (bool)JsonNode.Parse(engine.DeleteSession("cs-session"))!["deleted"]!);
        Check("a deleted session is NotFound", Throws(() => engine.GetSession("cs-session"))?.Status == InferenceStatus.NotFound);

        Check("calibration reports its status", JsonNode.Parse(engine.CalibrationStatus())!["collecting"] is not null);
        var cacheStats = JsonNode.Parse(engine.CacheStats())!["data"]!.AsArray();
        Check("cache stats list the model's encoder cache", cacheStats.Count > 0 && cacheStats[0]!["encoder_cache"] is JsonObject);
        var badIsq = Throws(() => engine.ReIsq("""{"ggml_type": "no-such-type"}"""));
        Check("an unknown ISQ type is InvalidRequest", badIsq?.Status == InferenceStatus.InvalidRequest);
    }

    private static void ACancelledStreamEndsWithItsUsage(InferenceEngine engine)
    {
        var request = JsonNode.Parse(ChatRequest(true))!;
        request["max_tokens"] = LongCompletion;
        request["ignore_eos"] = true;
        using var stream = engine.ChatStream(request.ToJsonString());
        var events = new List<StreamEvent>();
        if (stream.TryNext(TimeSpan.FromMinutes(1), out var first)) events.Add(first);
        var canceller = Task.Run(stream.Cancel);
        events.AddRange(stream);
        canceller.Wait();
        var last = events.Last(streamEvent => streamEvent.Name == "chunk").Data;
        Check("a cancelled stream finishes as canceled",
            last.GetProperty("choices")[0].GetProperty("finish_reason").GetString() == "canceled");
        Check("a cancelled stream reports its usage",
            last.GetProperty("usage").GetProperty("completion_tokens").GetInt32() < LongCompletion);
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

    private static void ModelsAreManagedAtRuntime(string model)
    {
        using var engine = InferenceEngine.Load(Spec(model));
        var selected = JsonNode.Parse(Spec(model))!["model"]!;
        var spec = new JsonObject { ["model"] = selected.DeepClone(), ["model_id"] = "second" }.ToJsonString();
        Check("a model is added at runtime", (string?)JsonNode.Parse(engine.AddModel(spec))!["status"] == "loaded");
        engine.SetDefaultModel("""{"model_id": "second"}""");
        Check("the added model becomes the default",
            JsonNode.Parse(engine.ListModels())!["data"]!.AsArray().Any(card =>
                (string?)card!["id"] == "second" && (bool?)card["default"] == true));
        engine.AddModelAlias("""{"alias": "spare", "model_id": "second"}""");
        Check("an alias names the added model", (bool)JsonNode.Parse(engine.ModelServed("""{"model_id": "spare"}"""))!["served"]!);
        engine.RemoveModel("""{"model_id": "second"}""");
        Check("a removed model is no longer served",
            !(bool)JsonNode.Parse(engine.ModelServed("""{"model_id": "second"}"""))!["served"]!);
    }

    private static void FilesRoundTrip(InferenceEngine engine)
    {
        var contents = Encoding.UTF8.GetBytes("col_a,col_b\n1,2\n");
        var uploaded = JsonNode.Parse(engine.UploadFile(contents, "table.csv", "user_data", "text/csv"))!;
        var id = (string)uploaded["id"]!;
        var blob = engine.FileContent(id);
        Check("file content round-trips", blob.Data.SequenceEqual(contents) && blob.MimeType == "text/csv");
        Check("a container no run used has no files",
            JsonNode.Parse(engine.ListContainerFiles("cntr_unused"))!["data"]!.AsArray().Count == 0);
        Check("a file outside the container is NotFound",
            Throws(() => engine.GetContainerFile("cntr_unused", id))?.Status == InferenceStatus.NotFound);
        Check("its content is NotFound too",
            Throws(() => engine.ContainerFileContent("cntr_unused", id))?.Status == InferenceStatus.NotFound);
        engine.DeleteFile(id);
        Check("a deleted file is NotFound", Throws(() => engine.GetFile(id))?.Status == InferenceStatus.NotFound);
        Check("a deleted file's content is NotFound",
            Throws(() => engine.FileContent(id))?.Status == InferenceStatus.NotFound);
        Check("an empty file uploads", JsonNode.Parse(engine.UploadFile([], "empty.txt", "user_data")) is JsonObject);
    }

    private static void OwnersStayApart(InferenceEngine engine)
    {
        using var teamA = engine.ForOwner("team-a");
        using var teamB = engine.ForOwner("team-b");
        var uploaded = JsonNode.Parse(teamA.UploadFile(Encoding.UTF8.GetBytes("a,b\n"), "table.csv", "user_data"))!;
        var id = (string)uploaded["id"]!;
        Check("an owner reads its own file", JsonNode.Parse(teamA.GetFile(id)) is JsonObject);
        Check("another owner can't", Throws(() => teamB.GetFile(id))?.Status == InferenceStatus.NotFound);
        Check("nor can the unscoped engine", Throws(() => engine.GetFile(id))?.Status == InferenceStatus.NotFound);
        Check("an empty owner is refused", Throws(() => engine.ForOwner(""))?.Status == InferenceStatus.InvalidArgument);
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
