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

    // the tiny Parakeet the tiny_checkpoint example writes beside the PaddleOCR-VL checkpoint
    private const string ParakeetDir = "parakeet";
    private const string StreamingDir = "parakeet_streaming";
    // a streamed session's pieces: 100 ms of 16-bit samples, across the tiny model's chunks
    private const int PcmPiece = 3200;
    private const string SileroDir = "silero";
    private const string DiarizationDir = "diarization";
    private const int ToneRate = 16000;
    private const double ToneHz = 440.0;

    private static int failures;

    private static int Main()
    {
        ModelInventoryNeedsNoEngine();
        var model = Environment.GetEnvironmentVariable(ModelVariable);
        if (string.IsNullOrEmpty(model))
        {
            Console.WriteLine($"{ModelVariable} is not set; skipping the engine.");
            return failures == 0 ? 77 : 1;
        }
        Check("system info is JSON", JsonNode.Parse(InferenceEngine.SystemInfo()) is JsonObject);
        var badTune = Throws(() => InferenceEngine.TuneModel("""{"model_id": "org/model", "dtype": "no-such-dtype"}"""));
        Check("tuning with an unknown dtype is InvalidRequest", badTune?.Status == InferenceStatus.InvalidRequest);

        using (var engine = InferenceEngine.Load(Spec(model)))
        {
            ChatAndStreamAgree(engine);
            MediaAttachmentsMatchDataUrls(engine);
            AnEditTakesItsImagesAsAttachments(engine);
            OtherProtocolsStream(engine);
            ModelsAreListed(engine);
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
        ALoadReportsItsProgress(model);
        AudioTranscribes(Path.Combine(model, ParakeetDir));
        StreamedAudioTranscribes(Path.Combine(model, StreamingDir), Path.Combine(model, ParakeetDir));
        VoiceActivityFindsTheSpeech(Path.Combine(model, SileroDir));
        DiarizationNamesWhoSpeaks(Path.Combine(model, DiarizationDir));

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

    private static void AnEditTakesItsImagesAsAttachments(InferenceEngine engine)
    {
        var png = File.ReadAllBytes(FindInCheckout(ImageFixture));
        const string edit = """{"prompt": "make it blue", "images": [{"image_url": "media://0"}]}""";
        // the buffer reaches the engine, which a chat model refuses as it refuses generation
        var refused = Throws(() => engine.ImageEdit(edit, [new MediaAttachment(png, "image/png")]));
        Check("an edit's attachment reaches the engine, which a chat model refuses",
            refused?.Status == InferenceStatus.InvalidRequest && refused.Detail.Contains("incompatible"));
        var unattached = Throws(() => engine.ImageEdit(edit));
        Check("an edit naming no attachment is refused", unattached?.Detail.Contains("invalid_image") == true);
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
    }

    private static void ModelsAreListed(InferenceEngine engine)
    {
        var models = JsonNode.Parse(engine.ListModels())!["data"]!.AsArray();
        Check("the model list starts with the default alias", (string?)models[0]!["id"] == "default");
        Check("one model card is marked default", models.Count(card => (bool?)card!["default"] == true) == 1);
        Check("the default alias is served", (bool)JsonNode.Parse(engine.ModelServed("""{"model_id": "default"}"""))!["served"]!);
        Check("an unknown model is not served",
            !(bool)JsonNode.Parse(engine.ModelServed("""{"model_id": "no-such-model"}"""))!["served"]!);
        Check("a model without MCP servers lists no MCP tools",
            JsonNode.Parse(engine.ListMcpTools())!["data"]!.AsArray().Count == 0);
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
        Check("calibration of an unknown model is NotFound",
            Throws(() => engine.CalibrationStatus("""{"model": "no-such-model"}"""))?.Status == InferenceStatus.NotFound);
        var chatTokens = JsonNode.Parse(engine.TokenizeChat("""{"messages": [{"role": "user", "content": "Reply with ok"}]}"""))!["tokens"]!.AsArray();
        Check("a chat tokenizes to more than its text", chatTokens.Count > tokens.AsArray().Count);
        var cacheStats = JsonNode.Parse(engine.CacheStats())!["data"]!.AsArray();
        Check("cache stats list the model's encoder cache", cacheStats.Count > 0 && cacheStats[0]!["encoder_cache"] is JsonObject);
        var speculativeStats = JsonNode.Parse(engine.SpeculativeStats())!["data"]!.AsArray();
        Check("speculative stats stay zero without a proposer", speculativeStats.Count > 0 && (int)speculativeStats[0]!["drafts"]! == 0);
        var badIsq = Throws(() => engine.ReIsq("""{"ggml_type": "no-such-type"}"""));
        Check("an unknown ISQ type is InvalidRequest", badIsq?.Status == InferenceStatus.InvalidRequest);
        var unquantized = Throws(() => engine.ReIsq("""{"ggml_type": "q8_0"}"""));
        Check("requantizing a model loaded without ISQ fails", unquantized?.Status == InferenceStatus.InvalidRequest && unquantized.Detail.Contains("requantize_failed"));
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

    private static void ALoadReportsItsProgress(string model)
    {
        var events = new System.Collections.Concurrent.ConcurrentQueue<LoadProgress>();
        using (InferenceEngine.Load(Spec(model), new HostCallbacks { Progress = events.Enqueue }))
        {
        }
        Check("a load reports its progress", !events.IsEmpty);
        var ends = events.GroupBy(progress => progress.Stage).Select(stage => (stage.First(), stage.Last())).ToList();
        Check("each stage runs from none to all of its items",
            ends.All(end => end.Item1.Done == 0 && end.Item2.Done == end.Item2.Total && end.Item2.Unit == "items"));
    }

    private static byte[] ToneWav()
    {
        using var buffer = new MemoryStream();
        using var writer = new BinaryWriter(buffer);
        var samples = ToneRate;
        writer.Write("RIFF"u8.ToArray());
        writer.Write(36 + samples * 2);
        writer.Write("WAVEfmt "u8.ToArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(ToneRate);
        writer.Write(ToneRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write("data"u8.ToArray());
        writer.Write(samples * 2);
        for (var i = 0; i < samples; i++)
        {
            writer.Write((short)(0.3 * short.MaxValue * Math.Sin(2 * Math.PI * ToneHz * i / ToneRate)));
        }
        writer.Flush();
        return buffer.ToArray();
    }

    private static void AudioTranscribes(string parakeet)
    {
        var spec = new JsonObject
        {
            ["model"] = new JsonObject { ["Transcription"] = new JsonObject { ["model_id"] = parakeet, ["dtype"] = "f32" } },
            ["runtime"] = new JsonObject { ["device"] = "cpu" },
        }.ToJsonString();
        using var engine = InferenceEngine.Load(spec);
        var verbose = engine.Transcription("""{"response_format": "verbose_json", "timestamp_granularities": ["word"]}""", ToneWav());
        Check("a verbose transcript is JSON", verbose.MimeType == "application/json");
        var parsed = JsonNode.Parse(verbose.Data)!;
        Check("a verbose transcript carries its words", parsed["words"] is JsonArray);
        var text = engine.Transcription("""{"response_format": "text"}""", ToneWav());
        Check("the text transcript is the verbose one's text",
            System.Text.Encoding.UTF8.GetString(text.Data) == (string?)parsed["text"]);
        var bad = Throws(() => engine.Transcription("{}", "not audio"u8.ToArray()));
        Check("audio that does not decode is InvalidRequest", bad?.Status == InferenceStatus.InvalidRequest);
    }

    private static void StreamedAudioTranscribes(string streaming, string offline)
    {
        static string Spec(string modelId) => new JsonObject
        {
            ["model"] = new JsonObject { ["Transcription"] = new JsonObject { ["model_id"] = modelId, ["dtype"] = "f32" } },
            ["runtime"] = new JsonObject { ["device"] = "cpu" },
        }.ToJsonString();
        using (var engine = InferenceEngine.Load(Spec(streaming)))
        {
            var whole = (string?)JsonNode.Parse(engine.Transcription("""{"response_format": "json"}""", ToneWav()).Data)!["text"];
            // the WAV's samples after its 44-byte header
            var pcm = ToneWav()[44..];
            var deltas = new System.Text.StringBuilder();
            using (var session = engine.OpenTranscriptionSession($$"""{"sample_rate": {{ToneRate}}}"""))
            {
                for (var at = 0; at < pcm.Length; at += PcmPiece)
                {
                    deltas.Append(session.Push(pcm[at..Math.Min(at + PcmPiece, pcm.Length)]));
                }
                var (last, transcript) = session.Finish();
                deltas.Append(last);
                Check("a streamed transcript is JSON", transcript.MimeType == "application/json");
                Check("a streamed transcript is the whole one's",
                    (string?)JsonNode.Parse(transcript.Data)!["text"] == whole);
            }
            Check("streamed deltas add up to the transcript", deltas.ToString().Trim() == whole);
            using var odd = engine.OpenTranscriptionSession();
            Check("an odd PCM16 byte count is InvalidRequest",
                Throws(() => odd.Push(new byte[1]))?.Status == InferenceStatus.InvalidRequest);
        }
        using var wholeOnly = InferenceEngine.Load(Spec(offline));
        Check("a model that transcribes whole recordings refuses a session",
            Throws(() => wholeOnly.OpenTranscriptionSession())?.Status == InferenceStatus.InvalidRequest);
    }

    private static void VoiceActivityFindsTheSpeech(string silero)
    {
        var spec = new JsonObject
        {
            ["model"] = new JsonObject { ["VoiceActivity"] = new JsonObject { ["model_id"] = silero } },
            ["runtime"] = new JsonObject { ["device"] = "cpu" },
        }.ToJsonString();
        using var engine = InferenceEngine.Load(spec);
        // a zero threshold makes every chunk speech, so the clip is one segment whatever the weights
        var activity = JsonNode.Parse(engine.VoiceActivity("""{"threshold": 0.0, "neg_threshold": -1.0}""", ToneWav()))!;
        Check("voice activity finds one segment", (activity["segments"] as JsonArray)?.Count == 1);
    }

    private static void DiarizationNamesWhoSpeaks(string diarization)
    {
        var spec = new JsonObject
        {
            ["model"] = new JsonObject { ["Diarization"] = new JsonObject { ["model_id"] = diarization, ["dtype"] = "f32" } },
            ["runtime"] = new JsonObject { ["device"] = "cpu" },
        }.ToJsonString();
        using var engine = InferenceEngine.Load(spec);
        // a zero threshold makes every speaker speak throughout, so each has one segment whatever the weights
        var blob = engine.Diarization("""{"threshold": 0.0}""", ToneWav());
        var parsed = JsonNode.Parse(blob.Data)!;
        Check("diarization gives each speaker a segment",
            (parsed["segments"] as JsonArray)?.Count == (int?)parsed["num_speakers"]);
    }

    private static void ModelInventoryNeedsNoEngine()
    {
        var folder = Directory.CreateTempSubdirectory("inventory");
        try
        {
            var scan = JsonNode.Parse(InferenceEngine.ScanModels(new JsonObject { ["folders"] = new JsonArray(folder.FullName) }.ToJsonString()))!;
            Check("an empty folder scans to no models", scan["models"]!.AsArray().Count == 0);
            var missing = Throws(() => InferenceEngine.InspectModel(new JsonObject { ["path"] = Path.Combine(folder.FullName, "missing.gguf") }.ToJsonString()));
            Check("inspecting a missing file is InvalidRequest", missing?.Status == InferenceStatus.InvalidRequest);
            var escape = Throws(() => InferenceEngine.DeleteCachedRepo("""{"repo_id": "../escape"}"""));
            Check("deleting a path as a repository is InvalidRequest", escape?.Status == InferenceStatus.InvalidRequest);
            var badDownload = Throws(() => InferenceEngine.DownloadModel("""{"repo_id": "acme/tiny", "quant": 4}""").Dispose());
            Check("a malformed download is InvalidRequest", badDownload?.Status == InferenceStatus.InvalidRequest);
        }
        finally
        {
            folder.Delete(recursive: true);
        }
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
