"""The typed engine on the tiny checkpoint: real responses and stream events parse into the generated classes."""

import dataclasses
import json
import os
import sys
import typing
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_types

import inference_rs as ir
from inference_rs import types as t
from tests.test_engine import MAX_TOKENS, MODEL_VARIABLE, PROMPT, spec


def chat_request(stream: bool = False) -> t.ChatCompletionRequest:
    return t.ChatCompletionRequest(
        model="default",
        messages=[t.Message(role="user", content=PROMPT)],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_k=1,
        stream=stream,
    )


def strip_nulls(value):
    if isinstance(value, dict):
        return {key: strip_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [strip_nulls(item) for item in value]
    return value


def untyped(value, path="value"):
    """Paths where a dict stands in for a class the schema names, which means the data did not parse."""
    found = []
    if dataclasses.is_dataclass(value):
        hints = typing.get_type_hints(type(value), vars(t))
        for f in dataclasses.fields(value):
            item = getattr(value, f.name)
            expects = [a for a in typing.get_args(hints[f.name]) or (hints[f.name],) if dataclasses.is_dataclass(a)]
            if isinstance(item, dict) and expects:
                found.append(f"{path}.{f.name}")
            found += untyped(item, f"{path}.{f.name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += untyped(item, f"{path}[{index}]")
    return found


class Generated(unittest.TestCase):
    def test_types_match_the_openapi_document(self):
        self.assertEqual(
            generate_types.generate(),
            generate_types.OUTPUT.read_text(),
            "inference_rs/types.py is stale; run bindings/python/scripts/generate_types.py",
        )

    def test_requests_serialize_without_unset_fields(self):
        request = t.ChatCompletionRequest(
            messages=[t.Message(role="user", content="hi")],
            response_format=t.ResponseFormatJsonObject(),
            reasoning_effort=t.ReasoningEffort.LOW,
        )
        self.assertEqual(
            ir.to_data(request),
            {
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
            },
        )

    def test_newer_fields_and_values_inside_optional_objects_still_parse(self):
        data = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "a_newer_field": 1,
            },
            "incomplete_details": {"reason": "a_newer_reason"},
        }
        resource = ir.from_data(t.ResponseResource, data)
        self.assertIsInstance(resource.usage, t.ResponseUsage)
        self.assertIsInstance(resource.incomplete_details, t.IncompleteDetails)
        self.assertEqual(resource.incomplete_details.reason, "a_newer_reason")

    def test_renamed_fields_use_their_wire_names_both_ways(self):
        @dataclasses.dataclass(kw_only=True)
        class Renamed:
            from_: str
            _wire: typing.ClassVar = {"from_": "from"}

        self.assertEqual(ir.to_data(Renamed(from_="a")), {"from": "a"})
        self.assertEqual(ir.from_data(Renamed, {"from": "a"}), Renamed(from_="a"))

    def test_externally_tagged_variants_wrap_and_unwrap(self):
        spec = t.EngineSpec(model=t.ModelSelectedPlain(model_id="org/model"), runtime=t.RuntimeSpec(device="cpu"))
        # The engine's defaults are the fields' defaults, so they are written out.
        plain = {"model_id": "org/model", "dtype": "auto", "max_seq_len": 4096, "max_batch_size": 1}
        data = {"model": {"Plain": plain}, "runtime": {"device": "cpu"}}
        self.assertEqual(ir.to_data(spec), data)
        self.assertEqual(ir.from_data(t.EngineSpec, data), spec)
        self.assertEqual(json.loads(ir.to_json(spec)), data)
        self.assertEqual(json.loads(ir.to_json({"model": spec.model})), {"model": data["model"]})
        # An unknown or padded tag names no variant, so it stays as it came; exactly, it fits nothing.
        for model in ({"NoSuchVariant": {"model_id": "m"}}, {"Plain": {"model_id": "m"}, "extra": 1}):
            self.assertEqual(ir.from_data(t.ModelSelected, model), model)
            with self.assertRaises(ValueError):
                ir.from_data(t.ModelSelectedPlain, model, strict=True)

    def test_loader_enums_carry_the_names_the_engine_accepts(self):
        self.assertEqual(t.NormalLoaderType.QWEN3.value, "qwen3")
        spec = t.EngineSpec(model=t.ModelSelectedPlain(model_id="m", arch=t.NormalLoaderType.QWEN3))
        self.assertEqual(ir.to_data(spec)["model"]["Plain"]["arch"], "qwen3")

    def test_a_gguf_quant_leaves_the_filename_to_the_engine(self):
        model = t.ModelSelectedGGUF(
            quantized_model_id="org/model-GGUF", quant="4", mmproj_selection=t.MmprojSelection.ARTIFACT_REPO
        )
        data = ir.to_data(model)["GGUF"]
        self.assertNotIn("quantized_filename", data)
        self.assertEqual((data["quant"], data["mmproj_selection"]), ("4", "artifact_repo"))

    def test_a_dict_spec_loads_like_its_class(self):
        data = json.loads(spec("m"))
        written = ir.to_data(ir.from_data(t.EngineSpec, data))
        model = written["model"]["MultimodalPlain"]
        defaults = {"max_seq_len": 4096, "max_batch_size": 1, "max_num_images": 1, "max_image_length": 1024}
        self.assertEqual({key: model.pop(key) for key in defaults}, defaults)
        self.assertEqual(written, data)

    def test_tags_default_to_their_value(self):
        self.assertEqual(
            ir.to_data(t.Tool(function=t.Function(name="f"))),
            {"function": {"name": "f"}, "type": "function"},
        )

    def test_untagged_unions_pick_the_variant_that_fits(self):
        schema = {"type": "json_schema", "json_schema": {"name": "n", "schema": {}}}
        self.assertIsInstance(ir.from_data(t.ResponseFormat, schema), t.ResponseFormatJsonSchema)
        self.assertEqual(ir.from_data(t.StopTokens, ["a", "b"]), ["a", "b"])


class TypedEngine(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        model = os.environ.get(MODEL_VARIABLE)
        if not model:
            raise unittest.SkipTest(f"{MODEL_VARIABLE} is not set")
        cls.engine = ir.Engine(
            t.EngineSpec(
                model=t.ModelSelectedMultimodalPlain(model_id=model, dtype=t.ModelDType.F32),
                runtime=t.RuntimeSpec(device="cpu"),
            )
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.close()

    def assert_parsed(self, annotation, text: str):
        """`text` parses into `annotation` with nothing left as a dict, and serializes back to itself."""
        parsed = ir.from_json(annotation, text)
        self.assertEqual(untyped(parsed), [])
        self.assertEqual(ir.to_data(parsed), strip_nulls(json.loads(text)))
        return parsed

    def test_real_payloads_round_trip(self):
        raw = self.engine.json
        self.assert_parsed(t.ChatCompletionResponse, raw.chat(ir.to_json(chat_request())))
        with raw.chat_stream(ir.to_json(chat_request(stream=True))) as stream:
            for event in stream:
                self.assert_parsed(t.ChatCompletionChunkResponse, json.dumps(event.data))
        request = t.OpenResponsesCreateRequest(
            model="default",
            input=PROMPT,
            max_output_tokens=MAX_TOKENS,
            temperature=0.0,
            top_k=1,
            stream=True,
        )
        with raw.response_stream(ir.to_json(request)) as stream:
            for event in stream:
                self.assert_parsed(t.OpenResponsesStreamEvent, json.dumps(event.data))
        self.assert_parsed(t.ModelObjects, raw.list_models())
        self.assert_parsed(t.FileMetadata, raw.upload_file(b"x", "x.txt", "user_data"))
        self.assert_parsed(t.FileListObject, raw.list_files())

    def test_other_protocols_and_management_are_typed(self):
        completion = self.engine.completion(
            t.CompletionRequest(
                model="default",
                prompt=PROMPT,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                top_k=1,
            )
        )
        self.assertIsInstance(completion.choices[0], t.CompletionResponseChoice)
        message = self.engine.anthropic_messages(
            t.AnthropicMessagesRequest(
                model="default",
                max_tokens=MAX_TOKENS,
                messages=[t.AnthropicMessage(role="user", content=PROMPT)],
            )
        )
        self.assertIsInstance(message, t.AnthropicMessageResponse)
        model_id = next(m.id for m in self.engine.list_models().data if m.id != "default")
        self.assertEqual(self.engine.unload_model(model_id).status, t.ModelStatus.UNLOADED)
        self.assertEqual(self.engine.reload_model(model_id).status, t.ModelStatus.LOADED)
        skill_md = b"---\nname: typed-skill\ndescription: A typed upload.\n---\n"
        skill = self.engine.upload_skill([ir.SkillFile("SKILL.md", skill_md)])
        self.assertIsInstance(skill, t.SkillObject)
        self.assertIn(skill.id, [s.id for s in self.engine.list_skills().data])
        self.assertIsInstance(self.engine.list_skill_versions(skill.id), t.AnthropicSkillVersionListObject)
        deleted = self.engine.delete_file(self.engine.upload_file(b"x", "x.txt", "user_data").id)
        self.assertIsInstance(deleted, t.FileDeleted)

    def test_chat_parses_and_agrees_with_its_stream(self):
        response = self.engine.chat(chat_request())
        self.assertIsInstance(response, t.ChatCompletionResponse)
        self.assertGreater(response.usage.completion_tokens, 0)
        text = response.choices[0].message.content or ""
        streamed = ""
        with self.engine.chat_stream(chat_request(stream=True)) as stream:
            for event in stream:
                self.assertIsInstance(event.data, t.ChatCompletionChunkResponse)
                streamed += event.data.choices[0].delta.content or ""
        self.assertEqual(streamed, text)

    def test_responses_stream_events_are_their_variants(self):
        request = t.OpenResponsesCreateRequest(
            model="default",
            input=PROMPT,
            max_output_tokens=MAX_TOKENS,
            temperature=0.0,
            top_k=1,
            stream=True,
        )
        with self.engine.response_stream(request) as stream:
            events = list(stream)
        self.assertIsInstance(events[0].data, t.OpenResponsesStreamEventResponseCreated)
        # The random weights never stop on their own, so the token cap ends the run.
        capped = events[-1].data
        self.assertIsInstance(capped, t.OpenResponsesStreamEventResponseIncomplete)
        self.assertIsInstance(capped.response, t.ResponseResource)
        stored = self.engine.get_response(capped.response.id)
        self.assertEqual(stored.status, t.ResponseStatus.INCOMPLETE)

    def test_management_calls_return_their_classes(self):
        models = self.engine.list_models()
        self.assertEqual(models.data[0].id, "default")
        status = self.engine.model_status(next(m.id for m in models.data if m.id != "default"))
        self.assertEqual(status.status, t.ModelStatus.LOADED)
        stats = self.engine.cache_stats()
        self.assertIsInstance(stats, t.CacheStats)
        self.assertIsInstance(stats.data[0].encoder_cache, t.EncoderCacheStats)
        uploaded = self.engine.upload_file(b"a,b\n", "table.csv", "user_data", "text/csv")
        self.assertIsInstance(uploaded, t.FileMetadata)
        self.assertIn(uploaded.id, [f.id for f in self.engine.list_files().data])
        with self.assertRaises(ir.InferenceError) as unknown:
            self.engine.model_status("no-such-model")
        self.assertEqual(unknown.exception.status, ir.Status.NOT_FOUND)
        self.assertEqual(sum(1 for m in models.data if m.default), 1)
        self.assertTrue(self.engine.model_served("default"))
        self.assertFalse(self.engine.model_served("no-such-model"))
        self.assertEqual(self.engine.list_mcp_tools().data, [])
        request = {"model": "default", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok"}]}
        self.assertGreater(self.engine.anthropic_count_tokens(json.dumps(request)).input_tokens, 0)
        self.assertEqual(self.engine.list_container_files("cntr_unused").data, [])
        for call in (self.engine.get_container_file, self.engine.container_file_content):
            with self.assertRaises(ir.InferenceError) as outside:
                call("cntr_unused", uploaded.id)
            self.assertEqual(outside.exception.status, ir.Status.NOT_FOUND)
        with self.assertRaises(ir.InferenceError) as bad_tune:
            ir.tune_model(json.dumps({"model_id": "org/model", "dtype": "no-such-dtype"}))
        self.assertEqual(bad_tune.exception.status, ir.Status.INVALID_REQUEST)

    def test_an_owner_reaches_only_what_it_stored(self):
        with self.engine.for_owner("team-a") as team_a, self.engine.for_owner("team-b") as team_b:
            uploaded = team_a.upload_file(b"a,b\n", "table.csv", "user_data", "text/csv")
            self.assertIn(uploaded.id, [f.id for f in team_a.list_files().data])
            for other in (team_b, self.engine):
                self.assertNotIn(uploaded.id, [f.id for f in other.list_files().data])
                with self.assertRaises(ir.InferenceError) as hidden:
                    other.get_file(uploaded.id)
                self.assertEqual(hidden.exception.status, ir.Status.NOT_FOUND)
            session = t.SerializedSession(messages=[{"role": {"Left": "user"}, "content": {"Left": "hi"}}])
            team_a.put_session("owned-session", session)
            self.assertNotIn("owned-session", team_b.list_sessions().data)
            with self.assertRaises(ir.InferenceError):
                team_b.put_session("owned-session", session)
            self.assertTrue(team_a.delete_session("owned-session").deleted)

    def test_models_are_added_and_removed_at_runtime(self):
        model = os.environ[MODEL_VARIABLE]
        selected = t.ModelSelectedMultimodalPlain(model_id=model, dtype=t.ModelDType.F32)
        spec = t.EngineSpec(model=selected, runtime=t.RuntimeSpec(device="cpu"))
        with ir.Engine(spec) as engine:
            added = engine.add_model(t.ModelSpec(model=selected, model_id="second"))
            self.assertEqual((added.model_id, added.status), ("second", t.ModelStatus.LOADED))
            self.assertEqual(engine.set_default_model("second").model_id, "second")
            self.assertEqual(next(m.id for m in engine.list_models().data if m.default), "second")
            self.assertEqual(engine.add_model_alias("spare", "second").alias, "spare")
            self.assertTrue(engine.model_served("spare"))
            self.assertEqual(engine.remove_model("second").model_id, "second")
            self.assertFalse(engine.model_served("second"))

    def test_a_prompt_is_scored(self):
        scores, logits = self.engine.prompt_logits("Reply with ok")
        self.assertIsInstance(scores, t.PromptLogits)
        self.assertIsNone(logits)
        self.assertIsNone(scores.token_logprobs[0])
        self.assertTrue(all(p <= 0.0 for p in scores.token_logprobs[1:]))
        _, logits = self.engine.prompt_logits(scores.tokens, output="logits")
        self.assertEqual(len(logits), len(scores.tokens) * scores.vocab_size)

    def test_a_registered_logits_processor_steers_the_requests_that_name_it(self):
        steps = []

        def force_last_token(logits, context):
            steps.append(len(context))
            for index in range(len(logits)):
                logits[index] = float("-inf")
            logits[len(logits) - 1] = 0.0

        request = chat_request()
        request.logits_processors = ["py-forced"]
        with self.engine.register_logits_processor("py-forced", force_last_token):
            with self.assertRaises(ir.InferenceError) as again:
                self.engine.register_logits_processor("py-forced", force_last_token)
            self.assertEqual(again.exception.status, ir.Status.INVALID_REQUEST)
            response = self.engine.chat(request)
        self.assertEqual(len(steps), response.usage.completion_tokens)
        self.assertTrue(all(a < b for a, b in zip(steps, steps[1:])), steps)
        with self.assertRaises(ir.InferenceError) as unknown:
            self.engine.chat(request)
        self.assertEqual(unknown.exception.status, ir.Status.INVALID_REQUEST)

        late = ir.HostTool(
            json.dumps({"type": "function", "function": {"name": "py_late", "parameters": {"type": "object"}}}),
            lambda call: "found",
        )
        with self.engine.register_tool(late):
            with self.assertRaises(ir.InferenceError) as twice:
                self.engine.register_tool(late)
            self.assertEqual(twice.exception.status, ir.Status.INVALID_REQUEST)
        named = chat_request()
        named.host_tools = ["py_late"]
        with self.assertRaises(ir.InferenceError) as gone:
            self.engine.chat(named)
        self.assertEqual(gone.exception.status, ir.Status.INVALID_REQUEST)

        # A registration outlives the handle it came through, and closing it afterwards still unregisters the name.
        scoped = self.engine.json.for_owner("py-processor-owner")
        registration = scoped.register_logits_processor("py-scoped", force_last_token)
        scoped.close()
        registration.close()
        self.engine.register_logits_processor("py-scoped", force_last_token).close()

    def test_runtime_operations_are_typed(self):
        tokens = self.engine.tokenize("Reply with ok")
        self.assertTrue(tokens and all(isinstance(token, int) for token in tokens))
        # The tiny tokenizer has no decoder, so its word-boundary markers come back as they are.
        self.assertEqual(self.engine.detokenize(tokens).replace("\u2581", " "), "Reply with ok")
        session = t.SerializedSession(messages=[{"role": {"Left": "user"}, "content": {"Left": "hi"}}])
        self.assertEqual(self.engine.put_session("typed-session", session).id, "typed-session")
        self.assertIn("typed-session", self.engine.list_sessions().data)
        self.assertEqual(self.engine.get_session("typed-session").messages, session.messages)
        fork = self.engine.fork_session("typed-session", 0).id
        self.assertNotEqual(fork, "typed-session")
        self.assertIsInstance(self.engine.get_session(fork), t.SerializedSession)
        self.engine.delete_session(fork)
        with self.assertRaises(ir.InferenceError) as unknown_fork:
            self.engine.fork_session("no-such-session", 0)
        self.assertEqual(unknown_fork.exception.status, ir.Status.INVALID_REQUEST)
        self.assertTrue(self.engine.delete_session("typed-session").deleted)
        with self.assertRaises(ir.InferenceError) as gone:
            self.engine.get_session("typed-session")
        self.assertEqual(gone.exception.status, ir.Status.NOT_FOUND)
        self.assertIsInstance(self.engine.calibration_status(), t.CalibrationStatus)
        with self.assertRaises(ir.InferenceError):
            self.engine.re_isq("no-such-type")
        models = self.engine.list_models().data
        self.assertGreater(next(m for m in models if m.id != "default").max_model_len, 0)


if __name__ == "__main__":
    unittest.main()
