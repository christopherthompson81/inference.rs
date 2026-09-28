"""The engine through the package, on the tiny random-weight checkpoint that
`cargo run -p inference-ffi --example tiny_checkpoint -- DIR` writes (INFERENCE_TEST_TINY_CHECKPOINT)."""

import base64
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_rs as ir

MODEL_VARIABLE = "INFERENCE_TEST_TINY_CHECKPOINT"
MAX_TOKENS = 6
PROMPT = "Reply with the single word: ok"
POLL_TIMEOUT = 60.0
IMAGE_FIXTURE = "crates/inference/tests/fixtures/paddleocr_vl/page_00.png"
SKILL_MD = (
    "---\nname: csv-summary\ndescription: Summarizes a CSV file.\n---\nRead the file.\n"
)


def spec(model: str) -> str:
    return json.dumps(
        {
            "model": {"MultimodalPlain": {"model_id": model, "dtype": "f32"}},
            "runtime": {"device": "cpu"},
        }
    )


def chat_request(stream: bool = False, content=PROMPT) -> str:
    return json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "stream": stream,
        }
    )


def find_in_checkout(relative: str) -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / relative).is_file():
            return parent / relative
    raise FileNotFoundError(relative)


class EngineTest(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(MODEL_VARIABLE)
        if not cls.model:
            raise unittest.SkipTest(f"{MODEL_VARIABLE} is not set")
        cls.engine = ir.Engine(spec(cls.model))

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.close()

    def test_chat_and_stream_agree(self):
        text = (
            json.loads(self.engine.chat(chat_request()))["choices"][0]["message"][
                "content"
            ]
            or ""
        )
        streamed = ""
        with self.engine.chat_stream(chat_request(stream=True)) as stream:
            for event in stream:
                self.assertEqual(event.name, "chunk")
                streamed += event.data["choices"][0]["delta"].get("content") or ""
            self.assertTrue(stream.done)
            self.assertIsNone(stream.next(0))
        self.assertEqual(streamed, text)

    def test_an_attached_image_decodes_like_a_data_url(self):
        png = find_in_checkout(IMAGE_FIXTURE).read_bytes()

        def request(url):
            return chat_request(
                content=[
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "OCR:"},
                ]
            )

        by_url = json.loads(
            self.engine.chat(
                request("data:image/png;base64," + base64.b64encode(png).decode())
            )
        )
        by_media = json.loads(
            self.engine.chat(
                request("media://0"), [ir.MediaAttachment(png, "image/png")]
            )
        )
        self.assertEqual(
            by_url["choices"][0]["message"]["content"],
            by_media["choices"][0]["message"]["content"],
        )

    def test_other_protocols(self):
        completion = json.dumps(
            {
                "model": "default",
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "top_k": 1,
            }
        )
        self.assertEqual(
            json.loads(self.engine.completion(completion))["object"], "text_completion"
        )
        with self.engine.completion_stream(completion) as stream:
            self.assertTrue(all(event.name == "chunk" for event in stream))

        messages = json.dumps(
            {
                "model": "default",
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "top_k": 1,
                "messages": [{"role": "user", "content": PROMPT}],
            }
        )
        self.assertEqual(
            json.loads(self.engine.anthropic_messages(messages))["type"], "message"
        )
        with self.engine.anthropic_messages_stream(messages) as stream:
            names = []
            while (event := stream.next(POLL_TIMEOUT)) is not None:
                names.append(event.name)
        self.assertEqual((names[0], names[-1]), ("message_start", "message_stop"))

        request = {
            "model": "default",
            "input": PROMPT,
            "max_output_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
        }
        response_id = json.loads(self.engine.create_response(json.dumps(request)))["id"]
        self.assertEqual(
            json.loads(self.engine.get_response(response_id))["id"], response_id
        )
        self.engine.delete_response(response_id)
        with self.assertRaises(ir.InferenceError) as missing:
            self.engine.get_response(response_id)
        self.assertEqual(missing.exception.status, ir.Status.NOT_FOUND)
        with self.engine.response_stream(
            json.dumps({**request, "stream": True})
        ) as stream:
            self.assertEqual(list(stream)[-1].name, "response.completed")

    def test_errors_carry_the_envelope(self):
        with self.assertRaises(ir.InferenceError) as unknown:
            self.engine.chat(
                json.dumps(
                    {
                        "model": "no-such-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            )
        self.assertEqual(
            (unknown.exception.status, unknown.exception.code),
            (ir.Status.NOT_FOUND, "model_not_found"),
        )
        with self.assertRaises(ir.InferenceError) as malformed:
            self.engine.chat("{not json")
        self.assertEqual(malformed.exception.status, ir.Status.INVALID_REQUEST)
        self.assertEqual(
            json.loads(self.engine.list_models())["data"][0]["id"], "default"
        )
        with self.assertRaises(ir.InferenceError) as approval:
            self.engine.resolve_approval(
                "never-issued", json.dumps({"decision": "approve"})
            )
        self.assertEqual(approval.exception.status, ir.Status.NOT_FOUND)

    def test_files_round_trip(self):
        contents = b"col_a,col_b\n1,2\n"
        file_id = json.loads(
            self.engine.upload_file(contents, "table.csv", "user_data", "text/csv")
        )["id"]
        self.assertEqual(
            self.engine.file_content(file_id), ir.Blob(contents, "text/csv")
        )
        self.engine.delete_file(file_id)
        with self.assertRaises(ir.InferenceError) as missing:
            self.engine.file_content(file_id)
        self.assertEqual(missing.exception.status, ir.Status.NOT_FOUND)
        self.assertIn(
            "id", json.loads(self.engine.upload_file(b"", "empty.txt", "user_data"))
        )

    def test_skills_are_stored(self):
        skill = json.loads(
            self.engine.upload_skill([ir.SkillFile("SKILL.md", SKILL_MD.encode())])
        )
        self.assertEqual(skill["name"], "csv-summary")
        self.assertEqual(len(json.loads(self.engine.list_skills())["data"]), 1)
        with self.assertRaises(ir.InferenceError) as bad:
            self.engine.upload_skill([ir.SkillFile("SKILL.md", b"no frontmatter")])
        self.assertEqual(bad.exception.status, ir.Status.INVALID_REQUEST)

    def test_a_stream_outlives_its_engine(self):
        engine = ir.Engine(spec(self.model))
        stream = engine.chat_stream(chat_request(stream=True))
        engine.close()
        self.assertGreater(len(list(stream)), 0)
        self.assertTrue(stream.done)
        stream.close()
        with self.assertRaises(ValueError):
            engine.chat(chat_request())

    def test_host_tools_load_and_malformed_ones_are_refused(self):
        definition = json.dumps(
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        )
        callbacks = ir.HostCallbacks(
            tools=[ir.HostTool(definition, lambda call: call.arguments_json)],
            search=lambda query: "[]",
        )
        with ir.Engine(spec(self.model), callbacks) as engine:
            self.assertIn("choices", json.loads(engine.chat(chat_request())))
        bad = ir.HostCallbacks(tools=[ir.HostTool("not json", lambda call: "")])
        with self.assertRaises(ir.InferenceError) as refused:
            ir.Engine(spec(self.model), bad)
        self.assertEqual(refused.exception.status, ir.Status.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
