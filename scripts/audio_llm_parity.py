#!/usr/bin/env python3
"""Parity of our audio-in decoders (Voxtral, Qwen2-Audio, ...) against transformers on the same checkpoint: greedy
answers to a question about each clip and to a text-only prompt, and, for a model with a transcription mode (Voxtral, GLM-ASR),
each clip's transcript (`--language`, Voxtral's only), must match exactly.

Not part of local_ci.sh: it needs an `inference` build, transformers (with mistral_common for Voxtral, whose processor
reads the tekken tokenizer through it) and the checkpoint in the Hugging Face cache. Both run in F32, on the CPU unless
`--cuda` is given.

Usage: python3 scripts/audio_llm_parity.py <hf model id> <wav>... [--inference <binary>] [--port 18498]
           [--question "..."] [--max-tokens 48] [--language en] [--cuda]
  e.g. scripts/audio_llm_parity.py mistralai/Voxtral-Mini-3B-2507 clip.wav dialogue.wav
       scripts/audio_llm_parity.py Qwen/Qwen2-Audio-7B-Instruct clip.wav
"""

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request

DEFAULT_QUESTION = "What is said in this recording? Answer in two sentences."
TEXT_PROMPT = "Name three rivers in Europe and one fact about each."
STARTUP_SECONDS = 600


def ours(port: int, content) -> str:
    body = {
        "model": "default",
        "temperature": 0,
        "max_tokens": ARGS.max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()


def our_transcript(port: int, wav: str) -> str:
    out = subprocess.run(
        [
            "curl",
            "-s",
            f"localhost:{port}/v1/audio/transcriptions",
            "-F",
            f"file=@{wav}",
            "-F",
            "model=default",
            "-F",
            f"language={ARGS.language}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)["text"]


def reference_transcript(model, processor, wav: str) -> str:
    if model.config.model_type == "voxtral":
        inputs = processor.apply_transcription_request(
            language=ARGS.language, audio=wav, model_id=ARGS.model
        )
    else:
        inputs = processor.apply_transcription_request(wav)
    inputs = inputs.to(model.device, dtype=model.dtype)
    output = model.generate(
        **inputs, max_new_tokens=ARGS.max_tokens * 4, do_sample=False
    )
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()


def reference(model, processor, content) -> str:
    conversation = [{"role": "user", "content": content}]
    if model.config.model_type == "voxtral":
        inputs = processor.apply_chat_template(conversation, return_tensors="pt")
    else:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    inputs = inputs.to(model.device)
    output = model.generate(**inputs, max_new_tokens=ARGS.max_tokens, do_sample=False)
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()


def wait_ready(port: int, server: subprocess.Popen) -> None:
    deadline = time.time() + STARTUP_SECONDS
    while time.time() < deadline:
        if server.poll() is not None:
            sys.exit(f"inference serve exited with {server.returncode}")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return
        except OSError:
            time.sleep(2)
    sys.exit("inference serve did not come up")


def main() -> None:
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    device = ["--cpu"] if not ARGS.cuda else []
    server = subprocess.Popen(
        [
            ARGS.inference,
            "serve",
            *device,
            "-p",
            str(ARGS.port),
            "--no-ui",
            "-m",
            ARGS.model,
            "--dtype",
            "f32",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    failed = False
    try:
        wait_ready(ARGS.port, server)
        processor = AutoProcessor.from_pretrained(ARGS.model)
        architecture = AutoConfig.from_pretrained(ARGS.model).architectures[0]
        model = getattr(transformers, architecture).from_pretrained(
            ARGS.model, dtype=torch.float32
        )
        model = model.to("cuda" if ARGS.cuda else "cpu")
        cases = [("text", [{"type": "text", "text": TEXT_PROMPT}], TEXT_PROMPT)]
        for wav in ARGS.wavs:
            with open(wav, "rb") as file:
                data = base64.b64encode(file.read()).decode()
            cases.append(
                (
                    wav,
                    [
                        {"type": "audio", "path": wav},
                        {"type": "text", "text": ARGS.question},
                    ],
                    [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data, "format": "wav"},
                        },
                        {"type": "text", "text": ARGS.question},
                    ],
                )
            )
        results = [
            (
                name,
                reference(model, processor, ref_content),
                ours(ARGS.port, our_content),
            )
            for name, ref_content, our_content in cases
        ]
        if hasattr(processor, "apply_transcription_request"):
            results += [
                (
                    f"{wav} (transcript)",
                    reference_transcript(model, processor, wav),
                    our_transcript(ARGS.port, wav),
                )
                for wav in ARGS.wavs
            ]
        for name, ref, got in results:
            if ref == got:
                print(f"{name}: match ({len(got)} chars)")
            else:
                failed = True
                print(f"{name}: MISMATCH\n  ref : {ref}\n  ours: {got}")
    finally:
        server.terminate()
        server.wait()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("wavs", nargs="+")
    parser.add_argument("--inference", default="target/debug/inference")
    parser.add_argument("--port", type=int, default=18498)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--language", default="en", help="the transcription language (Voxtral)"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="run both on the GPU (default: the CPU)"
    )
    ARGS = parser.parse_args()
    main()
