#!/usr/bin/env python3
"""Parity of our Whisper against OpenAI's `whisper` package: the transcript must match exactly and every word's start
and end must fall within a tolerance of OpenAI's (greedy, no previous-text conditioning, word timestamps on).

Not part of local_ci.sh: it needs a CUDA `inference` build, OpenAI's `whisper` package (`pip install openai-whisper`,
which downloads its own copy of the weights) and the checkpoint in the Hugging Face cache.

Usage: python3 scripts/whisper_parity.py <hf model id> <openai model name> <wav>... [--inference <binary>]
           [--max-seconds 0.04] [--port 18499] [--fp16 | --fp32]
  e.g. scripts/whisper_parity.py openai/whisper-large-v3-turbo turbo clip.wav long.wav
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

# two encoder positions: precision moves DTW word boundaries by a frame or two
DEFAULT_MAX_SECONDS = 0.04
STARTUP_SECONDS = 180


def ours(port: int, wav: str) -> dict:
    out = subprocess.run(
        [
            "curl", "-s", f"localhost:{port}/v1/audio/transcriptions", "-F", f"file=@{wav}", "-F", "model=default",
            "-F", "response_format=verbose_json", "-F", "timestamp_granularities[]=word",
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    body = json.loads(out)
    return {"text": body["text"], "words": [(w["word"], w["start"], w["end"]) for w in body.get("words") or []]}


def reference(model, wav: str, fp16: bool) -> dict:
    result = model.transcribe(wav, word_timestamps=True, temperature=0.0, condition_on_previous_text=False, fp16=fp16)
    words = [(w["word"].strip(), w["start"], w["end"]) for s in result["segments"] for w in s["words"]]
    return {"text": result["text"].strip(), "words": words}


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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("openai_model")
    parser.add_argument("wavs", nargs="+")
    parser.add_argument("--inference", default="target/debug/inference")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--port", type=int, default=18499)
    parser.add_argument("--fp32", action="store_true", help="both sides in F32 (default F16, OpenAI's on CUDA)")
    args = parser.parse_args()
    import whisper

    dtype = "f32" if args.fp32 else "f16"
    server = subprocess.Popen(
        [args.inference, "serve", "-p", str(args.port), "--no-ui", "-m", args.model, "--dtype", dtype],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    failed = False
    try:
        wait_ready(args.port, server)
        model = whisper.load_model(args.openai_model, device="cuda")
        for wav in args.wavs:
            ref, got = reference(model, wav, not args.fp32), ours(args.port, wav)
            problems = []
            if ref["text"] != got["text"]:
                problems.append(f"text\n  ref : {ref['text']}\n  ours: {got['text']}")
            elif [w[0] for w in ref["words"]] != [w[0] for w in got["words"]]:
                problems.append("word lists differ")
            else:
                deltas = [max(abs(a[1] - b[1]), abs(a[2] - b[2])) for a, b in zip(ref["words"], got["words"])]
                worst = max(deltas, default=0.0)
                print(f"{wav}: {len(deltas)} words, max |dt| {worst:.3f} s")
                problems += [
                    f"{a} vs {b}" for a, b, d in zip(ref["words"], got["words"], deltas) if d > args.max_seconds + 1e-9
                ]
            if problems:
                failed = True
                print(f"{wav}: MISMATCH")
                for problem in problems:
                    print(f"  {problem}")
    finally:
        server.terminate()
        server.wait()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
