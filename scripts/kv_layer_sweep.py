#!/usr/bin/env python3
"""Measure how sensitive each layer's KV cache is to low bit widths, and print a KV cache profile for the model.

Each layer in turn stores K and V at the probe type (default `q3/q8`: 3-bit K, 8-bit V) while the rest stay at q8;
the cost is teacher-forced perplexity as decoding sees the cache (see kv_cache_quant_investigation.md, Run 6). A
layer whose cost passes the threshold gets a floor of 8 bits in the profile, which goes in
crates/inference-nn/src/paged_attention/kv_profiles.json.

Needs a running `inference serve` with its model manager, admin access, the model's `tokenizer.json` and the Python
`tokenizers` package. A model loads once per layer, under the model id `kv-sweep`.

Usage: kv_layer_sweep.py --model <source> --tokenizer <tokenizer.json> --text <file> --config <config.json>
       [--server http://127.0.0.1:1234] [--layers 0-27] [--probe q3/q8] [--threshold 0.01] [--name "Model"]
"""

import argparse
import json
import math
import sys
import urllib.request

MODEL_ID = "kv-sweep"
CONTEXT = 16384
PASSAGES = 10
PREFIX = 256
SCORED = 96
TOP = 20
BASE = "q8"
PROTECTED = {"k": 8, "v": 8}


def post(server, path, body):
    request = urllib.request.Request(server + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(request, timeout=3600)


def perplexity(args, passages, layers):
    """Loads the model with `layers` over the q8 base, then scores every passage's tail one token at a time."""
    body = {"source": args.model, "model_id": MODEL_ID, "paged_cache_type": BASE, "paged_context_len": CONTEXT}
    if layers:
        body["paged_cache_layers"] = layers
    events = post(args.server, "/v1/manager/load", body).read().decode()
    if "event: loaded" not in events:
        sys.exit(f"the load failed: {events[-400:]}")
    logprobs = []
    for passage in passages:
        for i in range(PREFIX, PREFIX + SCORED):
            # the prompt repeats the previous one plus a token, so prefix caching serves it from the cache under test
            out = json.load(post(args.server, "/v1/completions", {
                "model": MODEL_ID, "prompt": passage[:i], "max_tokens": 1, "temperature": 0, "logprobs": TOP,
            }))
            top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            found = [entry["logprob"] for entry in top if entry["token"] == passage[i]]
            logprobs.append(found[0] if found else min(entry["logprob"] for entry in top))
    return math.exp(-sum(logprobs) / len(logprobs))


def layer_list(spec, count):
    if not spec:
        return list(range(count))
    layers = []
    for item in spec.split(","):
        first, _, last = item.partition("-")
        layers.extend(range(int(first), int(last or first) + 1))
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="the source `/v1/manager/load` takes")
    parser.add_argument("--tokenizer", required=True, help="the model's tokenizer.json")
    parser.add_argument("--text", required=True, help="text to score, a few thousand words")
    parser.add_argument("--config", required=True, help="the model's config.json, for the profile's fingerprint")
    parser.add_argument("--server", default="http://127.0.0.1:1234")
    parser.add_argument("--layers", help="layers to probe, such as 0-27 or 3,7,11 (default: all)")
    parser.add_argument("--probe", default="q3/q8", help="the type a probed layer stores")
    parser.add_argument("--threshold", type=float, default=0.01, help="relative cost that earns a layer a floor")
    parser.add_argument("--name", help="the profile's name (default: the model source)")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    config = json.load(open(args.config))
    text_config = config.get("text_config", config)
    heads = text_config["num_attention_heads"]
    fingerprint = {
        "layers": text_config["num_hidden_layers"],
        "hidden": text_config["hidden_size"],
        "heads": heads,
        "kv_heads": text_config.get("num_key_value_heads", heads),
        "head_dim": text_config.get("head_dim") or text_config["hidden_size"] // heads,
    }
    ids = Tokenizer.from_file(args.tokenizer).encode(open(args.text).read()).ids
    step = (len(ids) - PREFIX - SCORED) // PASSAGES
    if step <= 0:
        sys.exit(f"the text has {len(ids)} tokens; it needs more than {PREFIX + SCORED + PASSAGES}")
    passages = [ids[p * step : p * step + PREFIX + SCORED] for p in range(PASSAGES)]

    baseline = perplexity(args, passages, None)
    print(f"all layers {BASE}: {baseline:.4f}", flush=True)
    floors = {}
    for layer in layer_list(args.layers, fingerprint["layers"]):
        cost = perplexity(args, passages, f"{layer}={args.probe}") / baseline - 1
        print(f"layer {layer:3} at {args.probe}: {100 * cost:+6.2f}%", flush=True)
        if cost > args.threshold:
            floors[str(layer)] = PROTECTED
    profile = {"name": args.name or args.model, "fingerprint": fingerprint, "floors": floors}
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
