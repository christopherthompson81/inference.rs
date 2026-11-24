#!/usr/bin/env bash
# One-off parity check of our control vectors against llama.cpp's: a vector llama.cpp's `cvector-generator` writes for
# a GGUF Llama (mean method, happy against sad), then the same greedy raw completion from llama.cpp's `llama-completion`
# and from our server, unsteered and steered, both on the CPU. The texts should match each other in both runs, and
# the steered ones should differ from the unsteered ones.
# Not part of local_ci.sh: it needs a llama.cpp build and a GGUF on disk.
#
# Usage: scripts/control_vector_parity.sh <model.gguf> [llama.cpp bin dir]
#   CV_SCALE (0.8), CV_LAYERS ("10 21") and CV_TOKENS (32) set the steering and the completion length; CV_KV_F32=1
#   keeps both KV caches at f32 (both store f16 on the CPU by default, each rounding its own way)
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '9,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
bin=${2:-$HOME/Programming/llama.cpp/build_cuda/bin}
scale=${CV_SCALE:-0.8}
read -r first last <<<"${CV_LAYERS:-10 21}"
tokens=${CV_TOKENS:-32}
port=${CV_PORT:-18234}
prompt="<|user|>
How are you feeling today?</s>
<|assistant|>
"
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
server=""
trap '[[ -n "$server" ]] && kill "$server" 2>/dev/null; rm -rf "$work"' EXIT

# TinyLlama's chat format; one prompt a line, newlines escaped as the generator reads them
for mood in happy sad; do
    for question in "Who are you?" "Hello" "How is your day going?" "Tell me about yourself."; do
        printf '<|system|>\\nAct like a person who is extremely %s.</s>\\n<|user|>\\n%s</s>\\n<|assistant|>\\n\n' \
            "$mood" "$question"
    done >"$work/$mood.txt"
done
"$bin/llama-cvector-generator" -m "$model" -ngl 0 --method mean \
    --positive-file "$work/happy.txt" --negative-file "$work/sad.txt" -o "$work/happy.gguf" >/dev/null 2>&1

kv=()
if [[ -n "${CV_KV_F32:-}" ]]; then
    kv=(-ctk f32 -ctv f32)
    export INFERENCE_RS_CPU_KV_F32=1
fi
llama() {
    "$bin/llama-completion" -m "$model" -ngl 0 --temp 0 -n "$tokens" -no-cnv --no-display-prompt "${kv[@]}" \
        -p "$prompt" "$@" 2>/dev/null
}
llama >"$work/llama_plain.txt"
llama --control-vector-scaled "$work/happy.gguf:$scale" --control-vector-layer-range "$first" "$last" \
    >"$work/llama_steered.txt"

# a directory of its own, so no projector beside the model is taken for a multimodal load
mkdir "$work/model"
ln -s "$model" "$work/model/model.gguf"
ours() {
    "$root/target/debug/inference" serve --cpu --no-ui -p "$port" -m "$work/model" -f model.gguf "$@" \
        >"$work/serve.log" 2>&1 &
    server=$!
    until curl -s -o /dev/null "localhost:$port/health"; do
        kill -0 "$server" 2>/dev/null || { tail -5 "$work/serve.log" >&2; exit 1; }
        sleep 1
    done
    python3 -c '
import json, sys, urllib.request
body = {"model": "default", "prompt": sys.argv[1], "max_tokens": int(sys.argv[2]), "temperature": 0}
r = urllib.request.urlopen(urllib.request.Request("http://localhost:" + sys.argv[3] + "/v1/completions",
    json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
sys.stdout.write(json.load(r)["choices"][0]["text"])' "$prompt" "$tokens" "$port"
    kill "$server"
    wait "$server" 2>/dev/null || true
    server=""
}
ours >"$work/ours_plain.txt"
ours --control-vector "$work/happy.gguf:$scale" --control-vector-layer-range "$first" "$last" >"$work/ours_steered.txt"

for run in plain steered; do
    echo "== $run"
    echo "llama.cpp: $(tr '\n' ' ' <"$work/llama_$run.txt")"
    echo "ours:      $(tr '\n' ' ' <"$work/ours_$run.txt")"
    if [[ "$(tr -d '[:space:]' <"$work/llama_$run.txt")" == "$(tr -d '[:space:]' <"$work/ours_$run.txt")" ]]; then
        echo "same text"
    else
        echo "texts differ"
    fi
done
