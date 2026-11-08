#!/usr/bin/env bash
# Whether a model holds its outputs when its large CPU f32 matmuls take bf16 inputs: runs it with them kept f32
# (INFERENCE_CPU_NO_BF16) and with them taken for every model (INFERENCE_CPU_BF16), and compares by the model's own
# measure. A model that passes sets `bf16_matmuls` in its loader; one that fails stays f32, its result noted there.
# Not part of local_ci.sh: it needs real checkpoints and a CPU with AVX-512 BF16. Use long inputs (minutes of audio,
# contexts of thousands of tokens): rounding grows with each dot product's length.
#
# Usage: scripts/bf16_check.sh asr <model id or path> <audio>        transcript words, and starts within a frame
#        scripts/bf16_check.sh diar <model id or path> <audio>       speaker-frames that move
#        scripts/bf16_check.sh kld <gguf> <kld base> [text]          KLD and top-1 against a llama-perplexity base
set -euo pipefail

PORT=1299
# what a run may move and still pass: none of an ASR's words, 0.5 % of a diarizer's speaker-frames, 1e-3 of KLD
MAX_FRAMES_MOVED=0.005
MAX_KLD_RISE=0.001

if [[ $# -lt 3 ]]; then
    sed -n '9,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
kind=$1
root=$(cd "$(dirname "$0")/.." && pwd)
grep -q avx512_bf16 /proc/cpuinfo || { echo "this CPU has no AVX-512 BF16: nothing to measure" >&2; exit 2; }
out=$(mktemp -d)
server=
# a failed request exits early: its server goes too
trap '[[ -n $server ]] && kill $server 2>/dev/null; rm -rf "$out"' EXIT

serve_and_post() { # mode route model file
    local mode=$1 route=$2 model=$3 file=$4 env
    env=$([[ $mode == bf16 ]] && echo INFERENCE_CPU_BF16=1 || echo INFERENCE_CPU_NO_BF16=1)
    env "$env" "$cli" serve --cpu --no-ui --port $PORT -m "$model" >"$out/serve_$mode.log" 2>&1 &
    server=$!
    for _ in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 1; done
    # a transcript with its words' times; a diarization's json holds its segments
    local form=(-F response_format=json)
    [[ $route == transcriptions ]] && form=(-F response_format=verbose_json -F "timestamp_granularities[]=word")
    curl -sf -o "$out/$mode.json" http://127.0.0.1:$PORT/v1/audio/$route -F file=@"$file" "${form[@]}"
    kill $server
    wait $server 2>/dev/null || true
    server=
}

case $kind in
asr | diar)
    cli=$(cargo build --manifest-path "$root/Cargo.toml" --release -p inference-cli \
        --message-format=json-render-diagnostics | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | tail -n 1)
    route=$([[ $kind == asr ]] && echo transcriptions || echo diarization)
    for mode in f32 bf16; do serve_and_post $mode $route "$2" "$(realpath "$3")"; done
    python3 - "$kind" "$out" "$MAX_FRAMES_MOVED" <<'EOF'
import json, sys, difflib
kind, out, max_moved = sys.argv[1], sys.argv[2], float(sys.argv[3])
f32, bf16 = (json.load(open(f"{out}/{m}.json")) for m in ("f32", "bf16"))
if kind == "asr":
    a, b = [w["word"] for w in f32["words"]], [w["word"] for w in bf16["words"]]
    spans = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    starts = sum(x["start"] != y["start"] for x, y in zip(f32["words"], bf16["words"]))
    print(f"{len(a)} words: {len(spans)} differing spans, {starts} starts moved")
    for _, i1, i2, j1, j2 in spans[:10]:
        print("   ", " ".join(a[i1:i2]), "->", " ".join(b[j1:j2]))
    sys.exit(1 if spans else 0)
def frames(d):
    return {(s["speaker"], f) for s in d["segments"] for f in range(round(s["start"] * 100), round(s["end"] * 100))}
x, y = frames(f32), frames(bf16)
moved = len(x ^ y) / max(len(x), 1)
print(f"{len(x)} speaker-frames: {100 * moved:.3f} % moved")
sys.exit(1 if moved > max_moved else 0)
EOF
    ;;
kld)
    ppl=$(cargo build --manifest-path "$root/Cargo.toml" --release -p inference-examples --example perplexity \
        --message-format=json-render-diagnostics | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | tail -n 1)
    gguf=$(realpath "$2")
    base=$(realpath "$3")
    text=${4:-/dev/null}
    for mode in f32 bf16; do
        env=$([[ $mode == bf16 ]] && echo INFERENCE_CPU_BF16=1 || echo INFERENCE_CPU_NO_BF16=1)
        env "$env" "$ppl" --gguf "$gguf" --cpu --kld-base "$base" -f "$text" | tee "$out/$mode.log" | grep -E "^KLD mean|^same top"
    done
    python3 - "$out" "$MAX_KLD_RISE" <<'EOF'
import re, sys
out, max_rise = sys.argv[1], float(sys.argv[2])
kld = {m: float(re.search(r"^KLD mean\s+([\d.]+)", open(f"{out}/{m}.log").read(), re.M).group(1)) for m in ("f32", "bf16")}
print(f"KLD f32 {kld['f32']}, bf16 {kld['bf16']}")
sys.exit(1 if kld["bf16"] - kld["f32"] > max_rise else 0)
EOF
    ;;
*)
    echo "kind must be asr, diar or kld, got $kind" >&2
    exit 2
    ;;
esac
