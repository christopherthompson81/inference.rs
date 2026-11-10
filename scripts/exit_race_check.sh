#!/usr/bin/env bash
# Whether a process exits cleanly with GPU work in flight: loads Parakeet through the Python bindings, frees the engine
# three windows into a transcription stream (the stream left running) and exits, RUNS times; counts the exits that
# fault or time out. The driver's exit handlers race any GPU call still running, so a clean run needs the library to
# stop its work first. Not part of local_ci.sh: it needs a GPU and a real checkpoint (the tiny one never loads cuBLAS,
# whose runtime's exit handler faulted).
#
# Usage: scripts/exit_race_check.sh <device: cuda:0 | vulkan:0> <audio, a few minutes> [runs]
set -euo pipefail

RUNS_DEFAULT=15
EXIT_TIMEOUT_SECONDS=120
WINDOWS_BEFORE_EXIT=3

if [[ $# -lt 2 ]]; then
    sed -n '9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
device=$1 audio=$2 runs=${3:-$RUNS_DEFAULT}
root=$(cd "$(dirname "$0")/.." && pwd)
child=$(mktemp --suffix=.py)
trap 'rm -f "$child"' EXIT
cat > "$child" <<PY
import json, sys
sys.path.insert(0, "$root/bindings/python")
from inference_rs._engine import JsonEngine
engine = JsonEngine(json.dumps({"model": {"Transcription": {"model_id": "nvidia/parakeet-tdt-0.6b-v3"}},
                                "runtime": {"device": "$device"}}))
stream = engine.transcription_stream(json.dumps({"path": "$audio", "max_window_seconds": 15}))
windows = 0
while (event := stream.next()) is not None and event.name not in ("done", "error", "canceled"):
    windows += event.name == "window"
    if windows == $WINDOWS_BEFORE_EXIT:
        break
engine.close()
PY
failed=0
for _ in $(seq "$runs"); do
    status=0
    timeout "$EXIT_TIMEOUT_SECONDS" python3 "$child" >/dev/null 2>&1 || status=$?
    [[ $status -eq 0 ]] || failed=$((failed + 1))
done
echo "$device: $failed of $runs exits faulted or hung"
[[ $failed -eq 0 ]]
