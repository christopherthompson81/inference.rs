"""Compares the C consumer's `parity:` lines with the Rust detect example's JSON; exits 1 on any mismatch."""
import json
import sys

rust = [d for line in open(sys.argv[1]) for d in json.loads(line)["detections"]]
c = [line.split()[1:] for line in open(sys.argv[2])]
if len(rust) != len(c):
    sys.exit(f"{len(rust)} Rust vs {len(c)} C detections")
for i, (r, (idx, cls, label, score, *bbox)) in enumerate(zip(rust, c)):
    # numeric tolerances: both sides print the same f32s, through different formatters
    same = int(idx) == i and int(cls) == r["class_id"] and label == r["label"]
    same = same and abs(float(score) - r["score"]) < 2e-4 and all(abs(float(b) - v) < 0.1 for b, v in zip(bbox, r["bbox"]))
    if not same:
        sys.exit(f"detection {i}: Rust {r} vs C {idx} {cls} {label} {score} {bbox}")
print(f"C consumer matches the Rust detector: {len(c)} detections")
