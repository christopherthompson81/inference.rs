"""Prints transformers greedy goldens for the fixtures; usage: make_goldens.py <PaddleOCR-VL checkpoint dir> [float32|bfloat16]."""
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

HERE = Path(__file__).parent
MAX_NEW_TOKENS = 64
# (image, prompt); None is the text-only case
CASES = [
    ("ocr.png", "OCR:"),
    ("page_00.png", "OCR:"),
    ("page_01.png", "OCR:"),
    ("table.png", "Table Recognition:"),
    (None, "Reply with the single word: ok"),
]

model_dir = sys.argv[1]
dtype = getattr(torch, sys.argv[2] if len(sys.argv) > 2 else "float32")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_dir)
model = PaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).to(device).eval()

for image, prompt in CASES:
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": Image.open(HERE / image).convert("RGB")})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = ids[0, inputs["input_ids"].shape[1]:].tolist()
    print(f"{image} [{prompt}] {new}\n    {processor.decode(new, skip_special_tokens=True)!r}")
