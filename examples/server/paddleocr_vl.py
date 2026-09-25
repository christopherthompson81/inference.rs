"""
PaddleOCR-VL document OCR through the OpenAI-compatible HTTP API.

Start the server:
    inference serve -p 1234 -m PaddlePaddle/PaddleOCR-VL-1.6

The model is task-prompted: "OCR:" (text), "Table Recognition:" (OTSL <fcel>/<nl> markup),
"Formula Recognition:" (LaTeX), "Chart Recognition:". Run from the repository root.
"""

import base64
from pathlib import Path

from openai import OpenAI

IMAGE = Path("inference/tests/fixtures/paddleocr_vl/table.png")
image_url = "data:image/png;base64," + base64.b64encode(IMAGE.read_bytes()).decode()

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Table Recognition:"},
            ],
        },
    ],
    max_tokens=512,
    temperature=0.0,
)
print(completion.choices[0].message.content)
