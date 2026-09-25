"""
PaddleOCR-VL document OCR with the Python SDK.

The model is task-prompted: "OCR:" (text), "Table Recognition:" (OTSL <fcel>/<nl> markup),
"Formula Recognition:" (LaTeX), "Chart Recognition:". Run from the repository root.
"""

import base64
from pathlib import Path

from inference_rs import ChatCompletionRequest, MultimodalArchitecture, Runner, Which

IMAGE = Path("inference/tests/fixtures/paddleocr_vl/table.png")
image_url = "data:image/png;base64," + base64.b64encode(IMAGE.read_bytes()).decode()

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="PaddlePaddle/PaddleOCR-VL-1.6",
        arch=MultimodalArchitecture.PaddleOcrVl,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Table Recognition:"},
                ],
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )
)
print(res.choices[0].message.content)
print(res.usage)
