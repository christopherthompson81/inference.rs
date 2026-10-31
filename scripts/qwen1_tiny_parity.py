#!/usr/bin/env python3
"""Parity check of our Qwen 1 (QWenLMHeadModel) text model and qwen.tiktoken tokenizer against Qwen's own code.

Qwen 1 has no transformers class: its repo's remote code (modeling_qwen.py, tokenization_qwen.py, fetched from
Qwen/Qwen-7B-Chat at REVISION with qwen.tiktoken) builds a tiny model with seeded random weights over the real
151,936-token vocabulary, patched where transformers 5 removed what it calls. Both sides score the same token ids
(mean KL(reference || ours), top-1 agreement, largest logit difference) and decode greedily from them, the reference
by full forwards; both tokenizers encode the repository README and a few special-token strings, and our chat template
renders a conversation to the tokens Qwen's `make_context` builds. With --llama-cpp,
that checkout's converter also writes an f32 GGUF, scored the same way. Not part of local_ci.sh: it needs transformers
5 with torch, tiktoken and einops, and our library built with `cargo build -p inference-ffi` (add `--features cuda` for
--device cuda:0), found through the checkout's bindings.

Usage: PYTHONPATH=bindings/python python3 scripts/qwen1_tiny_parity.py [--max-kl X] [--device cpu|cuda:0]
           [--dtype f32|bf16] [--keep DIR] [--seed N] [--llama-cpp DIR]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "Qwen/Qwen-7B-Chat"
REVISION = "93a65d34827a3cc269b727e67004743b723e2f83"
REMOTE_FILES = [
    "configuration_qwen.py",
    "modeling_qwen.py",
    "qwen_generation_utils.py",
    "tokenization_qwen.py",
    "cpp_kernels.py",
    "qwen.tiktoken",
    "tokenizer_config.json",
    "config.json",
]
# what transformers 5 no longer provides: the streaming helper package and PreTrainedModel.get_head_mask
PATCHES = [
    (
        "        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig",
        '        raise NotImplementedError("chat_stream")',
    ),
    (
        "        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)",
        "        head_mask = [None] * self.config.num_hidden_layers",
    ),
]
DEFAULT_MAX_KL = 1e-6
# bf16 weights and activations against an f32 reference
BF16_MAX_KL = 2e-3
# how far below the reference's best logit a bf16 step's token may score: a near tie, not a wrong selection
BF16_TIE_MARGIN = 0.25
GENERATED = 12
IDS = [
    11,
    57,
    203,
    4,
    98,
    140,
    33,
    77,
    19,
    64,
    222,
    5,
    180,
    91,
    42,
    7,
    130,
    151644,
    872,
    198,
]
SPECIAL_TEXTS = [
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n",
    "tabs\tand  spaces, numbers 12345, accents tést, CJK 中文<|endoftext|>",
]
# make_context's (query, response) history, then the query; our chat holds the same turns as messages
HISTORY = [("What is 2 + 2?", "4.")]
QUERY = "And 3 + 3?"
SYSTEM = "You are a terse assistant."
ROOT = Path(__file__).resolve().parent.parent
GGUF_FILE = "model.gguf"
# the converter reads the tokenizer through the remote code, which needs the same patches
CONVERT = """
import runpy, sys
sys.argv = ["convert_hf_to_gguf.py", *sys.argv[1:]]
runpy.run_path("convert_hf_to_gguf.py", run_name="__main__")
"""
TINY = {
    "hidden_size": 64,
    "intermediate_size": 192,
    "num_attention_heads": 4,
    "kv_channels": 16,
    "num_hidden_layers": 3,
    "seq_length": 256,
    "max_position_embeddings": 256,
    "use_flash_attn": False,
    "fp32": True,
    # wide enough that the logits are peaked: at the default 0.02 every distribution is near uniform
    "initializer_range": 0.1,
}


def fetch(source: Path):
    from huggingface_hub import hf_hub_download

    source.mkdir(parents=True, exist_ok=True)
    for name in REMOTE_FILES:
        shutil.copy(hf_hub_download(REPO, name, revision=REVISION), source / name)
    modeling = source / "modeling_qwen.py"
    text = modeling.read_text()
    for old, new in PATCHES:
        if old not in text:
            sys.exit(f"modeling_qwen.py no longer has {old.strip()!r}")
        text = text.replace(old, new)
    modeling.write_text(text)
    config = json.loads((source / "config.json").read_text())
    (source / "config.json").write_text(json.dumps({**config, **TINY}))


def build(model_dir: Path, seed: int):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, dtype=torch.float32
    ).eval()
    # random init leaves norms at one and biases at zero; perturb them so a dropped one shows
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight", ".bias")):
                param.add_(torch.randn_like(param) * 0.1)
    model.save_pretrained(model_dir, safe_serialization=True)
    return model


def reference(model):
    import torch

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS]), use_cache=False).logits[0].float()
        ids = list(IDS)
        for _ in range(GENERATED):
            step = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
            ids.append(int(step.argmax()))
    return logits.tolist(), ids[len(IDS) :]


def tie_gaps(model, continuation: list[int]) -> list[float]:
    """Teacher-forced: at each step, the reference's best logit minus the logit of the token we chose."""
    import torch

    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor([IDS + continuation]), use_cache=False
        ).logits[0, len(IDS) - 1 : -1]
    return [(row.max() - row[token]).item() for row, token in zip(logits, continuation)]


def convert(model_dir: Path, llama_cpp: Path) -> Path:
    gguf = model_dir / GGUF_FILE
    command = [
        sys.executable,
        "-E",
        "-c",
        CONVERT,
        str(model_dir),
        "--outtype",
        "f32",
        "--outfile",
        str(gguf),
    ]
    subprocess.run(command, cwd=llama_cpp, check=True, capture_output=True)
    return gguf


def spec(model: Path, device: str, dtype: str) -> str:
    if model.suffix == ".gguf":
        selected = {
            "GGUF": {
                "quantized_model_id": str(model.parent),
                "quantized_filename": model.name,
                "dtype": dtype,
            }
        }
    else:
        selected = {"Plain": {"model_id": str(model), "dtype": dtype}}
    return json.dumps(
        {
            "model": selected,
            "runtime": {"device": device, "paged_attn": device != "cpu"},
        }
    )


def chat_messages() -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    for query, response in HISTORY:
        messages += [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
    return messages + [{"role": "user", "content": QUERY}]


def reference_chat(model_dir: Path, tokenizer) -> list[int]:
    sys.path.insert(0, str(model_dir))
    from qwen_generation_utils import make_context

    return make_context(tokenizer, QUERY, history=HISTORY, system=SYSTEM)[1]


def ours(model: Path, device: str, dtype: str, texts: list[str]):
    import inference_rs as ir

    request = {
        "model": "default",
        "prompt": IDS,
        "max_tokens": GENERATED,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": 1,
    }
    with ir.JsonEngine(spec(model, device, dtype)) as engine:
        scores, logits = engine.prompt_logits(
            json.dumps({"prompt": IDS, "output": "logits"})
        )
        choice = json.loads(engine.completion(json.dumps(request)))["choices"][0]
        encoded = [
            json.loads(
                engine.tokenize(json.dumps({"text": text, "add_special_tokens": False}))
            )["tokens"]
            for text in texts
        ]
        request = {"model": "default", "messages": chat_messages()}
        chat = json.loads(engine.tokenize_chat(json.dumps(request)))["tokens"]
    vocab = json.loads(scores)["vocab_size"]
    rows = [list(logits[i * vocab : (i + 1) * vocab]) for i in range(len(IDS))]
    # greedy, so each step's top alternative is the token it chose, and that entry carries the id
    continuation = [
        step["top_logprobs"][0]["token"] for step in choice["logprobs"]["content"]
    ]
    return rows, continuation, encoded, chat


def compare(expected, actual):
    import torch

    e, a = torch.tensor(expected), torch.tensor(actual)
    if e.shape != a.shape:
        sys.exit(f"the reference's logits are {tuple(e.shape)}, ours {tuple(a.shape)}")
    le, la = e.log_softmax(-1), a.log_softmax(-1)
    kl = (le.exp() * (le - la)).sum(-1)
    agree = (e.argmax(-1) == a.argmax(-1)).float().mean().item()
    return kl.mean().item(), agree, (e - a).abs().max().item()


def check(label, model_path, model, expected, args, dtype, max_kl, texts) -> list[str]:
    logits, generated, tokens, chat = expected
    actual, continuation, encoded, our_chat = ours(
        model_path, args.device, dtype, texts
    )
    kl, agree, max_diff = compare(logits, actual)
    print(
        f"{label}: {len(IDS)} tokens, mean KL {kl:.2e}, top-1 agreement {agree:.3f}, max logit diff {max_diff:.4f}"
    )
    print(
        f"{label}: greedy {GENERATED} tokens, reference {generated}, ours {continuation}"
    )
    failures = []
    if kl > max_kl:
        failures.append(f"{label}: mean KL {kl:.2e} exceeds {max_kl:.0e}")
    # bf16 can flip a near tie, after which the continuations part ways; each of our steps must still be one
    if dtype == "f32" and continuation != generated:
        failures.append(f"{label}: the greedy continuations differ")
    gap = max(tie_gaps(model, continuation))
    if gap > BF16_TIE_MARGIN:
        failures.append(
            f"{label}: a step's token scores {gap:.3f} below the reference's best"
        )
    mismatched = sum(ours != theirs for ours, theirs in zip(encoded, tokens))
    print(f"{label}: tokenization of {len(texts)} texts, {mismatched} differ")
    if mismatched:
        failures.append(f"{label}: {mismatched} tokenizations differ")
    print(
        f"{label}: chat of {len(chat)} tokens, {'the same as' if our_chat == chat else 'not'} make_context's"
    )
    if our_chat != chat:
        failures.append(f"{label}: the chat renders {our_chat}, make_context {chat}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-kl", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["f32", "bf16"])
    parser.add_argument(
        "--keep", type=Path, help="write the tiny checkpoint here and keep it"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        help="a llama.cpp checkout whose converter writes the GGUF",
    )
    args = parser.parse_args()
    dtype = args.dtype or ("f32" if args.device == "cpu" else "bf16")
    max_kl = args.max_kl or (DEFAULT_MAX_KL if dtype == "f32" else BF16_MAX_KL)
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = args.keep or Path(tmp) / "qwen1"
        shutil.rmtree(model_dir, ignore_errors=True)
        fetch(model_dir)
        model = build(model_dir, args.seed)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        texts = [(ROOT / "README.md").read_text(), *SPECIAL_TEXTS]
        tokens = [tokenizer.encode(text) for text in texts]
        expected = (*reference(model), tokens, reference_chat(model_dir, tokenizer))
        failures = check(
            "safetensors", model_dir, model, expected, args, dtype, max_kl, texts
        )
        if args.llama_cpp:
            gguf = convert(model_dir, args.llama_cpp.resolve())
            failures += check("gguf", gguf, model, expected, args, dtype, max_kl, texts)
    if failures:
        sys.exit("; ".join(failures))
    print("OK")


if __name__ == "__main__":
    main()
