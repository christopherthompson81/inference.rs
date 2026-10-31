#!/usr/bin/env python3
"""Copy a GGUF and add tensors from a Hugging Face safetensors checkpoint under their own names, fetching only their
bytes.

llama.cpp's converters drop some tensors inference.rs can use, such as ERNIE 4.5's MTP layer (`model.mtp_*`, which
our ERNIE GGUF bindings read under these names). This reads each shard's safetensors header over HTTP, then fetches
just the tensors whose names start with --prefix by byte range (ERNIE-4.5-21B-A3B's MTP layer is 0.23 GiB of a
43 GiB checkpoint), and writes a copy of the GGUF with them appended, BF16 and F32 kept as they are. Needs llama.cpp's
gguf-py (--gguf-py) and huggingface_hub.

Usage: python3 scripts/gguf_add_hf_tensors.py <in.gguf> <out.gguf> --repo baidu/ERNIE-4.5-21B-A3B-PT
           --revision <sha> --prefix model.mtp_ [--gguf-py ~/Programming/llama.cpp/gguf-py]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HEADER_LEN_BYTES = 8
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
# safetensors dtype -> (numpy view, GGML type name)
DTYPES = {
    "BF16": (np.uint16, "BF16"),
    "F32": (np.float32, "F32"),
    "F16": (np.float16, "F16"),
}


def fetch_range(session, url: str, start: int, end: int) -> bytes:
    response = session.get(
        url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=600
    )
    response.raise_for_status()
    if len(response.content) != end - start:
        sys.exit(f"{url}: asked for {end - start} bytes, got {len(response.content)}")
    return response.content


def shard_urls(repo: str, revision: str) -> list[str]:
    from huggingface_hub import hf_hub_download, hf_hub_url
    from huggingface_hub.utils import EntryNotFoundError

    try:
        index = json.loads(
            Path(hf_hub_download(repo, INDEX_FILE, revision=revision)).read_text()
        )
        shards = sorted(set(index["weight_map"].values()))
    except EntryNotFoundError:
        shards = [SINGLE_FILE]
    return [hf_hub_url(repo, shard, revision=revision) for shard in shards]


def fetch_tensors(
    repo: str, revision: str, prefix: str
) -> dict[str, tuple[str, list[int], bytes]]:
    from huggingface_hub import get_session
    from huggingface_hub.utils import build_hf_headers

    session = get_session()
    session.headers.update(build_hf_headers())
    tensors = {}
    for url in shard_urls(repo, revision):
        (header_len,) = struct.unpack(
            "<Q", fetch_range(session, url, 0, HEADER_LEN_BYTES)
        )
        header = json.loads(
            fetch_range(session, url, HEADER_LEN_BYTES, HEADER_LEN_BYTES + header_len)
        )
        base = HEADER_LEN_BYTES + header_len
        for name, info in header.items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            if info["dtype"] not in DTYPES:
                sys.exit(f"{name}: dtype {info['dtype']} is not one this script writes")
            start, end = info["data_offsets"]
            tensors[name] = (
                info["dtype"],
                info["shape"],
                fetch_range(session, url, base + start, base + end),
            )
            print(f"fetched {name} {info['dtype']} {info['shape']}")
    if not tensors:
        sys.exit(f"no tensor of {repo} starts with {prefix!r}")
    return tensors


def write(
    gguf, source: Path, out: Path, added: dict[str, tuple[str, list[int], bytes]]
):
    reader = gguf.GGUFReader(source)
    clash = [tensor.name for tensor in reader.tensors if tensor.name in added]
    if clash:
        sys.exit(f"{source} already has {clash}")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        # the writer writes the architecture and the GGUF.* fields itself
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith(
            "GGUF."
        ):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    arrays = {}
    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
    for name, (dtype, shape, data) in added.items():
        view, ggml = DTYPES[dtype]
        array = np.frombuffer(data, dtype=view).reshape(shape)
        arrays[name] = array
        writer.add_tensor_info(
            name,
            array.shape,
            array.dtype,
            array.nbytes,
            gguf.GGMLQuantizationType[ggml],
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for array in arrays.values():
        writer.write_tensor_data(array)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="the checkpoint commit, so the tensors are reproducible",
    )
    parser.add_argument(
        "--prefix", required=True, help="add the tensors whose names start with this"
    )
    parser.add_argument(
        "--gguf-py", type=Path, default=Path.home() / "Programming/llama.cpp/gguf-py"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_py))
    import gguf

    added = fetch_tensors(args.repo, args.revision, args.prefix)
    write(gguf, args.source, args.out, added)
    print(f"wrote {args.out} with {len(added)} added tensors")


if __name__ == "__main__":
    main()
