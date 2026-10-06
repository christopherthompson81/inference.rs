#!/usr/bin/env python3
"""Build the inference_rs wheel for this machine: the pure-Python package with libinference_ffi bundled in it.

Usage:
    python scripts/release/build_wheels.py                     # release build, accelerator picked for this machine
    python scripts/release/build_wheels.py --accelerator cuda --features nccl
    python scripts/release/build_wheels.py --library target/bundle/libinference_ffi.so    # package a built library

The wheel is as portable as its library: a Linux CPU wheel needs the newest glibc the build host's library links
against, and a CUDA wheel needs the CUDA runtime its library links and a GPU of the compute capability it was built for.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPO_ROOT / "bindings" / "python"
PACKAGE_FILES = ("pyproject.toml", "setup.py", "README.md", "inference_rs")
BUNDLED_DIR = Path("inference_rs") / "_lib"
DEFAULT_OUT = REPO_ROOT / "target" / "wheels"
PLATFORM_VARIABLE = "INFERENCE_WHEEL_PLATFORM"
ACCELERATOR_FEATURES = {"cpu": [], "cuda": ["cuda"], "metal": ["metal"]}
# Metal needs the macOS 15 SDK's APIs; a CPU build runs on anything Apple still supports.
MACOS_DEPLOYMENT_TARGETS = {"metal": "15.0", "cpu": "11.0"}
MIN_SETUPTOOLS = (77, 0)
GLIBC_VERSION = re.compile(rb"GLIBC_2\.(\d+)")
CUDART_NEEDED = re.compile(rb"(?:lib)?cudart(?:64_)?\.?(?:so\.)?(\d+)")
NEEDED_ENTRY = re.compile(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]")
# A dev CUDA build loads its kernel libraries from the checkout by absolute path, so it cannot travel.
CHECKOUT_KERNELS = b"/cuda-kernels/"
# The libraries every manylinux policy lets a wheel assume; anything else needs a plain linux tag.
MANYLINUX_LIBRARIES = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "librt.so.1",
    "libpthread.so.0",
    "libgcc_s.so.1",
    "libstdc++.so.6",
    "ld-linux-x86-64.so.2",
    "ld-linux-aarch64.so.1",
}


def library_name() -> str:
    if sys.platform == "win32":
        return "inference_ffi.dll"
    if sys.platform == "darwin":
        return "libinference_ffi.dylib"
    return "libinference_ffi.so"


def machine() -> str:
    arch = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)


def default_accelerator() -> str:
    if sys.platform == "darwin" and machine() == "aarch64":
        return "metal"
    return "cpu"


def compute_capability() -> str | None:
    """The SMs the kernels build for (`sm80.sm86`): the build scripts read CUDA_COMPUTE_CAP, else the first GPU."""
    if os.environ.get("CUDA_COMPUTE_CAP"):
        # cudaforge's list: 86, 8.6, sm_90 or 90a, split on commas, semicolons or spaces
        archs = {
            int(re.sub(r"\D", "", arch))
            for arch in re.split(r"[,;\s]+", os.environ["CUDA_COMPUTE_CAP"])
            if arch
        }
        return ".".join(f"sm{arch}" for arch in sorted(archs))
    if not shutil.which("nvidia-smi"):
        return None
    query = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    lines = subprocess.run(
        query, capture_output=True, text=True, check=False
    ).stdout.split()
    return f"sm{lines[0].replace('.', '')}" if lines else None


def needed_libraries(library: Path) -> list[str] | None:
    if not shutil.which("readelf"):
        return None
    dynamic = subprocess.run(
        ["readelf", "-d", str(library)], capture_output=True, text=True, check=True
    ).stdout
    return NEEDED_ENTRY.findall(dynamic)


def cuda_local_version(data: bytes) -> str:
    found = CUDART_NEEDED.search(data)
    if found is None:
        sys.exit(
            "the library uses CUDA but names no CUDA runtime to version the wheel by"
        )
    sm = compute_capability()
    if sm is None:
        sys.exit(
            "set CUDA_COMPUTE_CAP to the compute capability the library was built for"
        )
    return f"cu{found.group(1).decode()}.{sm}"


def platform_tag(library: Path, accelerator: str) -> str:
    """The wheel platform the library's own requirements allow."""
    if sys.platform == "win32":
        return "win_arm64" if machine() == "aarch64" else "win_amd64"
    if sys.platform == "darwin":
        target = os.environ.get(
            "MACOSX_DEPLOYMENT_TARGET",
            MACOS_DEPLOYMENT_TARGETS.get(accelerator, "11.0"),
        )
        major, _, minor = target.partition(".")
        return f"macosx_{major}_{minor or '0'}_{'arm64' if machine() == 'aarch64' else 'x86_64'}"
    needed = needed_libraries(library)
    if (
        accelerator == "cuda"
        or needed is None
        or not set(needed) <= MANYLINUX_LIBRARIES
    ):
        linked = (
            ", ".join(sorted(set(needed or []) - MANYLINUX_LIBRARIES))
            or "unknown libraries"
        )
        print(
            f"note: the library links {linked}, so the wheel claims plain linux, not manylinux"
        )
        return f"linux_{machine()}"
    minors = [int(minor) for minor in GLIBC_VERSION.findall(library.read_bytes())]
    return f"manylinux_2_{max(minors, default=17)}_{machine()}"


def build_library(accelerator: str, features: list[str]) -> Path:
    features = ACCELERATOR_FEATURES[accelerator] + features
    command = ["cargo", "build", "--profile", "bundle", "-p", "inference-ffi"]
    if features:
        command += ["--features", ",".join(features)]
    env = dict(os.environ)
    # Set, even empty, it overrides the checkout's target-cpu=native, so the wheel runs on any CPU of its arch.
    env.setdefault("RUSTFLAGS", "")
    if sys.platform == "darwin":
        env.setdefault(
            "MACOSX_DEPLOYMENT_TARGET",
            MACOS_DEPLOYMENT_TARGETS.get(accelerator, "11.0"),
        )
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
    return REPO_ROOT / "target" / "bundle" / library_name()


def stage(library: Path, into: Path, local_version: str | None) -> Path:
    for name in PACKAGE_FILES:
        source = PACKAGE_SOURCE / name
        if source.is_dir():
            shutil.copytree(
                source,
                into / name,
                ignore=shutil.ignore_patterns("__pycache__", "_lib"),
            )
        elif source.exists():
            shutil.copy2(source, into / name)
    (into / BUNDLED_DIR).mkdir()
    shutil.copy2(library, into / BUNDLED_DIR / library_name())
    if local_version:
        pyproject = into / "pyproject.toml"
        text = re.sub(
            r'^version = "([^"]+)"',
            rf'version = "\1+{local_version}"',
            pyproject.read_text(),
            flags=re.M,
        )
        pyproject.write_text(text)
    return into


def build_wheel(staged: Path, out: Path, tag: str) -> Path:
    built = staged.parent / "wheel"
    # No cache: the staging path is new every build, so a cached wheel would never be reused.
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
    ]
    command += ["-w", str(built), str(staged)]
    print("+", " ".join(command), f"({PLATFORM_VARIABLE}={tag})")
    subprocess.run(command, check=True, env={**os.environ, PLATFORM_VARIABLE: tag})
    (wheel,) = built.glob("*.whl")
    out.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(wheel, out / wheel.name))


def check_setuptools() -> None:
    try:
        import setuptools
    except ImportError:
        sys.exit(
            f"building a wheel needs setuptools>={'.'.join(map(str, MIN_SETUPTOOLS))} in {sys.executable}"
        )
    if (
        tuple(int(part) for part in setuptools.__version__.split(".")[:2])
        < MIN_SETUPTOOLS
    ):
        sys.exit(
            f"setuptools {setuptools.__version__} is older than {'.'.join(map(str, MIN_SETUPTOOLS))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--accelerator",
        choices=sorted(ACCELERATOR_FEATURES),
        default=default_accelerator(),
    )
    parser.add_argument(
        "--features", default="", help="extra inference-ffi features, comma separated"
    )
    parser.add_argument(
        "--library",
        type=Path,
        help="package this built library instead of building one",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    check_setuptools()

    extra = [feature for feature in args.features.split(",") if feature]
    library = (
        args.library.resolve()
        if args.library
        else build_library(args.accelerator, extra)
    )
    data = library.read_bytes()
    if CHECKOUT_KERNELS in data:
        sys.exit(
            "the library loads its CUDA kernels from this checkout (a dev build); package a release build"
        )
    accelerator = "cuda" if b"cudart" in data else args.accelerator
    local_version = cuda_local_version(data) if accelerator == "cuda" else None
    with tempfile.TemporaryDirectory(prefix="inference-wheel-") as scratch:
        staged = Path(scratch) / "package"
        staged.mkdir()
        stage(library, staged, local_version)
        wheel = build_wheel(
            staged, args.out.resolve(), platform_tag(library, accelerator)
        )
    print(wheel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
