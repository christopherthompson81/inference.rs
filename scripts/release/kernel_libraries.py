"""The CUDA kernel libraries a built `libinference_ffi` loads, found the way its build laid them out.

Each `cuda-kernels/<name>.lib` beside the host library names one library's nominal path, `<dir>/lib<name>.so`, which
the host records; the build links a copy per GPU arch into `<dir>/sm<arch>/` (`sm86`, `sm90a`) and writes the build id
every copy carries to `<dir>/lib<name>.so.id`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KERNEL_POINTERS = "cuda-kernels"
# where a package puts them, beside the library that loads them
KERNEL_SUBDIR = "kernels"
# a dev build loads its kernels from the checkout and records paths that mean nothing elsewhere
CHECKOUT_KERNELS = b"/cuda-kernels/"
POINTER_SUFFIX = ".lib"
# inference-kernel-build's ARCH_DIR_PREFIX and BUILD_ID_EXTENSION
ARCH_DIR_PREFIX = "sm"
ID_SUFFIX = ".id"
# one GGUF quant type's kernels (`inferencequant_q4_k`), which a model needs only when it holds that type
TYPE_LIBRARY = re.compile(r"^inferencequant_\w+$")


class MissingKernels(Exception):
    """Kernel libraries the host library was built with are gone, or are another build's."""


@dataclass(frozen=True)
class KernelFile:
    library: str
    arch: str
    path: Path
    build_id: str

    @property
    def per_type(self) -> bool:
        return TYPE_LIBRARY.match(self.library) is not None

    @property
    def relative(self) -> str:
        """Where it sits in a kernels dir: `sm86/libinferencequant.so`."""
        return f"{ARCH_DIR_PREFIX}{self.arch}/{self.path.name}"


def built_kernels(library: Path) -> list[KernelFile]:
    """Every arch's copy of each kernel library `library` was built with (not another build's, which a target dir
    also holds)."""
    data = library.read_bytes()
    pointers = sorted((library.parent / KERNEL_POINTERS).glob(f"*{POINTER_SUFFIX}"))
    nominal = [Path(pointer.read_text().strip()) for pointer in pointers]
    own = [path for path in nominal if str(path).encode() in data]
    files: list[KernelFile] = []
    missing: list[str] = []
    for path in own:
        id_file = path.with_name(path.name + ID_SUFFIX)
        copies = sorted(path.parent.glob(f"{ARCH_DIR_PREFIX}*/{path.name}"))
        if not copies or not id_file.is_file():
            missing.append(str(path))
            continue
        name = path.name.removeprefix("lib").removesuffix(".so")
        build_id = id_file.read_text().strip()
        for copy in copies:
            # every arch's copy carries the id the host checks at load
            if build_id.encode() not in copy.read_bytes():
                missing.append(f"{copy} (another build's)")
                continue
            arch = copy.parent.name.removeprefix(ARCH_DIR_PREFIX)
            files.append(KernelFile(name, arch, copy, build_id))
    if missing:
        raise MissingKernels(f"kernel libraries {library.name} was built with are gone: {missing}")
    return files
