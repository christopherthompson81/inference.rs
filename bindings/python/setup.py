"""A wheel that carries libinference_ffi in inference_rs/_lib is tagged for its platform; without one it stays pure."""

import os
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel

LIBRARY_DIR = Path(__file__).parent / "inference_rs" / "_lib"
# Set by scripts/release/build_wheels.py to the tag the library's own requirements allow, e.g. manylinux_2_39_x86_64.
PLATFORM_VARIABLE = "INFERENCE_WHEEL_PLATFORM"


def bundles_library() -> bool:
    return any(LIBRARY_DIR.glob("*inference_ffi*"))


class LibraryDistribution(Distribution):
    # Native content, so the package builds and installs as platlib.
    def has_ext_modules(self):
        return bundles_library()


class LibraryWheel(bdist_wheel):
    def get_tag(self):
        python, abi, platform = super().get_tag()
        if not bundles_library():
            return python, abi, platform
        # The library is loaded through ctypes, so no Python version or ABI is involved.
        return "py3", "none", os.environ.get(PLATFORM_VARIABLE, platform)


setup(distclass=LibraryDistribution, cmdclass={"bdist_wheel": LibraryWheel})
