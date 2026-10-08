"""Renders the Python and C# native declarations from inference.h, the one hand-written description of the C ABI.

Python gets its ctypes SIGNATURES table (between markers in _native.py) and C# its P/Invoke declarations
(NativeMethods.g.cs), both with the ABI version. Run with --check to fail when either is stale.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "crates/inference-ffi/include/inference.h"
PYTHON_NATIVE = ROOT / "bindings/python/inference_rs/_native.py"
CSHARP_METHODS = ROOT / "bindings/csharp/src/InferenceRs/Native/NativeMethods.g.cs"
PYTHON_BEGIN = "# BEGIN GENERATED from inference.h by bindings/scripts/generate_native.py\n"
PYTHON_END = "# END GENERATED\n"
INDENT = "    "
# bindings/python/pyproject.toml
PYTHON_LINE_LENGTH = 120
CSHARP_LINE_LENGTH = 120
# Pointers the header documents as arrays rather than single values; everything else follows the rules below.
ARRAY_PARAMS = {"out_bbox", "out_results"}
OUT_PREFIX = "out_"
CSHARP_STRUCT_PREFIX = "Native"

SCALARS = {
    "uint32_t": ("c_uint32", "uint"),
    "int32_t": ("c_int32", "int"),
    "int64_t": ("c_int64", "long"),
    "size_t": ("c_size_t", "nuint"),
    "float": ("c_float", "float"),
    "uint8_t": ("c_uint8", "byte"),
}
STATUS = "inference_status"
# C keywords are not C# ones: a parameter named `string` or `object` must be escaped
CSHARP_KEYWORDS = {
    "abstract",
    "as",
    "base",
    "bool",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "checked",
    "class",
    "const",
    "continue",
    "decimal",
    "default",
    "delegate",
    "do",
    "double",
    "else",
    "enum",
    "event",
    "explicit",
    "extern",
    "false",
    "finally",
    "fixed",
    "float",
    "for",
    "foreach",
    "goto",
    "if",
    "implicit",
    "in",
    "int",
    "interface",
    "internal",
    "is",
    "lock",
    "long",
    "namespace",
    "new",
    "null",
    "object",
    "operator",
    "out",
    "override",
    "params",
    "private",
    "protected",
    "public",
    "readonly",
    "ref",
    "return",
    "sbyte",
    "sealed",
    "short",
    "sizeof",
    "stackalloc",
    "static",
    "string",
    "struct",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "uint",
    "ulong",
    "unchecked",
    "unsafe",
    "ushort",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
}


class Header:
    def __init__(self, text: str):
        code = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        self.version = tuple(
            int(re.search(rf"#define INFERENCE_ABI_VERSION_{part} (\d+)", code).group(1))
            for part in ("MAJOR", "MINOR", "PATCH")
        )
        self.structs = set(re.findall(r"(?:typedef )?struct (\w+)\s*\{", code))
        self.enums = set(re.findall(r"typedef enum (\w+) \{", code))
        self.callbacks = {
            name: (ret.strip(), split_params(params))
            for ret, name, params in re.findall(r"typedef ([\w ]+?)\s*\(\*(\w+)\)\((.*?)\);", code, flags=re.DOTALL)
        }
        self.functions = [
            (name, ret.strip(), split_params(params))
            for ret, name, params in re.findall(
                r"INFERENCE_API\s+([\w \*]+?)\s*\b(inference_\w+)\s*\((.*?)\);", code, flags=re.DOTALL
            )
        ]
        declared = len(re.findall(r"^(?!#).*\bINFERENCE_API\b", code, flags=re.MULTILINE))
        if declared != len(self.functions):
            raise ValueError(f"parsed {len(self.functions)} of the header's {declared} INFERENCE_API declarations")


def split_params(params: str):
    params = " ".join(params.split())
    if params in ("", "void"):
        return []
    out = []
    for param in params.split(","):
        param = param.strip()
        match = re.match(r"(.*?)(\w+)$", param)
        ty = re.sub(r"\bconst\b", "", match.group(1)).replace(" ", "")
        out.append((ty, match.group(2)))
    return out


def camel(name: str) -> str:
    head, *rest = name.split("_")
    name = head + "".join(part.capitalize() for part in rest)
    return f"@{name}" if name in CSHARP_KEYWORDS else name


def pascal(c_name: str) -> str:
    return "".join(part.capitalize() for part in c_name.removeprefix("inference_").split("_"))


def is_buffer(params, i) -> bool:
    ty, _ = params[i]
    if ty not in ("char*", "uint8_t*") or i + 1 >= len(params):
        return False
    length_ty, length = params[i + 1]
    return length_ty == "size_t" and (length == "len" or length.endswith("_len"))


def python_type(header: Header, ty: str, returned: bool = False) -> str:
    base = ty.rstrip("*")
    depth = len(ty) - len(base)
    if ty == "void":
        return "None"
    if depth == 0:
        if base == STATUS or base in header.enums:
            return "status" if returned else "c_int32"
        if base in header.callbacks:
            return base.removeprefix("inference_").upper()
        return SCALARS[base][0]
    if depth == 2:
        return "out"
    if returned or base in ("char", "void", "uint8_t") or base not in header.structs and base not in SCALARS:
        # strings and borrowed pointers stay raw; opaque handles cross as c_void_p
        return "c_char_p" if base == "char" and not returned else "c_void_p"
    if base in header.structs:
        return f"POINTER({pascal(base)})"
    return f"POINTER({SCALARS[base][0]})"


def python_signatures(header: Header) -> str:
    lines = [PYTHON_BEGIN]
    major, minor, patch = header.version
    lines.append(f"ABI_VERSION = ({major} << 16) | ({minor} << 8) | {patch}\n\n")
    lines.append("# name: (restype, argtypes)\nSIGNATURES = {\n")
    for name, ret, params in header.functions:
        args = []
        i = 0
        while i < len(params):
            if is_buffer(params, i):
                args.append("*buffer")
                i += 2
                continue
            args.append(python_type(header, params[i][0]))
            i += 1
        arg_text = ", ".join(args) + ("," if len(args) == 1 else "")
        restype = python_type(header, ret, returned=True)
        line = f'{INDENT}"{name}": ({restype}, ({arg_text})),\n'
        if len(line) - 1 > PYTHON_LINE_LENGTH:
            # the shape ruff gives an entry past the line length
            line = f'{INDENT}"{name}": (\n{INDENT * 2}{restype},\n{INDENT * 2}({arg_text}),\n{INDENT}),\n'
        lines.append(line)
    lines.append("}\n")
    lines.append(PYTHON_END)
    return "".join(lines)


def csharp_type(header: Header, ty: str, name: str, returned: bool = False, counted: bool = False) -> str:
    base = ty.rstrip("*")
    depth = len(ty) - len(base)
    if ty == "void":
        return "void"
    if depth == 0:
        if base == STATUS:
            return "InferenceStatus"
        if base in header.enums:
            return "int"
        if base in header.callbacks:
            ret, params = header.callbacks[base]
            parts = [csharp_type(header, p_ty, p_name) for p_ty, p_name in params]
            return f"delegate* unmanaged[Cdecl]<{', '.join(parts + [csharp_type(header, ret, '', True)])}>"
        return SCALARS[base][1]
    if returned:
        return "IntPtr"
    if depth == 2:
        return "out IntPtr" if name.startswith(OUT_PREFIX) and name not in ARRAY_PARAMS else "IntPtr*"
    if base == "char":
        return "string?"
    if base == "void":
        return "IntPtr"
    if base in header.structs:
        return f"{CSHARP_STRUCT_PREFIX}{pascal(base)}*"
    if base in SCALARS:
        # a single out value is an `out` parameter; inputs and arrays (counted, or listed) stay pointers
        single_out = name.startswith(OUT_PREFIX) and name not in ARRAY_PARAMS and not counted
        return f"out {SCALARS[base][1]}" if single_out else f"{SCALARS[base][1]}*"
    return "IntPtr"


def csharp_methods(header: Header) -> str:
    major, minor, patch = header.version
    lines = [
        "// <auto-generated> from inference.h by bindings/scripts/generate_native.py; edit the header, then rerun it.\n",
        "#nullable enable\n",
        "using System.Runtime.InteropServices;\n\n",
        "namespace InferenceRs.Native;\n\n",
        "internal static unsafe partial class NativeMethods\n{\n",
        f"{INDENT}/// <summary>The ABI these declarations mirror; while it is 0.0.x any other version may differ anywhere.</summary>\n",
        f"{INDENT}internal const uint AbiVersion = ({major} << 16) | ({minor} << 8) | {patch};\n",
    ]
    for name, ret, params in header.functions:
        args = []
        i = 0
        while i < len(params):
            ty, param = params[i]
            if is_buffer(params, i):
                args.append(f"byte* {camel(param)}")
                args.append(f"nuint {camel(params[i + 1][1])}")
                i += 2
                continue
            counted = i + 1 < len(params) and params[i + 1][0] == "size_t"
            args.append(f"{csharp_type(header, ty, param, counted=counted)} {camel(param)}")
            i += 1
        utf8 = any(arg.startswith("string?") for arg in args)
        attribute = (
            "[LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]"
            if utf8
            else "[LibraryImport(Library)]"
        )
        lines.append(f"\n{INDENT}{attribute}\n")
        head = f"{INDENT}internal static partial {csharp_type(header, ret, '', True)} {name}("
        line = f"{head}{', '.join(args)});\n"
        if len(line) - 1 > CSHARP_LINE_LENGTH:
            line = f"{head}\n{INDENT * 2}{', '.join(args)});\n"
        if any(len(text) > CSHARP_LINE_LENGTH for text in line.splitlines()):
            line = f"{head}\n" + ",\n".join(f"{INDENT * 2}{arg}" for arg in args) + ");\n"
        lines.append(line)
    lines.append("}\n")
    return "".join(lines)


def render_python(header: Header, current: str) -> str:
    start, end = current.index(PYTHON_BEGIN), current.index(PYTHON_END) + len(PYTHON_END)
    return current[:start] + python_signatures(header) + current[end:]


def rendered() -> dict:
    """Each generated file and the text the header gives it."""
    header = Header(HEADER.read_text())
    return {
        PYTHON_NATIVE: render_python(header, PYTHON_NATIVE.read_text()),
        CSHARP_METHODS: csharp_methods(header),
    }


def stale() -> list:
    return [path for path, text in rendered().items() if not path.exists() or path.read_text() != text]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a generated file is stale")
    args = parser.parse_args()
    if args.check:
        outdated = stale()
        for path in outdated:
            print(f"{path.relative_to(ROOT)} is stale; run bindings/scripts/generate_native.py", file=sys.stderr)
        return 1 if outdated else 0
    for path, text in rendered().items():
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
