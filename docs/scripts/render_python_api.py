#!/usr/bin/env python3
"""
Render the inference_rs Python package (bindings/python/inference_rs) as Starlight Markdown pages.

The package's source is the single source of truth for the Python API; its typed classes are generated from the
server's OpenAPI document. This script parses the modules with `ast` and writes one Markdown file per logical group
into `docs/src/content/docs/reference/python/`. The Starlight sidebar picks them up via its `autogenerate` rule.

Run from the repo root or the docs directory.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from pathlib import Path
from textwrap import dedent

SCRIPT_DIR = Path(__file__).resolve().parent
WEBSITE_DIR = SCRIPT_DIR.parent
REPO_DIR = WEBSITE_DIR.parent
PACKAGE_DIR = REPO_DIR / "bindings" / "python" / "inference_rs"
OUT_DIR = WEBSITE_DIR / "src" / "content" / "docs" / "reference" / "python"
STUB_REL = "bindings/python/inference_rs"

# (title, slug, description, rule): a rule picks the classes (and type aliases) a page covers, by name.
GROUPS = [
    (
        "Engine",
        "engine",
        "Load a model and serve requests; streams, results, errors and host callbacks.",
        lambda n: (
            n
            in {
                "Engine",
                "JsonEngine",
                "Stream",
                "StreamEvent",
                "MediaAttachment",
                "SkillFile",
                "Blob",
                "InferenceError",
                "Status",
                "HostCallbacks",
                "HostTool",
                "HostToolCall",
            }
        ),
    ),
    (
        "Engine spec",
        "spec",
        "What to load and how to run it: EngineSpec, the ModelSelected variants and their options.",
        lambda n: (
            n.startswith(
                (
                    "EngineSpec",
                    "ModelSpec",
                    "RuntimeSpec",
                    "AgenticSpec",
                    "AdapterSpec",
                    "SkillsSpec",
                    "ModelSelected",
                    "AnyMoe",
                    "Mcp",
                    "Mtp",
                    "PagedCache",
                    "CodeExecution",
                    "ShellConfig",
                    "SandboxPolicy",
                    "SandboxMode",
                    "NetworkMode",
                    "SearchSpec",
                    "SearchEmbeddingModel",
                )
            )
            or n.endswith("LoaderType")
            or n
            in {
                "ModelDType",
                "IsqOrganization",
                "UqffWriteSpec",
                "UqffWriteSpecConfig",
                "LoraAdapterSpec",
                "LoraRuntimeConfig",
                "AgentPermission",
            }
        ),
    ),
    (
        "Chat and completions",
        "chat",
        "Chat completion, completion and embedding requests and responses, tools and output formats.",
        lambda n: n.startswith(
            (
                "ChatCompletion",
                "Completion",
                "Embedding",
                "Message",
                "Tool",
                "Function",
                "ResponseFormat",
                "JsonSchema",
                "Grammar",
                "StopTokens",
                "WebSearch",
                "ApproximateUserLocation",
                "SearchContextSize",
                "OpenAi",
                "NamedFunction",
                "AllowedTool",
                "BuiltinTool",
                "ReasoningEffort",
                "PromptTokens",
                "AdapterGeneration",
                "AdapterSelection",
                "SerializedVideo",
            )
        ),
    ),
    (
        "Responses",
        "responses",
        "OpenResponses requests, resources and stream events.",
        lambda n: n.startswith(
            (
                "OpenResponses",
                "Response",
                "Output",
                "IncompleteDetails",
                "IncompleteReason",
                "InputTokens",
                "IncludeOption",
                "ReasoningConfig",
                "ReasoningSummary",
                "TextConfig",
                "TextFormat",
                "StreamOptions",
                "TruncationStrategy",
                "UrlCitation",
                "FileCitation",
                "FilePathInfo",
            )
        ),
    ),
    (
        "Anthropic",
        "anthropic",
        "Anthropic Messages requests, responses and skill listings.",
        lambda n: n.startswith("Anthropic"),
    ),
    (
        "Models, adapters, files and skills",
        "management",
        "Model status, LoRA adapters, files, skills, approvals, sessions, calibration, tokenization and the media generation calls.",
        lambda n: n.startswith(
            (
                "Model",
                "Lora",
                "LoadLora",
                "UnloadLora",
                "File",
                "ContainerFile",
                "SourceMeta",
                "Skill",
                "Approval",
                "ImageGeneration",
                "ImageChoice",
                "SpeechGeneration",
                "AudioResponseFormat",
                "Calibration",
                "ReIsq",
                "Tune",
                "Serialized",
                "Session",
                "Tokenize",
                "Detokenize",
            )
        ),
    ),
    (
        "Layout",
        "layout",
        "PP-DocLayoutV3 document layout detection.",
        lambda n: n in {"LayoutModel", "LayoutImage", "LayoutDetection", "PixelFormat"},
    ),
]

SIG_WRAP_THRESHOLD = 70  # wrap arg list onto multiple lines beyond this width


def _unparse(node) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _field_default(node: ast.expr | None) -> str:
    if not isinstance(node, ast.Call):
        return _unparse(node) if node is not None else ""
    if not isinstance(node.func, ast.Name) or node.func.id != "field":
        return _unparse(node)

    values = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    default = values.get("default")
    if default is None and "default_factory" in values:
        rendered = f"factory: {_unparse(values['default_factory'])}"
    else:
        rendered = _unparse(default)
    if (
        isinstance(values.get("kw_only"), ast.Constant)
        and values["kw_only"].value is True
    ):
        return f"{rendered} (keyword-only)"
    return rendered


def _is_enum(cls: ast.ClassDef) -> bool:
    return any(isinstance(b, ast.Name) and b.id == "Enum" for b in cls.bases)


def _collect_args(func: ast.FunctionDef) -> list[tuple[str, str, str | None]]:
    """Return [(name, annotation, default_str_or_None)] skipping `self`."""
    out: list[tuple[str, str, str | None]] = []
    args = func.args
    defaults = list(args.defaults)
    positional = list(args.args)
    pad = len(positional) - len(defaults)
    for i, arg in enumerate(positional):
        if arg.arg == "self":
            continue
        ann = _unparse(arg.annotation)
        default_idx = i - pad
        default = _unparse(defaults[default_idx]) if default_idx >= 0 else None
        out.append((arg.arg, ann, default))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        ann = _unparse(arg.annotation)
        d = _unparse(default) if default is not None else None
        out.append((arg.arg, ann, d))
    return out


def _format_signature_block(func_name: str, func: ast.FunctionDef) -> str:
    """Format a signature as a Python code block, wrapping onto multiple lines
    when the single-line form exceeds SIG_WRAP_THRESHOLD."""
    ret = _unparse(func.returns)
    args = _collect_args(func)

    def fmt(a: tuple[str, str, str | None]) -> str:
        name, ann, default = a
        s = name
        if ann:
            s += f": {ann}"
        if default is not None:
            s += f" = {default}"
        return s

    parts = [fmt(a) for a in args]
    if func.args.kwonlyargs:
        positional_count = sum(arg.arg != "self" for arg in func.args.args)
        parts.insert(positional_count, "*")
    single = f"{func_name}({', '.join(parts)})"
    if ret:
        single += f" -> {ret}"

    if len(single) <= SIG_WRAP_THRESHOLD:
        return single

    indent = "    "
    multi = [f"{func_name}("]
    for p in parts:
        multi.append(f"{indent}{p},")
    closing = ")"
    if ret:
        closing += f" -> {ret}"
    multi.append(closing)
    return "\n".join(multi)


# Matches a docstring "Args:" block and captures indented argument descriptions.
ARGS_SECTION_RE = re.compile(r"(?m)^[ \t]*Args?\s*:\s*\n(?P<body>(?:[ \t]+[^\n]*\n?)+)")
RETURNS_SECTION_RE = re.compile(
    r"(?m)^[ \t]*Returns?\s*:\s*\n(?P<body>(?:[ \t]+[^\n]*\n?)+)"
)
RAISES_SECTION_RE = re.compile(
    r"(?m)^[ \t]*Raises\s*:\s*\n(?P<body>(?:[ \t]+[^\n]*\n?)+)"
)


def _parse_doc_sections(doc: str) -> tuple[str, list[tuple[str, str]], str, str]:
    """Pull Args/Returns/Raises out of a docstring.

    Returns (summary, params, returns_text, raises_text).
    `summary` is everything before the first recognized section.
    `params` is [(name, description)] extracted from Args:.
    """
    if not doc:
        return "", [], "", ""

    sections = []
    for pat in (ARGS_SECTION_RE, RETURNS_SECTION_RE, RAISES_SECTION_RE):
        m = pat.search(doc)
        if m:
            sections.append(m.start())
    cut = min(sections) if sections else len(doc)
    summary = doc[:cut].strip()

    params: list[tuple[str, str]] = []
    m = ARGS_SECTION_RE.search(doc)
    if m:
        body = dedent(m.group("body")).strip("\n")
        current_name: str | None = None
        current_desc: list[str] = []
        for line in body.splitlines():
            stripped = line.rstrip()
            if not stripped:
                continue
            lead_ws = len(line) - len(line.lstrip(" "))
            if lead_ws == 0 and ":" in stripped:
                if current_name is not None:
                    params.append((current_name, " ".join(current_desc).strip()))
                name, _, rest = stripped.partition(":")
                current_name = name.strip()
                current_desc = [rest.strip()]
            else:
                current_desc.append(stripped.strip())
        if current_name is not None:
            params.append((current_name, " ".join(current_desc).strip()))

    returns_text = ""
    m = RETURNS_SECTION_RE.search(doc)
    if m:
        returns_text = dedent(m.group("body")).strip()

    raises_text = ""
    m = RAISES_SECTION_RE.search(doc)
    if m:
        raises_text = dedent(m.group("body")).strip()

    return summary, params, returns_text, raises_text


def _clean_doc(doc: str | None) -> str:
    if not doc:
        return ""
    return dedent(doc).strip()


def _md_escape_cell(text: str) -> str:
    """Escape a string for safe inclusion in a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _md_code_cell(text: str) -> str:
    """Inline-code a value and escape pipes so it survives a table cell."""
    if not text:
        return ""
    # Pipes inside backticks still end the cell in CommonMark tables; escape them.
    return "`" + text.replace("|", "\\|") + "`"


def _render_params_table(
    args: list[tuple[str, str, str | None]],
    param_docs: dict[str, str],
) -> str:
    lines = [
        "**Parameters**",
        "",
        "| Name | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for name, ann, default in args:
        type_cell = _md_code_cell(ann)
        default_cell = _md_code_cell(default) if default is not None else "required"
        desc = _md_escape_cell(param_docs.get(name, ""))
        lines.append(f"| `{name}` | {type_cell} | {default_cell} | {desc} |")
    lines.append("")
    return "\n".join(lines)


def _render_function(
    func: ast.FunctionDef, heading: str, owner: str | None = None
) -> str:
    display_name = "__init__" if func.name == "__init__" else func.name
    anchor = f"{owner}.{display_name}" if owner else display_name

    sig_block = _format_signature_block(func.name, func)
    doc = _clean_doc(ast.get_docstring(func))
    summary, param_docs_list, returns_text, raises_text = _parse_doc_sections(doc)
    param_docs = {name: desc for name, desc in param_docs_list}

    args = _collect_args(func)
    lines = [f"{heading} `{anchor}`", ""]
    lines.append("```text")
    lines.append(sig_block)
    lines.append("```")
    lines.append("")

    if summary:
        lines.append(summary)
        lines.append("")

    if args:
        if param_docs:
            lines.append(_render_params_table(args, param_docs))

    if returns_text:
        lines.append(f"**Returns:** {returns_text}")
        lines.append("")

    if raises_text:
        lines.append(f"**Raises:** {raises_text}")
        lines.append("")

    return "\n".join(lines)


def _default_cell(value: str | None) -> str:
    if not value:
        return "required"
    return "optional" if value == "None" else _md_code_cell(value)


def _render_fields_table(owner: str, fields: list[tuple[str, str, str]]) -> str:
    has_default = any(v for _, _, v in fields)
    if has_default:
        lines = [
            "| Field | Type | Default |",
            "| --- | --- | --- |",
        ]
        for name, ftype, value in fields:
            t = _md_code_cell(ftype)
            lines.append(f"| `{name}` | {t} | {_default_cell(value)} |")
    else:
        lines = [
            "| Field | Type |",
            "| --- | --- |",
        ]
        for name, ftype, _ in fields:
            t = _md_code_cell(ftype)
            lines.append(f"| `{name}` | {t} |")
    lines.append("")
    return "\n".join(lines)


def _render_enum_table(owner: str, values: list[tuple[str, str]]) -> str:
    has_value = any(v for _, v in values)
    if has_value:
        lines = [
            "Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.",
            "",
            "| Member | Wire/config name |",
            "| --- | --- |",
        ]
        for name, value in values:
            v = _md_code_cell(value)
            lines.append(f"| `{owner}.{name}` | {v} |")
    else:
        lines = [
            "| Member |",
            "| --- |",
        ]
        for name, _ in values:
            lines.append(f"| `{owner}.{name}` |")
    lines.append("")
    return "\n".join(lines)


def _render_class(
    cls: ast.ClassDef, heading: str = "###", parent: str | None = None
) -> str:
    is_enum = _is_enum(cls)
    full_name = f"{parent}.{cls.name}" if parent else cls.name
    lines: list[str] = [f"{heading} `{full_name}`", ""]
    doc = _clean_doc(ast.get_docstring(cls))
    if doc:
        lines.append(doc)
        lines.append("")

    fields: list[tuple[str, str, str]] = []
    enum_values: list[tuple[str, str]] = []
    methods: list[ast.FunctionDef] = []
    nested: list[ast.ClassDef] = []

    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            ftype = _unparse(item.annotation)
            value = _field_default(item.value)
            fields.append((name, ftype, value))
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    enum_values.append((t.id, _unparse(item.value)))
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(item)
        elif isinstance(item, ast.ClassDef):
            nested.append(item)

    if is_enum:
        values = enum_values + [(n, v) for n, _, v in fields if v]
        if values:
            lines.append(_render_enum_table(cls.name, values))
        for nc in nested:
            sub_heading = "#" * (len(heading) + 1)
            lines.append(_render_class(nc, sub_heading, parent=cls.name))
    else:
        if fields:
            lines.append(_render_fields_table(full_name, fields))
        for nc in nested:
            sub_heading = "#" * (len(heading) + 1)
            lines.append(_render_class(nc, sub_heading, parent=full_name))

    init = [m for m in methods if m.name == "__init__"]
    rest = [m for m in methods if m.name != "__init__"]
    sub_heading = "#" * (len(heading) + 1)
    for m in init + rest:
        lines.append(_render_function(m, sub_heading, owner=full_name))

    return "\n".join(lines)


def _render_alias(name: str, value: ast.expr) -> str:
    return "\n".join([f"## `{name}`", "", f"One of: `{_unparse(value)}`.", ""])


def _render_page(
    title: str,
    description: str,
    class_names: list[str],
    classes_by_name: dict[str, ast.ClassDef],
    order: int,
    aliases: dict[str, ast.expr],
) -> str:
    safe_description = description.replace('"', '\\"')
    frontmatter = [
        "---",
        f"title: {title}",
        f'description: "{safe_description}"',
        "sidebar:",
        f"  order: {order}",
        "---",
        "",
    ]
    body: list[str] = []
    for name in class_names:
        if name in aliases:
            body.append(_render_alias(name, aliases[name]))
        else:
            body.append(_render_class(classes_by_name[name], heading="##"))
        body.append("")

    footer = [
        "---",
        "",
        f"<small>Generated from [`{STUB_REL}`](https://github.com/christopherthompson81/inference.rs/blob/master/{STUB_REL}).</small>",
        "",
    ]

    return "\n".join(frontmatter) + "\n".join(body) + "\n".join(footer)


def _render_index() -> str:
    lines = [
        "---",
        "title: Python API",
        'description: "The inference_rs Python package."',
        "sidebar:",
        "  order: 6",
        "---",
        "",
        "The `inference_rs` Python package runs the same engine as the `inference` CLI, through its C ABI "
        "(`libinference_ffi`). Specs, requests and responses are dataclasses generated from the server's OpenAPI "
        "document, so they match what the engine accepts.",
        "",
        "## Install",
        "",
        "inference.rs does not publish wheels yet; build the library from a checkout (add `--features cuda` or "
        "`metal`) and install the package. See [Python SDK getting started](/guides/python/getting-started/#installing) "
        "and [hardware support](/reference/hardware-support/).",
        "",
        "```bash",
        "cargo build --release -p inference-ffi",
        "pip install -e bindings/python",
        "```",
        "",
        "## Pages",
        "",
        "| Page | Covers |",
        "| --- | --- |",
    ]
    for title, slug, desc, _ in GROUPS:
        lines.append(f"| [{title}](/reference/python/{slug}/) | {desc} |")
    lines.append("")
    lines.append(
        "See [Python getting started](/guides/python/getting-started/) for a walkthrough and the [Python guides](/guides/python/) for task-oriented recipes."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"<small>Generated from [`{STUB_REL}`](https://github.com/christopherthompson81/inference.rs/blob/master/{STUB_REL}).</small>"
    )
    lines.append("")
    return "\n".join(lines)


def _collect(
    trees: list[ast.Module],
) -> tuple[dict[str, ast.ClassDef], dict[str, ast.expr]]:
    """Public classes, and module-level type aliases (`Name = Union[...]`), across the package's modules."""
    classes: dict[str, ast.ClassDef] = {}
    aliases: dict[str, ast.expr] = {}
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[node.name] = node
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id[:1].isupper()
                and isinstance(node.value, ast.Subscript)
            ):
                aliases[node.targets[0].id] = node.value
    return classes, aliases


def _exported() -> set[str]:
    """The names `inference_rs/__init__.py` lists in `__all__`."""
    for node in ast.parse((PACKAGE_DIR / "__init__.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            return {elt.value for elt in node.value.elts}
    return set()


def _render_pages(
    classes_by_name: dict[str, ast.ClassDef], aliases: dict[str, ast.expr]
) -> tuple[dict, list]:
    pages = {"index.md": _render_index()}
    names = sorted(set(classes_by_name) | set(aliases))
    placed: set[str] = set()
    for i, (title, slug, desc, rule) in enumerate(GROUPS, start=2):
        members = [n for n in names if n not in placed and rule(n)]
        placed |= set(members)
        pages[f"{slug}.md"] = _render_page(
            title, desc, members, classes_by_name, i, aliases
        )
    uncovered = [n for n in names if n not in placed]
    return pages, uncovered


def _check(pages: dict[str, str]) -> int:
    committed = {p.name: p.read_text() for p in OUT_DIR.glob("*.md")}
    drift = False
    for name in sorted(set(pages) - set(committed)):
        print(f"missing: {OUT_DIR / name}", file=sys.stderr)
        drift = True
    for name in sorted(set(committed) - set(pages)):
        print(f"stale: {OUT_DIR / name} (no longer generated)", file=sys.stderr)
        drift = True
    for name in sorted(set(pages) & set(committed)):
        if pages[name] != committed[name]:
            drift = True
            diff = difflib.unified_diff(
                committed[name].splitlines(keepends=True),
                pages[name].splitlines(keepends=True),
                fromfile=f"committed/{name}",
                tofile=f"generated/{name}",
            )
            sys.stderr.writelines(diff)
    if drift:
        print(
            f"error: generated Python reference is out of date with {STUB_REL}; "
            "run `python docs/scripts/render_python_api.py` and commit the result",
            file=sys.stderr,
        )
        return 1
    print(f"ok: {OUT_DIR} is up to date with {STUB_REL}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed pages match the package instead of writing",
    )
    opts = parser.parse_args()

    classes_by_name, aliases = _collect(
        [ast.parse(path.read_text()) for path in sorted(PACKAGE_DIR.glob("*.py"))]
    )
    # The private modules contribute only what the package exports; `types` is public as a whole.
    exported = _exported()
    generated_classes, generated_aliases = _collect(
        [ast.parse((PACKAGE_DIR / "types.py").read_text())]
    )
    classes_by_name = {
        n: c for n, c in classes_by_name.items() if n in exported
    } | generated_classes
    aliases = {n: a for n, a in aliases.items() if n in exported} | generated_aliases
    pages, uncovered = _render_pages(classes_by_name, aliases)
    if uncovered:
        # Every public name belongs on a page; a new generated type needs a group rule.
        print(
            "error: no reference page covers: " + ", ".join(uncovered), file=sys.stderr
        )
        return 1

    if opts.check:
        return _check(pages)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean previous output so removed classes do not linger.
    for existing in OUT_DIR.glob("*.md"):
        existing.unlink()
    for name, content in pages.items():
        path = OUT_DIR / name
        path.write_text(content)
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
