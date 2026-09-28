"""The examples, notebooks and guide snippets use only names the package defines, so a renamed type cannot break them."""

import ast
import dataclasses
import enum
import inspect
import json
import re
import sys
import types as pytypes
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_rs as ir

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "examples" / "python"
GUIDES = REPO / "docs" / "src" / "content" / "docs"
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
PACKAGE = "inference_rs"
# Sources that still need engine features the C ABI does not offer yet (AnyMoE, MCP client, code execution, shell and
# calibration configuration); they keep the pyo3 API until it does.
PYO3_SOURCES = {
    "anymoe.py",
    "anymoe_inference.py",
    "anymoe_lora.py",
    "code_execution.py",
    "code_execution_approval.py",
    "mcp_client.py",
    "online_calibration.py",
    "shell.py",
    "shell_skills.py",
    "guides/agents/connect-mcp-server.mdx",
    "guides/agents/enable-code-execution.mdx",
    "guides/agents/enable-shell.mdx",
    "guides/agents/skills.mdx",
}


def sources():
    """(name, source) for every example script, notebook and page of Python guide snippets that imports the package."""
    for path in sorted(EXAMPLES.glob("*.py")):
        yield path.name, path.read_text()
    for path in sorted(EXAMPLES.glob("*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
        # Magics are not Python.
        lines = [line for line in "\n".join(code).splitlines() if not line.lstrip().startswith(("%", "!"))]
        yield path.name, "\n".join(lines)
    for path in sorted([*GUIDES.rglob("*.md"), *GUIDES.rglob("*.mdx")]):
        if "examples" in path.relative_to(GUIDES).parts:
            continue
        # A page's snippets share one namespace: later ones use the engine an earlier one made.
        blocks = PYTHON_BLOCK.findall(path.read_text())
        if any(PACKAGE in block for block in blocks):
            yield str(path.relative_to(GUIDES)), "\n".join(blocks)


def owned(value) -> bool:
    """Whether attributes read from `value` are the package's to define."""
    if isinstance(value, pytypes.ModuleType):
        return value.__name__.startswith(PACKAGE)
    return inspect.isclass(value) and value.__module__.startswith(PACKAGE)


class Resolver:
    """Binds names to package objects and reports the reads and constructions the package cannot satisfy."""

    def __init__(self, tree: ast.AST):
        self.bound = {}
        self.missing = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] != PACKAGE:
                        continue
                    if alias.asname:
                        self.bind(alias.asname, self.module(alias.name))
                    else:
                        self.bound[PACKAGE] = ir
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == PACKAGE:
                module = self.module(node.module)
                for alias in node.names:
                    if module is not None and not hasattr(module, alias.name):
                        self.missing.append(f"{node.module}.{alias.name}")
                    else:
                        self.bind(alias.asname or alias.name, getattr(module, alias.name, None))
        # Engines are bound where they are made, from `with ... as engine`, `engine = ...` or an annotation.
        for node in ast.walk(tree):
            if isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                self.bind_instance(node.optional_vars.id, node.context_expr)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.bind_instance(node.targets[0].id, node.value)
            elif (
                isinstance(node, ast.arg) and node.annotation is not None and self.resolve(node.annotation) is ir.Engine
            ):
                self.bind(node.arg, ir.Engine)

    @staticmethod
    def module(name: str):
        value = ir
        for part in name.split(".")[1:]:
            value = getattr(value, part, None)
        return value

    def bind(self, name: str, value) -> None:
        if value is not None:
            self.bound[name] = value

    def bind_instance(self, name: str, value: ast.expr) -> None:
        if isinstance(value, ast.Call) and self.resolve(value.func) is ir.Engine:
            self.bind(name, ir.Engine)

    def resolve(self, node: ast.expr):
        """The package object `node` names, or None when it names something else."""
        if isinstance(node, ast.Name):
            return self.bound.get(node.id)
        if isinstance(node, ast.Attribute):
            owner = self.resolve(node.value)
            if owner is not None and owned(owner):
                return getattr(owner, node.attr, None)
        return None

    def check(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                owner = self.resolve(node.value)
                # Enum members are read from the class; a dataclass's fields only exist on instances.
                readable = owner is not None and owned(owner)
                if readable and dataclasses.is_dataclass(owner) and not issubclass(owner, enum.Enum):
                    readable = False
                if readable and not hasattr(owner, node.attr):
                    self.missing.append(f"{getattr(owner, '__name__', owner)}.{node.attr}")
            elif isinstance(node, ast.Call):
                self.check_construction(node)

    def check_construction(self, node: ast.Call) -> None:
        cls = self.resolve(node.func)
        if not dataclasses.is_dataclass(cls) or node.args or any(k.arg is None for k in node.keywords):
            return
        fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
        passed = {k.arg for k in node.keywords}
        required = {
            name
            for name, f in fields.items()
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }
        for name in sorted(passed - fields.keys()):
            self.missing.append(f"line {node.lineno}: {cls.__name__} has no field {name}")
        for name in sorted(required - passed):
            self.missing.append(f"line {node.lineno}: {cls.__name__} requires {name}")


def problems(name: str, source: str) -> list[str]:
    tree = ast.parse(source, filename=name)
    resolver = Resolver(tree)
    resolver.check(tree)
    return sorted(set(resolver.missing))


class Examples(unittest.TestCase):
    def test_sources_use_names_and_fields_the_package_defines(self):
        broken = {}
        for name, source in sources():
            found = problems(name, source)
            if found and name not in PYO3_SOURCES:
                broken[name] = found
        self.assertEqual(broken, {})

    def test_the_listed_sources_still_need_the_pyo3_api(self):
        ported = [name for name, source in sources() if name in PYO3_SOURCES and not problems(name, source)]
        self.assertEqual(ported, [])

    def test_the_checks_catch_what_they_claim_to(self):
        source = """
import inference_rs
import inference_rs.types as types_module
from inference_rs.types import ModelSelectedPlain
from inference_rs import NoSuchThing

with inference_rs.Engine(spec) as engine:
    engine.no_such_method(request)
inference_rs.types.NoSuchType
types_module.NormalLoaderType.NO_SUCH_MEMBER
ModelSelectedPlain(bogus=1)
"""
        found = problems("probe", source)
        for expected in (
            "inference_rs.NoSuchThing",
            "Engine.no_such_method",
            "inference_rs.types.NoSuchType",
            "NormalLoaderType.NO_SUCH_MEMBER",
            "line 11: ModelSelectedPlain has no field bogus",
            "line 11: ModelSelectedPlain requires model_id",
        ):
            self.assertIn(expected, found)


if __name__ == "__main__":
    unittest.main()
