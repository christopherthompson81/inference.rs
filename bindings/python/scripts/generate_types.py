"""Generates inference_rs/types.py from the server's OpenAPI document (docs/openapi.json).

Usage: python3 bindings/python/scripts/generate_types.py [--check]
"""

import json
import keyword
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPENAPI = ROOT / "docs" / "openapi.json"
OUTPUT = ROOT / "bindings" / "python" / "inference_rs" / "types.py"
REF_PREFIX = "#/components/schemas/"

HEADER = '''"""Typed request and response classes, generated from docs/openapi.json by scripts/generate_types.py; do not edit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Union

'''


def class_name(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[^A-Za-z0-9]+", value) if part)


def attribute(name: str) -> str:
    """A property's Python attribute: keywords and invalid names get a trailing underscore or a cleaned form."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if cleaned[:1].isdigit():
        cleaned = "_" + cleaned
    return cleaned + "_" if keyword.iskeyword(cleaned) else cleaned


def summary(schema: dict) -> str | None:
    description = (schema.get("description") or "").strip()
    if not description:
        return None
    first = description.split("\n\n")[0].replace("\n", " ").strip()
    return first.replace("\\", "\\\\").replace('"', '\\"')


class Generator:
    def __init__(self, schemas: dict):
        self.schemas = schemas
        self.blocks = []
        self.aliases = {}
        self.names = set(schemas)

    def type_of(self, schema: dict, owner: str, prop: str) -> str:
        """The annotation for a schema; inline tagged objects become named classes."""
        if "$ref" in schema:
            return schema["$ref"][len(REF_PREFIX) :]
        for key in ("oneOf", "anyOf"):
            if key in schema:
                members = [m for m in schema[key] if m.get("type") != "null"]
                nullable = len(members) < len(schema[key])
                union = self.union(members, owner, prop)
                return f"{union} | None" if nullable else union
        kind = schema.get("type")
        nullable = False
        if isinstance(kind, list):
            nullable = "null" in kind
            kinds = [k for k in kind if k != "null"]
            kind = kinds[0] if len(kinds) == 1 else None
        if "enum" in schema and kind == "string":
            values = [v for v in schema["enum"] if v is not None]
            nullable = nullable or len(values) < len(schema["enum"])
            base = "Literal[" + ", ".join(json.dumps(v) for v in values) + "]"
        elif kind == "string":
            base = "str"
        elif kind == "integer":
            base = "int"
        elif kind == "number":
            base = "float"
        elif kind == "boolean":
            base = "bool"
        elif kind == "array":
            base = f"list[{self.type_of(schema.get('items') or {}, owner, prop)}]"
        elif kind == "object" and "properties" in schema:
            base = self.inline_object(schema, owner, prop)
        elif kind == "object":
            values = schema.get("additionalProperties")
            base = (
                f"dict[str, {self.type_of(values, owner, prop)}]"
                if isinstance(values, dict) and values
                else "dict[str, Any]"
            )
        else:
            base = "Any"
        return f"{base} | None" if nullable and base != "Any" else base

    @staticmethod
    def external_tag(schema: dict) -> str | None:
        """The key of an externally tagged variant, `{"Key": {...fields}}`, whose fields become the class."""
        properties = schema.get("properties") or {}
        if schema.get("type") != "object" or len(properties) != 1 or schema.get("required") != list(properties):
            return None
        ((key, value),) = properties.items()
        return key if value.get("type") == "object" and "properties" in value else None

    def union(self, members: list, owner: str, prop: str) -> str:
        types = []
        for index, member in enumerate(members):
            key = self.external_tag(member)
            if key is not None:
                name = class_name(owner) + class_name(prop) + class_name(key)
                self.names.add(name)
                self.object_class(name, member["properties"][key], external=key)
                types.append(name)
                continue
            # A variant is named by its tag, else its schema title, else its position.
            tag = self.tag(member) or member.get("title")
            member_prop = class_name(tag) if tag else f"{prop}{index}"
            types.append(self.type_of(member, owner, member_prop))
        unique = list(dict.fromkeys(types))
        return unique[0] if len(unique) == 1 else "Union[" + ", ".join(unique) + "]"

    @staticmethod
    def tag(schema: dict) -> str | None:
        """The value of a single-valued `type` property, which tags a union's object variants."""
        tag = (schema.get("properties") or {}).get("type") or {}
        values = tag.get("enum") or []
        return values[0] if len(values) == 1 else None

    def inline_object(self, schema: dict, owner: str, prop: str) -> str:
        name = class_name(owner) + class_name(prop)
        while name in self.names:
            name += "_"
        self.names.add(name)
        self.object_class(name, schema)
        return name

    def tag_value(self, schema: dict) -> str | None:
        """The one value a property can take: an inline single-valued enum, or a reference to one."""
        if "$ref" in schema:
            schema = self.schemas.get(schema["$ref"][len(REF_PREFIX) :], {})
        values = schema.get("enum") or []
        return values[0] if len(values) == 1 and isinstance(values[0], str) else None

    def default_literal(self, schema: dict) -> str:
        """The engine's default for an optional field as Python source; `None` when it has none or it isn't a scalar."""
        value = schema.get("default")
        if isinstance(value, (bool, int, float)):
            return repr(value)
        if not isinstance(value, str):
            return "None"
        refs = [s["$ref"] for s in [schema, *schema.get("allOf", []), *schema.get("oneOf", [])] if "$ref" in s]
        if not refs:
            return json.dumps(value)
        enum = refs[0][len(REF_PREFIX) :]
        if value not in (self.schemas.get(enum, {}).get("enum") or []):
            return "None"
        return f"{enum}.{attribute(value.upper()) if value else 'EMPTY'}"

    def object_class(self, name: str, schema: dict, external: str | None = None) -> None:
        required = set(schema.get("required") or [])
        # Keyword-only, so a regenerated field order cannot silently move positional arguments.
        lines = ["@dataclass(kw_only=True)", f"class {name}:"]
        doc = summary(schema)
        if doc:
            lines.append(f'    """{doc}"""')
            lines.append("")
        wire, seen = {}, set()
        properties = sorted((schema.get("properties") or {}).items())
        for prop, prop_schema in properties:
            python = attribute(prop)
            if python in seen:
                raise ValueError(f"{name}: two properties become the attribute {python}")
            seen.add(python)
            if python != prop:
                wire[python] = prop
            annotation = self.type_of(prop_schema, name, class_name(prop))
            # A single-valued `type` is a variant's tag, so it defaults to its one value.
            tag = self.tag_value(prop_schema) if prop == "type" else None
            if tag is not None:
                default = f" = {json.dumps(tag)}"
            elif prop in required:
                default = ""
            else:
                default = f" = {self.default_literal(prop_schema)}"
                if "| None" not in annotation and annotation != "Any":
                    annotation = f"{annotation} | None"
            lines.append(f"    {python}: {annotation}{default}")
        if not properties:
            lines.append("    pass")
        if wire:
            lines.append(f"    _wire = {wire!r}")
        if external is not None:
            # Serialized as {"Key": {...fields}}.
            lines.append(f"    _external = {external!r}")
        self.blocks.append("\n".join(lines))

    def schema(self, name: str, schema: dict) -> None:
        if schema.get("type") == "string" and "enum" in schema:
            lines = [f"class {name}(str, Enum):"]
            doc = summary(schema)
            if doc:
                lines.append(f'    """{doc}"""')
                lines.append("")
            for value in schema["enum"]:
                member = attribute(value.upper()) if value else "EMPTY"
                lines.append(f"    {member} = {json.dumps(value)}")
            self.blocks.append("\n".join(lines))
        elif schema.get("type") == "object" and "oneOf" not in schema and "anyOf" not in schema:
            self.object_class(name, schema)
        else:
            self.aliases[name] = self.type_of(schema, name, "")

    def ordered_aliases(self) -> list:
        """Aliases after the aliases they mention, since an alias is evaluated when it is defined."""
        done, order = set(), []

        def visit(name):
            if name in done:
                return
            done.add(name)
            for other in self.aliases:
                if other != name and re.search(rf"\b{other}\b", self.aliases[name]):
                    visit(other)
            order.append(name)

        for name in sorted(self.aliases):
            visit(name)
        return order

    def render(self) -> str:
        for name in sorted(self.schemas):
            self.schema(name, self.schemas[name])
        aliases = [f"{name} = {self.aliases[name]}" for name in self.ordered_aliases()]
        return HEADER + "\n\n\n".join(self.blocks) + "\n\n\n" + "\n".join(aliases) + "\n"


def generate() -> str:
    schemas = json.loads(OPENAPI.read_text())["components"]["schemas"]
    return Generator(schemas).render()


def main() -> int:
    text = generate()
    if "--check" in sys.argv[1:]:
        if OUTPUT.read_text() != text:
            print(f"{OUTPUT} is stale; run {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
