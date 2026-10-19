import re

MASS = "mass"
COUNT = "count"

_UNITS = {
    "": (COUNT, 1),
    "item": (COUNT, 1),
    "items": (COUNT, 1),
    "dozen": (COUNT, 10),
    "g": (MASS, 1),
    "kg": (MASS, 100),
}

_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-z]*)\s*$")


def parse_quantity(text):
    """Returns (kind, amount) with amount in the kind's base unit (items or grams)."""
    match = _PATTERN.match(text.lower())
    if not match:
        raise ValueError(f"not a quantity: {text!r}")
    number, unit = match.groups()
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r} in {text!r}")
    kind, factor = _UNITS[unit]
    return kind, float(number) * factor
