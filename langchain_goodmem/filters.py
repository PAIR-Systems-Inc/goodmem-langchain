r"""Build GoodMem metadata filter expressions safely.

GoodMem filters are expressions evaluated by the server, not SQL. Putting a
caller's value into one with an f-string is both a filter-injection hole and a
correctness bug: ``x' OR '1'='1`` widens the filter to every row, and an
ordinary apostrophe produces a malformed expression. Build filters with this
module whenever a value comes from a user, a model or any other input you do
not control, and pass the result to ``GoodMemRetriever(filter=...)``, or pass
a mapping to ``GoodMemRetriever(metadata_filter=...)``.

The escaping and casting rules were verified live against GoodMem server
v1.0.320:

- A literal is single-quoted. ``'`` inside it escapes as ``\'`` and a
  backslash as ``\\``. SQL-style ``''`` doubling and double-quoted strings are
  both rejected with HTTP 400.
- A raw newline inside a literal is rejected, so control characters are
  refused here rather than sent.
- ``val()`` yields JSON, so a comparison must cast to the stored type:
  ``TEXT`` for strings, ``NUMERIC`` for numbers, ``BOOLEAN`` for booleans. The
  cast has to match. Comparing a boolean as ``TEXT`` is accepted with HTTP 200
  and matches nothing.
"""

import math
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

# Field names are restricted rather than escaped: the JSONPath member grammar
# is not worth quoting around.
_SAFE_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_OPERATORS = frozenset({">", ">=", "<", "<="})


class GoodMemFilterError(ValueError):
    """A filter cannot be expressed safely."""


def escape_literal(value: str) -> str:
    """Quote ``value`` as a GoodMem filter literal.

    Raises:
        GoodMemFilterError: ``value`` contains a control character, which the
            server rejects inside a literal.
    """
    if _CONTROL_CHARS.search(value):
        raise GoodMemFilterError(
            "Filter values cannot contain control characters (including newlines "
            "and tabs); the server rejects them inside a literal."
        )
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _check_field(field: str) -> str:
    if not isinstance(field, str) or not _SAFE_FIELD.fullmatch(field):
        raise GoodMemFilterError(
            f"Unsupported metadata field name {field!r}. Field names may contain "
            "letters, digits, underscore, dot and hyphen, and must start with a "
            "letter or underscore."
        )
    return field


def _accessor(field: str, cast: str) -> str:
    return f"CAST(val('$.{_check_field(field)}') AS {cast})"


def _number(value: int | float) -> str:
    # repr() would emit "nan", "inf" or exponent notation such as "1e+20".
    # The server rejects the first two with a parse error, so render finite
    # numbers as plain decimals and refuse the rest.
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GoodMemFilterError("Filter numbers must be finite.")
        return format(Decimal(repr(value)), "f")
    return str(value)


def _render(value: Any) -> tuple[str, str]:
    # bool is checked before int because bool is a subclass of int, and a
    # stringified boolean produces a filter that matches nothing.
    if isinstance(value, bool):
        return "BOOLEAN", "true" if value else "false"
    if isinstance(value, (int, float)):
        return "NUMERIC", _number(value)
    if isinstance(value, str):
        return "TEXT", escape_literal(value)
    raise GoodMemFilterError(
        f"Unsupported filter value type {type(value).__name__}; use str, int, "
        "float or bool."
    )


def equals(field: str, value: str | int | float | bool) -> str:
    """Match memories whose metadata ``field`` equals ``value``.

    The comparison is cast to the type of ``value``: ``TEXT``, ``NUMERIC`` or
    ``BOOLEAN``.
    """
    cast, literal = _render(value)
    return f"{_accessor(field, cast)} = {literal}"


def not_equals(field: str, value: str | int | float | bool) -> str:
    """Match memories whose metadata ``field`` does not equal ``value``."""
    cast, literal = _render(value)
    return f"{_accessor(field, cast)} != {literal}"


def compare(field: str, operator: str, value: int | float) -> str:
    """Compare a numeric metadata ``field`` with ``>``, ``>=``, ``<`` or ``<=``."""
    if operator not in _OPERATORS:
        raise GoodMemFilterError(f"Unsupported comparison operator {operator!r}.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoodMemFilterError("Ordering comparisons apply to numbers only.")
    return f"{_accessor(field, 'NUMERIC')} {operator} {_number(value)}"


def one_of(field: str, values: Iterable[str | int | float | bool]) -> str:
    """Match memories whose metadata ``field`` is any of ``values``.

    All values must share one type.
    """
    rendered: list[str] = []
    casts: set[str] = set()
    for value in values:
        cast, literal = _render(value)
        casts.add(cast)
        rendered.append(literal)
    if not rendered:
        raise GoodMemFilterError("one_of() needs at least one value.")
    if len(casts) > 1:
        raise GoodMemFilterError("one_of() values must all be of the same type.")
    return f"{_accessor(field, casts.pop())} IN ({', '.join(rendered)})"


def all_of(*expressions: str | None) -> str:
    """Combine expressions with ``AND``. ``None`` and empty ones are skipped."""
    parts = [e for e in expressions if e]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return " AND ".join(f"({p})" for p in parts)


def any_of(*expressions: str | None) -> str:
    """Combine expressions with ``OR``. ``None`` and empty ones are skipped."""
    parts = [e for e in expressions if e]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return " OR ".join(f"({p})" for p in parts)


def from_mapping(metadata: Mapping[str, str | int | float | bool] | None) -> str:
    """Build an ``AND`` of equalities, one per field, or ``""`` if empty.

    ``{"team": "blue", "archived": False}`` matches memories whose ``team`` is
    the text ``blue`` and whose ``archived`` flag is the boolean ``false``.
    """
    if not metadata:
        return ""
    return all_of(*(equals(key, metadata[key]) for key in sorted(metadata)))
