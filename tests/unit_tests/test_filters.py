"""Filter expressions are built, escaped and cast safely (P34)."""

import json
import re

import pytest
from pydantic import ValidationError

from langchain_goodmem import GoodMemRetriever, filters
from langchain_goodmem.filters import GoodMemFilterError
from tests.unit_tests.conftest import CHUNK, MEMORY, SPACE_ID, SPACE_ID_2, Wire, ndjson

INJECTION = "x' OR '1'='1"


def _unescaped_quotes(expression: str) -> int:
    return len(re.findall(r"(?<!\\)'", expression))


def test_injection_stays_inside_one_literal() -> None:
    expression = filters.equals("owner", INJECTION)
    assert expression == r"CAST(val('$.owner') AS TEXT) = 'x\' OR \'1\'=\'1'"
    # Two quotes around the field path, two around the value: nothing the
    # caller supplied is read as filter syntax.
    assert _unescaped_quotes(expression) == 4


def test_apostrophes_and_backslashes_escape_as_the_server_expects() -> None:
    assert filters.escape_literal("O'Brien") == r"'O\'Brien'"
    assert "''" not in filters.escape_literal("O'Brien")
    assert filters.escape_literal("back\\slash") == r"'back\\slash'"


@pytest.mark.parametrize("value", ["two\nlines", "tab\there", "nul\x00"])
def test_control_characters_are_refused(value: str) -> None:
    with pytest.raises(GoodMemFilterError, match="control characters"):
        filters.escape_literal(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("blue", "CAST(val('$.f') AS TEXT) = 'blue'"),
        (True, "CAST(val('$.f') AS BOOLEAN) = true"),
        (False, "CAST(val('$.f') AS BOOLEAN) = false"),
        (5, "CAST(val('$.f') AS NUMERIC) = 5"),
        (2.5, "CAST(val('$.f') AS NUMERIC) = 2.5"),
        (1e20, "CAST(val('$.f') AS NUMERIC) = 100000000000000000000"),
        (1.5e-7, "CAST(val('$.f') AS NUMERIC) = 0.00000015"),
    ],
)
def test_each_value_is_cast_to_its_own_type(
    value: str | int | float | bool, expected: str
) -> None:
    # A boolean compared as TEXT is accepted by the server and matches
    # nothing, so bool must be recognised before int.
    assert filters.equals("f", value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_refused(value: float) -> None:
    with pytest.raises(GoodMemFilterError, match="finite"):
        filters.equals("f", value)


@pytest.mark.parametrize("value", [None, [1], {"a": 1}, b"bytes"])
def test_values_without_a_safe_form_are_refused(value: object) -> None:
    with pytest.raises(GoodMemFilterError, match="Unsupported filter value type"):
        filters.equals("f", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["", "1st", "a b", "x') OR ('1'='1", "a;b"])
def test_unsafe_field_names_are_refused(field: str) -> None:
    with pytest.raises(GoodMemFilterError, match="Unsupported metadata field name"):
        filters.equals(field, "v")


def test_comparisons_and_sets() -> None:
    assert filters.compare("year", ">=", 2026) == (
        "CAST(val('$.year') AS NUMERIC) >= 2026"
    )
    assert filters.not_equals("state", "closed") == (
        "CAST(val('$.state') AS TEXT) != 'closed'"
    )
    assert filters.one_of("tier", ["gold", "silver"]) == (
        "CAST(val('$.tier') AS TEXT) IN ('gold', 'silver')"
    )
    with pytest.raises(GoodMemFilterError, match="Unsupported comparison"):
        filters.compare("year", "==", 1)
    with pytest.raises(GoodMemFilterError, match="numbers only"):
        filters.compare("flag", ">", True)
    with pytest.raises(GoodMemFilterError, match="same type"):
        filters.one_of("tier", ["gold", 1])
    with pytest.raises(GoodMemFilterError, match="at least one"):
        filters.one_of("tier", [])


def test_combinators_skip_empty_parts() -> None:
    assert filters.all_of("a", None, "", "b") == "(a) AND (b)"
    assert filters.any_of("a", "b") == "(a) OR (b)"
    assert filters.all_of("a") == "a"
    assert filters.all_of(None, "") == ""
    assert filters.from_mapping({}) == ""
    assert filters.from_mapping(None) == ""
    assert filters.from_mapping({"b": 2, "a": "one"}) == (
        "(CAST(val('$.a') AS TEXT) = 'one') AND (CAST(val('$.b') AS NUMERIC) = 2)"
    )


def test_metadata_filter_is_escaped_and_typed_on_the_wire(wire: Wire) -> None:
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    retriever = GoodMemRetriever(
        client=wire.sdk,
        space_ids=[SPACE_ID, SPACE_ID_2],
        metadata_filter={"owner": INJECTION, "archived": False, "year": 2026},
    )
    assert retriever.invoke("question")
    sent = json.loads(wire.requests[0].content)["spaceKeys"]
    expected = (
        "(CAST(val('$.archived') AS BOOLEAN) = false)"
        " AND (CAST(val('$.owner') AS TEXT) = 'x\\' OR \\'1\\'=\\'1')"
        " AND (CAST(val('$.year') AS NUMERIC) = 2026)"
    )
    assert sent == [
        {"spaceId": sid, "filter": expected} for sid in (SPACE_ID, SPACE_ID_2)
    ]


def test_filter_and_metadata_filter_combine_with_and(wire: Wire) -> None:
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    built = filters.compare("year", ">=", 2026)
    GoodMemRetriever(
        client=wire.sdk,
        space_ids=[SPACE_ID],
        filter=built,
        metadata_filter={"team": "blue"},
    ).invoke("question")
    sent = json.loads(wire.requests[0].content)["spaceKeys"][0]["filter"]
    assert sent == f"({built}) AND (CAST(val('$.team') AS TEXT) = 'blue')"


def test_empty_metadata_filter_sends_no_filter(wire: Wire) -> None:
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    GoodMemRetriever(client=wire.sdk, space_ids=[SPACE_ID], metadata_filter={}).invoke(
        "question"
    )
    body = json.loads(wire.requests[0].content)
    assert "spaceKeys" not in body or all("filter" not in k for k in body["spaceKeys"])


@pytest.mark.parametrize(
    "metadata_filter",
    [{"bad field": "v"}, {"f": float("nan")}, {"f": "line\nbreak"}],
)
def test_unsafe_metadata_filter_fails_at_configuration(
    wire: Wire, metadata_filter: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        GoodMemRetriever(
            client=wire.sdk,
            space_ids=[SPACE_ID],
            metadata_filter=metadata_filter,  # type: ignore[arg-type]
        )
    assert wire.requests == []
