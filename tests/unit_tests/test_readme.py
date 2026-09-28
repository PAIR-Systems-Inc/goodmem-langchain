"""The README's code runs as written, and what it names exists.

Only the server address and API key are substituted: the Python blocks run
in order, in one namespace (the tool block uses the retriever from the block
before it), through the real SDK and httpx against the loopback recording
server from ``test_untrusted_input``.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool

import langchain_goodmem
from langchain_goodmem import GoodMemCreateMemory, GoodMemRetriever
from langchain_goodmem._ids import require_uuid
from langchain_goodmem.retrievers import _INFORMATIONAL_CODES
from tests.unit_tests.conftest import RERANKER_ID
from tests.unit_tests.test_untrusted_input import RecordingServer

pytestmark = pytest.mark.enable_socket

PROJECT = Path(__file__).resolve().parents[2]
README = (PROJECT / "README.md").read_text(encoding="utf-8")
PYTHON_BLOCKS = re.findall(r"```python\n(.*?)```", README, re.S)
INLINE_CODE = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", README)


@pytest.fixture
def readme_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[RecordingServer]:
    recording = RecordingServer()
    monkeypatch.setenv("GOODMEM_BASE_URL", recording.url)
    monkeypatch.setenv("GOODMEM_API_KEY", "synthetic-key")
    try:
        yield recording
    finally:
        recording.close()


def test_python_blocks_run_as_written(readme_server: RecordingServer) -> None:
    assert len(PYTHON_BLOCKS) == 2
    namespace: dict[str, Any] = {}
    for number, block in enumerate(PYTHON_BLOCKS):
        exec(compile(block, f"README.md python block {number}", "exec"), namespace)
    space_id = namespace["space_id"]
    memory_id = namespace["memory_ids"][0]
    assert readme_server.lines() == [
        ("POST", "/v1/memories:batchCreate"),
        ("GET", f"/v1/memories/{memory_id}"),
        ("POST", "/v1/memories:retrieve"),
    ]
    search = json.loads(readme_server.requests[2][2])
    assert [key["spaceId"] for key in search["spaceKeys"]] == [space_id]
    assert [key["filter"] for key in search["spaceKeys"]] == [
        "CAST(val('$.team') AS TEXT) = 'blue'"
    ]
    assert isinstance(namespace["tool"], BaseTool)
    assert namespace["tool"].response_format == "content_and_artifact"

    # The reranking sentence: the same retriever with reranker_id and fetch_k=20.
    readme_server.requests.clear()
    reranked = namespace["retriever"].model_copy(
        update={"reranker_id": RERANKER_ID, "fetch_k": 20}
    )
    assert reranked.invoke("Who owns Cobalt's launch?")
    search = json.loads(readme_server.requests[0][2])
    assert search["requestedSize"] == 20
    assert search["postProcessor"]["config"]["reranker_id"] == RERANKER_ID


def test_inline_constructor_calls_run() -> None:
    calls = [code for code in INLINE_CODE if re.fullmatch(r"GoodMem\w+\(.*\)", code)]
    assert calls, "README shows how to construct the upload tool"
    for call in calls:
        tool = eval(call, vars(langchain_goodmem))
        if isinstance(tool, GoodMemCreateMemory):
            assert tool.upload_dir is not None
            assert "file_path" in tool.args


def test_inline_id_literals_are_uuids() -> None:
    literals = re.findall(r'(\w+_ids?)\s*=\s*\[?"([^"]*)"', README)
    assert literals
    for field, value in literals:
        require_uuid(value, field)


def test_named_symbols_exist() -> None:
    for name in set(re.findall(r"\bGoodMem\w+", README)):
        assert name in langchain_goodmem.__all__, name
    for name in ("add_documents", "wait_for_memory"):
        assert name in README and name in langchain_goodmem.__all__
    assert "created_memory_ids" in vars(
        langchain_goodmem.GoodMemIngestionError("", created_memory_ids=[])
    )
    for argument in (
        "filter",
        "metadata_filter",
        "reranker_id",
        "fetch_k",
        "k",
        "client",
    ):
        assert f"`{argument}" in README
        assert argument in GoodMemRetriever.model_fields, argument
    assert "upload_dir" in GoodMemCreateMemory.model_fields
    assert (PROJECT / "examples" / "live_smoke_test.py").is_file()


def test_versions_and_status_codes_match_the_package() -> None:
    # tomllib is 3.11+; CI also runs 3.10.
    manifest = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "0\.2\.\d+"$', manifest, re.M)
    assert "Version **0.2**" in README
    assert re.search(r'^requires-python = ">=3\.10', manifest, re.M)
    assert "Python 3.10+" in README
    named = re.search(r"Notices that carry no loss \(([^)]*)\)", README)
    assert named is not None
    assert set(re.findall(r"`(\w+)`", named.group(1))) == _INFORMATIONAL_CODES
