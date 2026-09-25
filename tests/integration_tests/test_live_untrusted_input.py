"""Live: model-supplied IDs and file paths are refused before reaching GoodMem.

Creates and deletes only its own test space. Requires GOODMEM_BASE_URL,
GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID; skipped otherwise.
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from goodmem import Goodmem
from langchain_core.tools import ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetSpace,
    GoodMemListMemories,
)


@pytest.fixture(scope="module")
def space() -> Iterator[tuple[Goodmem, str]]:
    required = ["GOODMEM_BASE_URL", "GOODMEM_API_KEY", "GOODMEM_EMBEDDER_ID"]
    if any(not os.getenv(key) for key in required):
        pytest.skip("Set endpoint, credentials and embedder ID for the live test")
    with Goodmem(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
        timeout=120,
    ) as sdk:
        sid = GoodMemCreateSpace(client=sdk).invoke(
            {
                "name": f"langchain-untrusted-input-{uuid.uuid4()}",
                "embedder_id": os.environ["GOODMEM_EMBEDDER_ID"],
            }
        )["space_id"]
        try:
            yield sdk, sid
        finally:
            GoodMemDeleteSpace(client=sdk).invoke({"space_id": sid})


def _memories(sdk: Goodmem, sid: str) -> list[dict[str, object]]:
    return GoodMemListMemories(client=sdk).invoke({"space_id": sid})


@pytest.mark.parametrize(
    "file_path",
    [
        "/etc/passwd",
        "../../../../../../../../etc/passwd",
        "../../../../../../../../etc/hostname",
    ],
)
def test_live_upload_outside_upload_dir_is_refused(
    space: tuple[Goodmem, str], tmp_path: Path, file_path: str
) -> None:
    sdk, sid = space
    tool = GoodMemCreateMemory(client=sdk, upload_dir=tmp_path)
    with pytest.raises(ToolException, match="outside|Cannot read"):
        tool.invoke({"space_id": sid, "file_path": file_path, "wait": False})
    assert _memories(sdk, sid) == []


def test_live_file_path_is_not_offered_without_upload_dir(
    space: tuple[Goodmem, str],
) -> None:
    sdk, sid = space
    tool = GoodMemCreateMemory(client=sdk)
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]
    assert "file_path" not in parameters["properties"]
    with pytest.raises(ValidationError, match="file_path"):
        tool.invoke({"space_id": sid, "file_path": "/etc/passwd"})
    with pytest.raises(ToolException, match="File uploads are disabled"):
        tool._run(space_id=sid, file_path="/etc/passwd")
    assert _memories(sdk, sid) == []


@pytest.mark.parametrize(
    "template",
    [
        "../spaces/{sid}",
        "a/../../spaces/{sid}",
        "%2e%2e/spaces/{sid}",
        "..%2Fspaces%2F{sid}",
        "{sid}/../../spaces/{sid}",
    ],
)
def test_live_memory_id_cannot_reach_the_space(
    space: tuple[Goodmem, str], template: str
) -> None:
    sdk, sid = space
    payload = template.format(sid=sid)
    with pytest.raises(ValidationError, match="memory_id must be a UUID"):
        GoodMemDeleteMemory(client=sdk).invoke({"memory_id": payload})
    with pytest.raises(ToolException, match="memory_id must be a UUID"):
        GoodMemDeleteMemory(client=sdk)._run(memory_id=payload)
    assert GoodMemGetSpace(client=sdk).invoke({"space_id": sid})["space_id"] == sid
