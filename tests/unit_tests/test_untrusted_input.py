"""IDs and file paths from a model or caller are refused before any request.

The SDK puts IDs into URL paths unescaped and httpx resolves dot segments, so
the memory ID "../spaces/<U>" used to send ``DELETE /v1/spaces/<U>``. A model
could also upload any file the process could read (``/etc/passwd``) and read
it back. These tests drive the real SDK and httpx transport against a
loopback HTTP server that records every request line it receives.
"""

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from goodmem import Goodmem
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemGetSpace,
    GoodMemListMemories,
    GoodMemRetrieveMemories,
    GoodMemRetriever,
    GoodMemUpdateSpace,
    add_documents,
    wait_for_memory,
)
from tests.unit_tests.conftest import CHUNK, MEMORY, SPACE

# The loopback server below is the only socket these tests open.
pytestmark = pytest.mark.enable_socket

U = "0f1e2d3c-4b5a-4968-8778-a9b8c7d6e5f4"
PAYLOADS = [
    f"../spaces/{U}",
    f"a/../../spaces/{U}",
    f"%2e%2e/spaces/{U}",
    f"..%2Fspaces%2F{U}",
    f"{U}/../../spaces/{U}",
    "",
    f" {U}",
    f"{U}?x=1",
    f"{U}#frag",
    f"{U}\n",
]
PAYLOAD_IDS = [
    "dotdot",
    "nested-dotdot",
    "encoded-dots",
    "encoded-slashes",
    "uuid-then-dotdot",
    "empty",
    "leading-space",
    "query",
    "fragment",
    "trailing-newline",
]


class RecordingServer:
    """A loopback GoodMem stand-in that records (method, raw path, body)."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes]] = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                recorder.requests.append((self.command, self.path, body))
                status, payload, content_type = recorder.reply(
                    self.command, urlsplit(self.path).path, body
                )
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_DELETE = _handle

            def log_message(self, *args: Any) -> None:
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        self.thread = threading.Thread(
            target=self.http.serve_forever, args=(0.01,), daemon=True
        )
        self.thread.start()
        self.sdk = Goodmem(base_url=self.url, api_key="synthetic-key")

    @staticmethod
    def reply(method: str, path: str, body: bytes) -> tuple[int, bytes, str]:
        """Answer like a lax server: IDs in request bodies are echoed back."""
        memory = dict(MEMORY)
        if body.startswith(b"{"):
            memory["memoryId"] = json.loads(body).get("memoryId", MEMORY["memoryId"])
        elif path.startswith("/v1/memories/"):
            memory["memoryId"] = path.removeprefix("/v1/memories/")
        if method == "DELETE":
            return 204, b"", "application/json"
        if path == "/v1/memories:retrieve":
            events = [CHUNK, {"memoryDefinition": MEMORY}]
            text = "\n".join(json.dumps(event) for event in events)
            return 200, text.encode(), "application/x-ndjson"
        if path == "/v1/memories:batchCreate":
            results = [
                {
                    "success": True,
                    "memory": MEMORY | {"memoryId": item.get("memoryId", U)},
                }
                for item in json.loads(body)["requests"]
            ]
            data: Any = {"results": results}
        elif path.startswith("/v1/spaces") and path.endswith("/memories"):
            data = {"memories": [MEMORY]}
        elif path.startswith("/v1/spaces"):
            data = SPACE
        elif path.startswith("/v1/memories"):
            data = memory
        else:
            data = {"message": "unexpected path"}
        return 200, json.dumps(data).encode(), "application/json"

    def lines(self) -> list[tuple[str, str]]:
        return [(method, path) for method, path, _ in self.requests]

    def close(self) -> None:
        self.sdk.close()
        self.http.shutdown()
        self.http.server_close()


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    recording = RecordingServer()
    try:
        yield recording
    finally:
        recording.close()


Args = Callable[[str], dict[str, Any]]

# Every model-facing tool argument that carries a GoodMem ID. The first six
# are URL path segments; create_memory's memory_id reaches a path when the
# tool waits for indexing; the rest travel in request bodies.
TOOL_IDS: list[tuple[type[BaseTool], Args, str]] = [
    (GoodMemGetMemory, lambda v: {"memory_id": v}, "memory_id"),
    (GoodMemDeleteMemory, lambda v: {"memory_id": v}, "memory_id"),
    (GoodMemGetSpace, lambda v: {"space_id": v}, "space_id"),
    (GoodMemDeleteSpace, lambda v: {"space_id": v}, "space_id"),
    (GoodMemUpdateSpace, lambda v: {"space_id": v, "name": "Renamed"}, "space_id"),
    (GoodMemListMemories, lambda v: {"space_id": v}, "space_id"),
    (
        GoodMemCreateMemory,
        lambda v: {"space_id": U, "memory_id": v, "original_content": "x"},
        "memory_id",
    ),
    (
        GoodMemCreateMemory,
        lambda v: {"space_id": v, "original_content": "x"},
        "space_id",
    ),
    (GoodMemCreateSpace, lambda v: {"name": "Docs", "embedder_id": v}, "embedder_id"),
    (
        GoodMemRetrieveMemories,
        lambda v: {"message": "q", "space_ids": [U, v]},
        "space_ids",
    ),
    (
        GoodMemRetrieveMemories,
        lambda v: {"message": "q", "space_ids": [U], "reranker_id": v},
        "reranker_id",
    ),
    (
        GoodMemRetrieveMemories,
        lambda v: {"message": "q", "space_ids": [U], "llm_id": v},
        "llm_id",
    ),
]


@pytest.mark.parametrize("payload", PAYLOADS, ids=PAYLOAD_IDS)
@pytest.mark.parametrize(
    "tool_class,args,field",
    TOOL_IDS,
    ids=[f"{cls.__name__}.{field}" for cls, _, field in TOOL_IDS],
)
def test_tools_refuse_non_uuid_ids_before_any_request(
    server: RecordingServer,
    tool_class: type[BaseTool],
    args: Args,
    field: str,
    payload: str,
) -> None:
    tool = tool_class(
        client=server.sdk, handle_tool_error=True, handle_validation_error=True
    )
    # As an agent's tool call: an error message, never a success.
    message = tool.invoke(
        {"type": "tool_call", "id": "call", "name": tool.name, "args": args(payload)}
    )
    assert server.lines() == [], f"server received {server.lines()}"
    assert isinstance(message, ToolMessage) and message.status == "error"
    # As a plain invoke: schema validation names the field.
    with pytest.raises(ValidationError, match=f"{field} must be a UUID"):
        tool_class(client=server.sdk).invoke(args(payload))
    # Bypassing the schema: the check next to the SDK call still refuses.
    with pytest.raises(ToolException, match=f"{field} must be a UUID"):
        tool._run(**args(payload))
    assert server.lines() == [], f"server received {server.lines()}"


def _reassigned_retriever(sdk: Goodmem, value: str) -> Any:
    retriever = GoodMemRetriever(client=sdk, space_ids=[U])
    retriever.space_ids = [value]  # Not revalidated by pydantic.
    return retriever.invoke("q")


# Developer-facing functions and configuration that carry a GoodMem ID.
DEVELOPER_IDS: list[Any] = [
    pytest.param(
        lambda sdk, v: wait_for_memory(sdk, v), "memory_id", id="wait_for_memory"
    ),
    pytest.param(
        lambda sdk, v: add_documents(sdk, v, [Document(page_content="x")]),
        "space_id",
        id="add_documents.space_id",
    ),
    pytest.param(
        lambda sdk, v: add_documents(sdk, U, [Document(id=v, page_content="x")]),
        "Document.id",
        id="add_documents.Document.id",
    ),
    pytest.param(
        lambda sdk, v: GoodMemRetriever(client=sdk, space_ids=[v]).invoke("q"),
        "space_ids",
        id="GoodMemRetriever.space_ids",
    ),
    pytest.param(
        lambda sdk, v: GoodMemRetriever(
            client=sdk, space_ids=[U], reranker_id=v
        ).invoke("q"),
        "reranker_id",
        id="GoodMemRetriever.reranker_id",
    ),
    pytest.param(
        _reassigned_retriever, "space_ids", id="GoodMemRetriever.space_ids-reassigned"
    ),
]


@pytest.mark.parametrize("payload", PAYLOADS, ids=PAYLOAD_IDS)
@pytest.mark.parametrize("call,field", DEVELOPER_IDS)
def test_developer_entry_points_refuse_non_uuid_ids_before_any_request(
    server: RecordingServer,
    call: Callable[[Goodmem, str], Any],
    field: str,
    payload: str,
) -> None:
    error: Exception | None = None
    try:
        call(server.sdk, payload)
    except Exception as exc:  # Checked below, after the request log.
        error = exc
    assert server.lines() == [], f"server received {server.lines()}"
    assert isinstance(error, ValueError), repr(error)
    assert f"{field} must be a UUID" in str(error)


@pytest.mark.parametrize(
    "tool_class,args,method,path",
    [
        (GoodMemGetMemory, {"memory_id": U.upper()}, "GET", f"/v1/memories/{U}"),
        (GoodMemDeleteMemory, {"memory_id": U.upper()}, "DELETE", f"/v1/memories/{U}"),
        (GoodMemGetSpace, {"space_id": U.upper()}, "GET", f"/v1/spaces/{U}"),
        (GoodMemDeleteSpace, {"space_id": U.upper()}, "DELETE", f"/v1/spaces/{U}"),
        (
            GoodMemUpdateSpace,
            {"space_id": U.upper(), "name": "Renamed"},
            "PUT",
            f"/v1/spaces/{U}",
        ),
        (
            GoodMemListMemories,
            {"space_id": U.upper(), "max_items": 1},
            "GET",
            f"/v1/spaces/{U}/memories",
        ),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_a_valid_uuid_reaches_exactly_the_intended_path(
    server: RecordingServer,
    tool_class: type[BaseTool],
    args: dict[str, Any],
    method: str,
    path: str,
) -> None:
    """Uppercase UUIDs are accepted and normalised to lowercase."""
    tool_class(client=server.sdk).invoke(args)
    assert [(m, urlsplit(p).path) for m, p in server.lines()] == [(method, path)]


def test_valid_ids_are_normalised_in_bodies_and_wait_paths(
    server: RecordingServer,
) -> None:
    created = GoodMemCreateMemory(client=server.sdk).invoke(
        {"space_id": U.upper(), "memory_id": U.upper(), "original_content": "x"}
    )
    assert created["memory_id"] == U
    (_, _, body), (method, path, _) = server.requests
    assert json.loads(body)["memoryId"] == U and json.loads(body)["spaceId"] == U
    assert (method, path) == ("GET", f"/v1/memories/{U}")
    assert wait_for_memory(server.sdk, U.upper()).processing_status == "COMPLETED"
    assert server.lines()[-1] == ("GET", f"/v1/memories/{U}")

    server.requests.clear()
    GoodMemRetrieveMemories(client=server.sdk).invoke(
        {"message": "q", "space_ids": [U.upper()], "reranker_id": U.upper()}
    )
    GoodMemRetriever(
        client=server.sdk, space_ids=[U.upper()], reranker_id=U.upper()
    ).invoke("q")
    for _, path, body in server.requests:
        assert path == "/v1/memories:retrieve"
        request = json.loads(body)
        assert request["spaceKeys"] == [{"spaceId": U}]
        assert request["postProcessor"]["config"]["reranker_id"] == U
    assert add_documents(
        server.sdk, U.upper(), [Document(id=U.upper(), page_content="x")], wait=False
    ) == [U]
    assert json.loads(server.requests[-1][2])["requests"][0]["spaceId"] == U


# --- File uploads -----------------------------------------------------------


def _model_parameters(tool: BaseTool) -> dict[str, Any]:
    return convert_to_openai_tool(tool)["function"]


def test_file_path_is_hidden_from_the_model_without_upload_dir(
    server: RecordingServer, tmp_path: Path
) -> None:
    tool = GoodMemCreateMemory(client=server.sdk, handle_tool_error=True)
    schema = tool.tool_call_schema
    assert isinstance(schema, type)
    assert "file_path" not in schema.model_json_schema()["properties"]
    function = _model_parameters(tool)
    assert "file_path" not in function["parameters"]["properties"]
    assert "file_path" not in function["description"]
    with pytest.raises(ValidationError, match="file_path"):
        tool.invoke({"space_id": U, "file_path": "/etc/passwd"})
    with pytest.raises(ToolException, match="File uploads are disabled"):
        tool._run(space_id=U, file_path="/etc/passwd")
    assert server.lines() == [], f"server received {server.lines()}"

    offered = GoodMemCreateMemory(client=server.sdk, upload_dir=tmp_path)
    function = _model_parameters(offered)
    assert "file_path" in function["parameters"]["properties"]
    assert "upload directory" in function["description"]
    assert "upload_dir" not in function["parameters"]["properties"]


@pytest.fixture
def uploads(tmp_path: Path) -> Path:
    """An upload_dir with one allowed file, next to a secret outside it."""
    upload_dir = tmp_path / "uploads"
    (upload_dir / "nested").mkdir(parents=True)
    (upload_dir / "notes.txt").write_text("allowed notes")
    (upload_dir / "nested" / "deep.txt").write_text("allowed deep notes")
    (tmp_path / "secret.txt").write_text("outside secret")
    (upload_dir / "escape").symlink_to(tmp_path / "secret.txt")
    (upload_dir / "escape-dir").symlink_to(tmp_path, target_is_directory=True)
    (upload_dir / "alias").symlink_to(upload_dir / "notes.txt")
    return upload_dir


@pytest.mark.parametrize(
    "file_path,reason",
    [
        ("/etc/passwd", "resolves outside"),
        ("../../../../../../../../etc/passwd", "resolves outside"),
        ("../secret.txt", "resolves outside"),
        ("nested/../../secret.txt", "resolves outside"),
        ("escape", "resolves outside"),  # symlink to a file outside
        ("escape-dir/secret.txt", "resolves outside"),  # symlinked directory
        ("{outside}", "resolves outside"),  # absolute path outside
        ("nested", "not a regular file"),
        ("missing.txt", "Cannot read"),
    ],
)
def test_upload_outside_upload_dir_is_refused_before_any_request(
    server: RecordingServer, uploads: Path, file_path: str, reason: str
) -> None:
    file_path = file_path.format(outside=uploads.parent / "secret.txt")
    tool = GoodMemCreateMemory(
        client=server.sdk, upload_dir=str(uploads), handle_tool_error=True
    )
    message = tool.invoke(
        {
            "type": "tool_call",
            "id": "call",
            "name": tool.name,
            "args": {"space_id": U, "file_path": file_path, "wait": False},
        }
    )
    assert server.lines() == [], f"server received {server.lines()}"
    assert isinstance(message, ToolMessage) and message.status == "error"
    assert reason in str(message.content)


@pytest.mark.parametrize(
    "file_path,content",
    [
        ("notes.txt", b"allowed notes"),
        ("nested/deep.txt", b"allowed deep notes"),
        ("nested/../notes.txt", b"allowed notes"),
        ("alias", b"allowed notes"),
        ("{inside}", b"allowed notes"),
    ],
)
def test_upload_inside_upload_dir_is_sent(
    server: RecordingServer, uploads: Path, file_path: str, content: bytes
) -> None:
    file_path = file_path.format(inside=uploads / "notes.txt")
    GoodMemCreateMemory(client=server.sdk, upload_dir=uploads).invoke(
        {"space_id": U, "file_path": file_path, "wait": False}
    )
    ((method, path, body),) = server.requests
    assert (method, path) == ("POST", "/v1/memories")
    assert content in body and b"outside secret" not in body
