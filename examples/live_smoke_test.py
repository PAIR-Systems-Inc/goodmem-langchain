"""Live end-to-end smoke test for every langchain-goodmem tool.

Runs against a real GoodMem server. Exercises all 11 tools and every
post-processor knob on `goodmem_retrieve_memories` (reranker, LLM,
relevance threshold, LLM temperature, chronological resort, max results).

Required environment variables:
    GOODMEM_BASE_URL      e.g. https://localhost:8080
    GOODMEM_API_KEY       your GoodMem API key

Optional environment variables (sensible defaults shown):
    GOODMEM_VERIFY_SSL    "true" | "false"  (default: "false")
    GOODMEM_EMBEDDER_ID   default: 019cfd1c-c033-7517-b7de-f73941a0464b
    GOODMEM_RERANKER_ID   default: 019cfda4-7e2f-743c-9edb-e469a97b95c6
    GOODMEM_LLM_ID        default: 019cfd9f-0963-76f9-b069-4cde19a64ba8

Run from the repo root after `pip install -e .`:
    python examples/live_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any

from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemGetSpace,
    GoodMemListEmbedders,
    GoodMemListMemories,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
    GoodMemUpdateSpace,
)

# --- Configuration -------------------------------------------------------

BASE_URL = os.environ.get("GOODMEM_BASE_URL")
API_KEY = os.environ.get("GOODMEM_API_KEY")
VERIFY_SSL = os.environ.get("GOODMEM_VERIFY_SSL", "false").lower() == "true"

EMBEDDER_ID = os.environ.get(
    "GOODMEM_EMBEDDER_ID", "019cfd1c-c033-7517-b7de-f73941a0464b"
)
RERANKER_ID = os.environ.get(
    "GOODMEM_RERANKER_ID", "019cfda4-7e2f-743c-9edb-e469a97b95c6"
)
LLM_ID = os.environ.get("GOODMEM_LLM_ID", "019cfd9f-0963-76f9-b069-4cde19a64ba8")

if not BASE_URL or not API_KEY:
    sys.exit(
        "ERROR: set GOODMEM_BASE_URL and GOODMEM_API_KEY before running this script."
    )

KW: dict[str, Any] = {
    "goodmem_base_url": BASE_URL,
    "goodmem_api_key": API_KEY,
    "goodmem_verify_ssl": VERIFY_SSL,
}

# --- Helpers -------------------------------------------------------------

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
INFO = "\033[36m›\033[0m"
results: list[tuple[str, bool, str]] = []


def step(label: str) -> None:
    print(f"\n{INFO} {label}")
    print("─" * (len(label) + 2))


def call(tool_name: str, tool: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Invoke a tool, parse the JSON, mark pass/fail, and return the result."""
    raw = tool.invoke(payload)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  {FAIL} {tool_name}: non-JSON response")
        print(f"      {raw[:400]}")
        results.append((tool_name, False, "non-JSON response"))
        return {}
    ok = bool(result.get("success"))
    icon = PASS if ok else FAIL
    summary = result.get("error") or result.get("message") or ""
    print(f"  {icon} {tool_name}{'  — ' + summary if summary else ''}")
    results.append((tool_name, ok, summary))
    return result


def preview(label: str, value: Any, limit: int = 200) -> None:
    text = (
        json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
    )
    if len(text) > limit:
        text = text[:limit] + " …"
    print(f"      {label}: {text}")


# --- Run ----------------------------------------------------------------

print(f"GoodMem live smoke test  →  {BASE_URL}")
print(f"  embedder_id  = {EMBEDDER_ID}")
print(f"  reranker_id  = {RERANKER_ID}")
print(f"  llm_id       = {LLM_ID}")

# 1. list_embedders ------------------------------------------------------
step("1. goodmem_list_embedders")
r = call("list_embedders", GoodMemListEmbedders(**KW), {})
embedders = r.get("embedders", [])
preview("count", len(embedders))
ids = {e.get("embedderId") for e in embedders}
if EMBEDDER_ID not in ids:
    print(
        f"  {FAIL} configured EMBEDDER_ID {EMBEDDER_ID} not found in server's "
        f"embedder list — subsequent steps may fail."
    )

# 2. list_spaces (snapshot) ---------------------------------------------
step("2. goodmem_list_spaces (before)")
r = call("list_spaces", GoodMemListSpaces(**KW), {})
preview("count", len(r.get("spaces", [])))

# 3. create_space --------------------------------------------------------
space_name = f"langchain-smoke-{int(time.time())}-{uuid.uuid4().hex[:6]}"
step(f"3. goodmem_create_space  (name={space_name})")
r = call(
    "create_space",
    GoodMemCreateSpace(**KW),
    {"name": space_name, "embedder_id": EMBEDDER_ID},
)
space_id = r.get("spaceId")
if not space_id:
    sys.exit(f"\n{FAIL} create_space returned no spaceId, aborting.")
preview("spaceId", space_id)

# 4. get_space -----------------------------------------------------------
step("4. goodmem_get_space")
r = call("get_space", GoodMemGetSpace(**KW), {"space_id": space_id})
preview("space.name", r.get("space", {}).get("name"))

# 5. update_space (rename) ----------------------------------------------
new_name = space_name + "-renamed"
step(f"5. goodmem_update_space  (rename → {new_name})")
r = call(
    "update_space",
    GoodMemUpdateSpace(**KW),
    {"space_id": space_id, "name": new_name},
)
preview("space.name", r.get("space", {}).get("name"))

# 6. create_memory (×3, varied content) ---------------------------------
step("6. goodmem_create_memory  (×3 varied texts)")
texts = [
    "LangChain is a framework for developing applications powered by language models.",
    "GoodMem provides server-side embedding, vector search, and reranking.",
    "Pasta carbonara is a Roman dish made with eggs, guanciale, and pecorino.",
]
memory_ids: list[str] = []
for i, text in enumerate(texts, 1):
    r = call(
        f"create_memory[{i}]",
        GoodMemCreateMemory(**KW),
        {"space_id": space_id, "text_content": text},
    )
    if r.get("memoryId"):
        memory_ids.append(r["memoryId"])
preview("created memory_ids", memory_ids)

if not memory_ids:
    sys.exit(f"\n{FAIL} no memories created, aborting.")

# 7. list_memories -------------------------------------------------------
step("7. goodmem_list_memories")
r = call("list_memories", GoodMemListMemories(**KW), {"space_id": space_id})
preview("count", len(r.get("memories", [])))

# 8. get_memory ----------------------------------------------------------
step(f"8. goodmem_get_memory  (memory_id={memory_ids[0]})")
r = call("get_memory", GoodMemGetMemory(**KW), {"memory_id": memory_ids[0]})
preview("memory.spaceId", r.get("memory", {}).get("spaceId"))

# 9. retrieve_memories — variants ---------------------------------------
print("\n--- goodmem_retrieve_memories variants ---")
print(
    "  (waits up to 60s for indexing on first call; the queried memories "
    "were just created)"
)

retrieve = GoodMemRetrieveMemories(**KW)
query = "Which framework develops applications powered by language models?"

# 9a. plain
step("9a. retrieve_memories  (plain, no postprocessor)")
r = call(
    "retrieve_plain",
    retrieve,
    {
        "query": query,
        "space_ids": space_id,
        "max_results": 5,
        "wait_for_indexing": True,
    },
)
preview("totalResults", r.get("totalResults"))
if r.get("results"):
    preview("top relevanceScore", r["results"][0].get("relevanceScore"))

# 9b. reranker only
step("9b. retrieve_memories  (+ reranker_id)")
r = call(
    "retrieve_reranker",
    retrieve,
    {
        "query": query,
        "space_ids": space_id,
        "max_results": 5,
        "wait_for_indexing": False,
        "reranker_id": RERANKER_ID,
    },
)
preview("totalResults", r.get("totalResults"))

# 9c. LLM only (expect abstractReply)
step("9c. retrieve_memories  (+ llm_id  → abstractReply)")
r = call(
    "retrieve_llm",
    retrieve,
    {
        "query": query,
        "space_ids": space_id,
        "max_results": 5,
        "wait_for_indexing": False,
        "llm_id": LLM_ID,
        "llm_temperature": 0.2,
    },
)
preview("totalResults", r.get("totalResults"))
preview("abstractReply", r.get("abstractReply"), limit=400)

# 9d. reranker + LLM + relevance threshold
step(
    "9d. retrieve_memories  (+ reranker_id + llm_id + relevance_threshold=0.1 "
    "+ llm_temperature=0.5)"
)
r = call(
    "retrieve_full",
    retrieve,
    {
        "query": query,
        "space_ids": space_id,
        "max_results": 5,
        "wait_for_indexing": False,
        "reranker_id": RERANKER_ID,
        "llm_id": LLM_ID,
        "relevance_threshold": 0.1,
        "llm_temperature": 0.5,
    },
)
preview("totalResults", r.get("totalResults"))
preview("abstractReply", r.get("abstractReply"), limit=400)

# 9e. chronological resort
step("9e. retrieve_memories  (+ chronological_resort=True)")
r = call(
    "retrieve_chrono",
    retrieve,
    {
        "query": query,
        "space_ids": space_id,
        "max_results": 5,
        "wait_for_indexing": False,
        "reranker_id": RERANKER_ID,
        "chronological_resort": True,
    },
)
preview("totalResults", r.get("totalResults"))

# 9f. max_results=1
step("9f. retrieve_memories  (max_results=1)")
r = call(
    "retrieve_max1",
    retrieve,
    {
        "query": query,
        "space_ids": space_id,
        "max_results": 1,
        "wait_for_indexing": False,
    },
)
preview("totalResults", r.get("totalResults"))

# 10. delete_memory (one) ------------------------------------------------
step(f"10. goodmem_delete_memory  (memory_id={memory_ids[-1]})")
call("delete_memory", GoodMemDeleteMemory(**KW), {"memory_id": memory_ids[-1]})

# 11. delete_space (cleanup) --------------------------------------------
step(f"11. goodmem_delete_space  (space_id={space_id})")
call("delete_space", GoodMemDeleteSpace(**KW), {"space_id": space_id})

# --- Summary ------------------------------------------------------------

print("\n" + "═" * 60)
print(" SUMMARY")
print("═" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
for name, ok, msg in results:
    icon = PASS if ok else FAIL
    line = f"  {icon} {name}"
    if not ok and msg:
        line += f"  — {msg}"
    print(line)
print("─" * 60)
print(f"  {passed}/{total} passed")
sys.exit(0 if passed == total else 1)
