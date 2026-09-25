# Changelog

## 0.2.3 — 2026-09-25

Security fixes. Both close ways for a model's tool arguments to reach data it was not given.

- **IDs must be UUIDs, and anything else is refused before a request is made.** The GoodMem SDK puts IDs into URL paths unescaped and httpx resolves dot segments, so `GoodMemDeleteMemory(memory_id="../spaces/<id>")` sent `DELETE /v1/spaces/<id>`, the request that deletes the whole space, and returned success. Get/delete memory, get/delete/update space and list memories (whose space ID is a path segment) were all affected, as were `wait_for_memory` and the indexing wait of `GoodMemCreateMemory` (a lax server echoing a `memory_id` of `../spaces/<id>` made it `GET /v1/spaces/<id>`). Every ID the package accepts — tool arguments, `GoodMemRetriever.space_ids`/`reranker_id`, `add_documents`' `space_id` and Document IDs, `wait_for_memory`'s `memory_id` — must now match `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (hex, any case; normalised to lowercase). Tool schemas declare `format: uuid` and the pattern, so the model is told; a non-UUID fails schema validation, and a direct `_run` call raises `ToolException` naming the field. Developer functions and the retriever raise `ValueError`/`ValidationError`. Percent-encoded forms (`%2e%2e`, `%2F`), query strings, fragments, whitespace and empty strings are refused too. The check returns a plain `str` of exactly the characters it checked, so a `str` subclass cannot pass it and then put a different string into the URL. **Migration:** anything that passed placeholder IDs such as `"space-1"` must pass real UUIDs; an empty `reranker_id` is no longer treated as "no reranker" — pass `None`; a `uuid.UUID` object, which 0.2.2 accepted in `wait_for_memory` and direct tool `_run` calls, is now refused ("must be a UUID string") — pass `str(value)`.
- **`GoodMemCreateMemory` no longer reads arbitrary host files.** `file_path` was model-visible and unrestricted: a model could upload `/etc/passwd` (sent byte-for-byte; a relative `../../../../../../../../etc/hostname` worked too) and read it back with `GoodMemGetMemory(include_content=True)`. The tool now takes an operator-set `upload_dir`. Without it (the default) the model's schema has no `file_path` and a direct call with one is refused. With it, `file_path` is resolved (symlinks included) and refused unless it is a regular file inside `upload_dir`, before anything is opened or sent. The confinement is ported from crewai-goodmem. Developers can still upload any file through the SDK, `client.memories.create(file_path=...)`, or through a tool constructed with `upload_dir`.
- **Filters can no longer be widened by the values put into them (P34).** `GoodMemRetriever.filter` is a raw expression, and the only way to filter by a user's value was to format it in, so `x' OR '1'='1` matched every memory live. New `langchain_goodmem.filters` builds expressions safely: `equals`, `not_equals`, `compare`, `one_of`, `all_of`, `any_of`, `from_mapping`. Literals are escaped as the server expects (`\'`, not `''`), control characters and unsafe field names are refused, and each value is cast to its own type (`TEXT`, `NUMERIC`, `BOOLEAN`; a boolean compared as text is accepted by the server and matches nothing). Non-finite numbers are refused and others are written as plain decimals. `GoodMemRetriever` gains `metadata_filter={"field": value}`, validated when the retriever is built and combined with `filter` by AND. `filter` is unchanged and still sent verbatim.
- **The README's example runs again.** The UUID check above made its `space_id = "your-space-uuid"` placeholder raise `ValueError` before any request, and the reranking sentence showed `reranker_id="your-reranker-uuid"`, which fails validation. The placeholder is now a UUID, and the upload sentence shows the constructor, `GoodMemCreateMemory(upload_dir="/srv/uploads")`. A new test runs every README snippet through the SDK against a loopback server and checks the symbols, IDs, versions and status codes it names.

## 0.2.2 — 2026-09-25

`GoodMemRetriever` follows the retrieval status contract decided on 2026-09-23, which every GoodMem integration now shares:

- **Statuses never raise.** A retrieval that reported a problem *and* returned Documents keeps those Documents, each carrying `metadata["goodmem_partial"] = True` and `metadata["goodmem_statuses"]`. Previously the whole result was discarded with `GoodMemRetrievalError`. A problem that left no Documents returns `[]` and emits a `UserWarning` and a warning log line with the statuses; previously it raised. Clean retrievals are unchanged: the two keys are present only when something went wrong.
- **Unknown status codes are surfaced, not dropped.** A code introduced by a newer server (decoded as `None` by the SDK) is reported as `UNKNOWN` with `unrecognized: True` and flags the result partial. 0.2.1 silently ignored it, so a server upgrade could change behaviour without any signal.
- **`FEATURE_DISABLED` is informational by its code alone.** The server defines it as "feature disabled due to missing configuration"; its details are no longer inspected.
- `GoodMemRetrievalError` is still raised for a malformed stream (a chunk whose memory definition never arrived) and remains exported. The internal `checked_events` helper is replaced by `classify_statuses`.

## 0.2.1 — 2026-09-14

`GoodMemRetriever` now tolerates status codes introduced by newer GoodMem servers. The SDK decodes these codes as `None`; they no longer cause the retriever to discard Documents or raise an error. Known retrieval failures still raise `GoodMemRetrievalError`. Regression tests exercise both `invoke` and `ainvoke` through the SDK's HTTP transport.

## 0.2.0 — 2026-09-14

0.2 is an intentional API break. The integration uses the official `goodmem` SDK and adds `GoodMemRetriever`, a LangChain `BaseRetriever` returning Documents with joined source metadata and identifiers.

### Migration from 0.1

- Replace `GoodMemClient` with `goodmem.Goodmem`. Pass the SDK instance through `client=`, or configure tools/retrievers with `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`.
- Tool calls return SDK-shaped Python dictionaries/lists, using snake_case fields, rather than JSON strings with `success`, counts, or custom content fields. Delete tools return `None`. LangChain serializes results for agents. SDK errors become `ToolException`, compatible with LangChain's `handle_tool_error` option.
- `GoodMemCreateSpace` creates a new space. It does not find or reuse spaces by name. Select existing spaces with `GoodMemListSpaces`; the tool uses SDK chunking defaults. Configure custom chunking through the SDK before giving an agent access to the space. Chunking controls are absent from the tool.
- `GoodMemCreateMemory` takes `original_content` or `file_path`, exactly one. It waits for the created memory to finish indexing by default. Set `wait=False` for background ingestion. A wait failure includes the created memory ID in the exception.
- `GoodMemRetrieveMemories` takes `message`, a list of `space_ids`, and SDK post-processing parameters (`requested_size`, `max_results`, `llm_temp`, etc.). It returns SDK events, preserving chunks and server statuses together. It does not classify them into `success` or `partial`. Use `GoodMemRetriever` for Documents with retrieval failures raised as exceptions.
- `GoodMemGetMemory` fetches metadata by default. Set `include_content=True` to receive the SDK's base64 `original_content` field in the same request. The separate content fetch, decoding fallbacks, and `contentError` wrapper are removed.
- Space/memory list tools use SDK pagination and `max_items` (default 100; `None` for all).
- `wait_for_indexing` and `public_read` are absent. Unknown tool arguments fail normal schema validation; there are no deprecated argument handlers. Search never polls empty results.
- API keys are stored as `SecretStr` and excluded from serialization.

### LangChain usage

`add_documents(client, space_id, documents)` writes LangChain Documents through the SDK batch API, preserving metadata and optional UUID IDs. It waits for indexing by default and returns memory IDs. `GoodMemIngestionError.created_memory_ids` retains successful writes if a batch or subsequent wait fails. This is a document ingestion path, not a VectorStore implementation.

`GoodMemRetriever` accepts a native GoodMem `filter` expression, applied on the server to every configured space before retrieval. It supports callbacks, LCEL, batching, per-call `k`, and executor-backed async. Reranking needs no LLM. Use LangChain's `create_retriever_tool` to expose it to an agent; there is no additional GoodMem factory. `wait_for_memory` checks a specific memory's processing status when coordinating ingestion through the SDK.

The pinned `langchain-tests` suites cover all eleven tools and the retriever, including live retriever tests. Native `AsyncGoodmem` support remains follow-up work. Internal review notes and validation reports are excluded from distributions.
