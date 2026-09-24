# Changelog

## Unreleased

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
