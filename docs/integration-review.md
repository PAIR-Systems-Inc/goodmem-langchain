# Integration review: clean 0.2 API

These are local, uncommitted changes on `feat/sdk-retriever`, based on `07f087e9570f2a6cf52205b3d01cae0382d17a35`. The version remains `0.2.0.dev0`; nothing has been published. This document supersedes the earlier compatibility proposal and is excluded from distributions.

The useful additions are the standard Document retriever and waiting for a known memory to finish indexing. Both apply to ordinary LangChain applications. The official SDK now owns transport, response parsing, pagination and resource semantics.

## What was removed

- The 510-line `GoodMemClient` compatibility facade and string-based tool dispatch.
- JSON success/count envelopes, retrieval-status classification into success/partial, and custom content decoding/error wrappers.
- Deprecated argument handling, including the `wait_for_indexing=False` exception and the special `public_read` rejection.
- Name-based space reuse, configuration comparison and independent chunking defaults.
- The extra retriever-tool factory. Applications use LangChain's `create_retriever_tool` and choose their own formatting.
- Tests for the deleted compatibility policies and duplicate schema/import snapshots.

Each tool has an explicit typed `_run` that calls the SDK. Results use SDK fields; server retrieval diagnostics remain in the event list. The Document retriever continues to raise on incomplete retrieval. Create-memory waits by default and identifies the created memory if waiting fails. Empty searches return immediately.

[The changelog](../CHANGELOG.md) documents the intentional API break. [The README](../README.md) is a quickstart again.

## Size

| Measure | Previous reviewed worktree | Clean 0.2 |
|---|---:|---:|
| Production Python lines | 1,787 | 824 |
| Production AST statements | 467 | 313 |
| Test Python lines | 1,808 | 911 |
| README words, including code | 1,655 | 349 |

Production code falls by 963 lines (53.9%). Against the original release's 1,994 production lines, the reduction is 58.7%. Counts include new modules, docstrings and blank lines; statement counts exclude docstrings. [Measurements](validation/code-size.json).

## Validation

- 37 unit tests pass on Python 3.10 and 3.13, using the real SDK over an in-memory HTTP transport.
- One owned-space live workflow covers all eleven tools, plain/reranked retrieval without an LLM, indexing readiness, native inline content, failure diagnostics, and cleanup. The standalone smoke example also passes.
- The demo's 12 tests pass; 16 live comparisons preserve the original retrieval text, metadata, scores and order.
- Ruff, formatting, mypy and distribution checks are recorded in [verification.json](validation/verification.json). Packaging checks exclude review documents and removed modules, and enforce the README limit.

Native `AsyncGoodmem` support and the `langchain-tests` standard suites remain follow-up work. The previous undeclared `typing_extensions` import has been removed. The SDK still supplies its own status model and defaults; the integration no longer duplicates those policies in a second client.

No commit, push, PR or publication is authorized before user review.
