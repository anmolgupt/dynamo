---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Embedding Shared-Memory Transport
---

Dynamo can move large OpenAI embedding request inputs and response vectors
through POSIX shared memory instead of serializing them on the normal request
plane. The feature currently applies to the vLLM embedding worker and is
disabled by default.

Frontend and worker processes must share the same IPC namespace and
`/dev/shm` mount. For Docker, use `--ipc=host` or another explicitly shared
IPC namespace.

## Transport formats

- Request SHM contains UTF-8 JSON for either the OpenAI `input` field or
  Rust-tokenized `token_ids`.
- Response SHM contains contiguous native-endian `float32` data with
  `[batch_size, embedding_dimension]` metadata.
- The request plane carries versioned metadata and all non-payload fields.
- The reader unlinks each segment after consuming it.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `DYN_EMBEDDING_SHM_REQUEST` | off | Enable frontend-to-worker request SHM. |
| `DYN_EMBEDDING_SHM_REQUEST_MIN_BYTES` | 65536 | Minimum serialized request size for SHM. |
| `DYN_EMBEDDING_SHM_REQUEST_MAX_BYTES` | 256 MiB | Maximum request payload accepted by the worker. |
| `DYN_EMBEDDING_SHM_RESPONSE` | off | Enable worker-to-frontend response SHM. |
| `DYN_EMBEDDING_SHM_MIN_BYTES` | 65536 | Minimum response-vector size for SHM. |
| `DYN_EMBEDDING_SHM_RESPONSE_MAX_BYTES` | 256 MiB | Maximum response SHM payload written or read. |
| `DYN_EMBEDDING_FRONTEND_TOKENIZATION` | off | Tokenize text in the Rust frontend and route token IDs to the worker. |
| `DYN_EMBEDDING_ADD_SPECIAL_TOKENS` | unset | Operator default for text tokenization. Explicit request values take precedence. |

When `DYN_EMBEDDING_ADD_SPECIAL_TOKENS` is unset, the Rust tokenizer uses
`true`, matching vLLM pooling behavior, while the vLLM-tokenized route leaves
the setting to vLLM. Client requests may send top-level
`add_special_tokens: true|false`; pre-tokenized inputs are never modified.

## Fallbacks and limits

Disabled SHM, small payloads, unsupported response encodings, and SHM writer
failures fall back to the normal request or JSON response plane. Reader
validation failures are request errors because the normal response payload is
not present after a worker publishes an SHM descriptor.

Names, exact segment sizes, response shapes, and payload limits are validated
before data is decoded. A response read failure mentioning `/dev/shm` usually
means the frontend and worker do not share an IPC namespace.

## Metrics

The frontend exports `dynamo_frontend_embedding_request_shm_*` and
`dynamo_frontend_embedding_shm_*` metric families. The vLLM worker exports
`dynamo_worker_embedding_shm_*`. Counters use `status="fallback"` whenever
the normal transport successfully takes over and reserve `status="error"` for
failures surfaced to the client.
