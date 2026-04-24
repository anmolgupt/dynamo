# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import statistics
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from vllm.inputs import TokensPrompt
from vllm.outputs import PoolingOutput
from vllm.pooling_params import PoolingParams

from dynamo._core import Context

logger = logging.getLogger(__name__)

# Per-request phase timing. Enabled by setting DYN_EMBEDDING_TIMING=1 in the
# environment. Emits one structured log line per generate() call so you can
# correlate batch size, token counts, and time spent in each phase with the
# client-side wall-clock latency from compare_embeddings.py.
_TIMING_ENABLED = os.environ.get("DYN_EMBEDDING_TIMING", "0") not in ("0", "", "false", "False")

# Dummy-handler mode. When DYN_EMBEDDING_DUMMY=1 the handler skips vLLM
# entirely and returns deterministic zero-filled embeddings of the same
# shape the real handler would produce. Useful for isolating Rust pipeline
# + HTTP + ser/de overhead from the engine/GPU work.
_DUMMY_ENABLED = os.environ.get("DYN_EMBEDDING_DUMMY", "0") not in ("0", "", "false", "False")
_DUMMY_DIM = int(os.environ.get("DYN_EMBEDDING_DUMMY_DIM", "2048"))


class EmbeddingWorkerHandler:
    """Request handler for vLLM embedding (pooling) models.

    Receives PreprocessedEmbeddingRequest dicts from the Rust frontend
    (which has already tokenized the input), calls vLLM encode() for each
    sequence, and returns EmbeddingsEngineOutput dicts.
    """

    def __init__(
        self,
        engine_client,
        config,
        shutdown_event: Optional[asyncio.Event] = None,
        vllm_config=None,
    ):
        self.engine_client = engine_client
        self.config = config
        self.shutdown_event = shutdown_event

        # The Rust tokenizer encodes with add_special_tokens=False, but vllm's
        # tokenizer adds BOS by default. Prepend BOS so token sequences match.
        self.bos_token_id: Optional[int] = None
        if vllm_config is not None:
            hf_cfg = vllm_config.model_config.hf_config
            bos = getattr(hf_cfg, "bos_token_id", None)
            if bos is not None:
                self.bos_token_id = int(bos)
                logger.info(
                    f"EmbeddingWorkerHandler: will prepend BOS token {self.bos_token_id}"
                )
        if _DUMMY_ENABLED:
            logger.warning(
                "EmbeddingWorkerHandler: DUMMY mode enabled "
                "(DYN_EMBEDDING_DUMMY=1, dim=%d). The GPU / vLLM engine will "
                "be bypassed; responses are zero-filled and meaningless. "
                "This is intended for pipeline-overhead measurement only.",
                _DUMMY_DIM,
            )
        logger.info("EmbeddingWorkerHandler initialized")

    async def _encode_single(
        self, token_ids: list[int], request_id: str,
        per_seq_times_ms: Optional[list[float]] = None,
    ) -> PoolingOutput:
        """Encode a single token sequence and return the EmbeddingOutput.

        If `per_seq_times_ms` is provided, records the wall-clock time (ms) from
        the first `async for` iteration entering this coroutine until the
        finished output arrives. Useful for diagnosing whether vLLM batches
        multiple in-flight encode() calls into the same scheduler step.
        """
        # Rust tokenizer uses add_special_tokens=False; prepend BOS to match
        # vllm's default tokenization (add_special_tokens=True).
        if self.bos_token_id is not None and (
            not token_ids or token_ids[0] != self.bos_token_id
        ):
            token_ids = [self.bos_token_id] + token_ids

        prompt = TokensPrompt(prompt_token_ids=token_ids)
        # task="embed" selects the SeqwisePooler (MEAN pooling), which returns
        # a 1D tensor of shape (hidden_dim,) matching LLM.embed() output.
        # Without this, tok_pooling_type='ALL' returns (n_tokens, hidden_dim).
        pooling_params = PoolingParams()
        pooling_params.task = "embed"
        t0 = time.perf_counter() if per_seq_times_ms is not None else 0.0
        async for output in self.engine_client.encode(prompt, pooling_params, request_id):
            if output.finished:
                if per_seq_times_ms is not None:
                    per_seq_times_ms.append((time.perf_counter() - t0) * 1000.0)
                return output.outputs
        raise RuntimeError(f"No output received for request {request_id}")

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate embeddings for a batch of pre-tokenized inputs.

        Args:
            request: PreprocessedEmbeddingRequest dict with token_ids, model, etc.
            context: Context object for cancellation handling.
        """
        t_entry = time.perf_counter() if _TIMING_ENABLED else 0.0

        token_ids_batch: list[list[int]] = request["token_ids"]
        base_request_id = str(uuid.uuid4())

        per_seq_times_ms: Optional[list[float]] = [] if _TIMING_ENABLED else None

        if _DUMMY_ENABLED:
            # Diagnostic path: skip vLLM entirely and fabricate embeddings.
            # Produces response bodies of the same shape (BS × hidden_dim)
            # so the downstream pipeline + JSON encoding + HTTP layer all
            # see the same payload size they would in the real path.
            # Everything except the GPU/engine is on the timing path.
            t_submit = time.perf_counter() if _TIMING_ENABLED else 0.0
            total_tokens = sum(len(t) for t in token_ids_batch)
            # One list allocation shared across all rows — avoids the Python
            # list-creation cost dominating the dummy measurement.  The
            # downstream extraction (typed PyO3 or depythonize) iterates the
            # elements anyway, so sharing the reference does not change the
            # shape of what they process.
            dummy_row = [0.0] * _DUMMY_DIM
            embeddings = [dummy_row for _ in range(len(token_ids_batch))]
            t_gathered = time.perf_counter() if _TIMING_ENABLED else 0.0

            if _TIMING_ENABLED:
                t_packed = time.perf_counter()
                prep_ms = (t_submit - t_entry) * 1000.0
                gather_ms = (t_gathered - t_submit) * 1000.0
                pack_ms = (t_packed - t_gathered) * 1000.0
                handler_ms = (t_packed - t_entry) * 1000.0
                logger.info(
                    "embedding_timing bs=%d total_tokens=%d handler_ms=%.2f "
                    "prep_ms=%.2f gather_ms=%.2f pack_ms=%.2f "
                    "per_seq_p50_ms=0.00 per_seq_min_ms=0.00 per_seq_max_ms=0.00 "
                    "mode=dummy",
                    len(token_ids_batch), total_tokens, handler_ms,
                    prep_ms, gather_ms, pack_ms,
                )

            yield {
                "embeddings": embeddings,
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            }
            return

        # Issue all encode() calls concurrently (one per input sequence)
        tasks = [
            self._encode_single(token_ids, f"{base_request_id}-{i}", per_seq_times_ms)
            for i, token_ids in enumerate(token_ids_batch)
        ]

        t_submit = time.perf_counter() if _TIMING_ENABLED else 0.0

        results = await asyncio.gather(*tasks)

        t_gathered = time.perf_counter() if _TIMING_ENABLED else 0.0

        # With task="embed", PoolingOutput.data is a 1D tensor (hidden_dim,)
        # matching LLM.embed() output — no manual pooling needed.
        embeddings = [result.data.tolist() for result in results]
        total_tokens = sum(len(t) for t in token_ids_batch)

        if _TIMING_ENABLED:
            t_packed = time.perf_counter()
            prep_ms = (t_submit - t_entry) * 1000.0
            gather_ms = (t_gathered - t_submit) * 1000.0
            pack_ms = (t_packed - t_gathered) * 1000.0
            handler_ms = (t_packed - t_entry) * 1000.0
            if per_seq_times_ms:
                p50 = statistics.median(per_seq_times_ms)
                mn = min(per_seq_times_ms)
                mx = max(per_seq_times_ms)
            else:
                p50 = mn = mx = 0.0
            logger.info(
                "embedding_timing bs=%d total_tokens=%d handler_ms=%.2f "
                "prep_ms=%.2f gather_ms=%.2f pack_ms=%.2f "
                "per_seq_p50_ms=%.2f per_seq_min_ms=%.2f per_seq_max_ms=%.2f",
                len(token_ids_batch), total_tokens, handler_ms,
                prep_ms, gather_ms, pack_ms,
                p50, mn, mx,
            )

        yield {
            "embeddings": embeddings,
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }
