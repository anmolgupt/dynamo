# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from vllm.inputs import TokensPrompt
from vllm.outputs import PoolingOutput
from vllm.pooling_params import PoolingParams

from dynamo._core import Context

logger = logging.getLogger(__name__)


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
        logger.info("EmbeddingWorkerHandler initialized")

    async def _encode_single(
        self, token_ids: list[int], request_id: str
    ) -> PoolingOutput:
        """Encode a single token sequence and return the EmbeddingOutput."""
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
        async for output in self.engine_client.encode(prompt, pooling_params, request_id):
            if output.finished:
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
        token_ids_batch: list[list[int]] = request["token_ids"]
        base_request_id = str(uuid.uuid4())

        # Issue all encode() calls concurrently (one per input sequence)
        tasks = [
            self._encode_single(token_ids, f"{base_request_id}-{i}")
            for i, token_ids in enumerate(token_ids_batch)
        ]

        results = await asyncio.gather(*tasks)

        # With task="embed", PoolingOutput.data is a 1D tensor (hidden_dim,)
        # matching LLM.embed() output — no manual pooling needed.
        embeddings = [result.data.tolist() for result in results]
        total_tokens = sum(len(t) for t in token_ids_batch)

        yield {
            "embeddings": embeddings,
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }
