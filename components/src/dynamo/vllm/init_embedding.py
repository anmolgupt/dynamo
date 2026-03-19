# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo import prometheus_names
from dynamo.common.utils.prometheus import LLMBackendMetrics
from dynamo.llm import ModelInput, ModelType
from dynamo.runtime import DistributedRuntime

from .args import Config
from .embedding_handler import EmbeddingWorkerHandler
from .health_check import VllmEmbeddingHealthCheckPayload
from .publisher import StatLoggerFactory

logger = logging.getLogger(__name__)


async def init_embedding(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    snapshot_engine: Optional[
        tuple[AsyncLLM, VllmConfig, Any, Any, LLMBackendMetrics]
    ] = None,
) -> None:
    """Initialize and serve a vLLM text embedding worker."""
    from .main import (
        register_vllm_model,
        setup_metrics_collection,
        setup_vllm_engine,
    )

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    fpm_worker_id = str(generate_endpoint.connection_id())
    if snapshot_engine is not None:
        (
            engine_client,
            vllm_config,
            _default_sampling_params,
            prometheus_temp_dir,
            component_gauges,
        ) = snapshot_engine
        vllm_config.additional_config["fpm_worker_id"] = fpm_worker_id
        factory = StatLoggerFactory(
            endpoint=generate_endpoint,
            component_gauges=component_gauges,
        )
    else:
        factory = StatLoggerFactory(endpoint=generate_endpoint)
        (
            engine_client,
            vllm_config,
            _default_sampling_params,
            prometheus_temp_dir,
            component_gauges,
        ) = setup_vllm_engine(config, factory, fpm_worker_id=fpm_worker_id)

    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.init_publish()

    handler = EmbeddingWorkerHandler(engine_client, config, shutdown_event, vllm_config=vllm_config)

    setup_metrics_collection(config, generate_endpoint, logger)

    await register_vllm_model(
        ModelInput.Tokens,
        ModelType.Embedding,
        generate_endpoint,
        config,
        engine_client,
        vllm_config,
    )

    health_check_payload = VllmEmbeddingHealthCheckPayload(engine_client).to_dict()

    model_metrics_labels = [
        (prometheus_names.labels.MODEL, config.served_model_name or config.model),
        (prometheus_names.labels.MODEL_NAME, config.served_model_name or config.model),
    ]

    try:
        logger.debug("Starting serve_endpoint for embedding worker")
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=model_metrics_labels,
            health_check_payload=health_check_payload,
        )
        logger.debug("serve_endpoint completed for embedding worker")
    except Exception as e:
        logger.error(f"Failed to serve embedding endpoint: {e}")
        raise
    finally:
        logger.debug("Cleaning up embedding worker")
        if prometheus_temp_dir is not None:
            prometheus_temp_dir.cleanup()
