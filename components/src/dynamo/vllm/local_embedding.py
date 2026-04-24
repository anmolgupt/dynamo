# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-process embedding server: Dynamo frontend + vLLM embedding worker
in one process with in-memory discovery and local (in-process) dispatch.

This avoids the TCP request-plane round-trip that the two-process
``agg_embedding.sh`` incurs, giving latency closer to standalone vLLM
while retaining the Dynamo preprocessing pipeline (Rust tokenizer,
OpenAI-compatible API, etc.).

Two in-process dispatch modes are supported, selected by the
``DYN_EMBEDDING_DISPATCH`` environment variable:

* ``json`` (default): generic ``serve_endpoint`` + ``LocalEngineAdapter``
  in the Rust watcher.  Requests/responses round-trip through
  ``serde_json::Value``.  Equivalent to the original implementation.
* ``typed``: ``serve_embedding_endpoint`` + the direct-extraction
  ``PythonEmbeddingEngine``.  Bypasses the ``serde_json::Value`` hop
  and typically saves 15–25 ms per large-batch request.

Usage:
    export DYN_DISCOVERY_BACKEND=mem
    export DYN_REQUEST_PLANE=local
    export DYN_EMBEDDING_DISPATCH=typed   # or "json"

    python -m dynamo.vllm.local_embedding \
        --model nvidia/llama-nemotron-embed-1b-v2 \
        --trust-remote-code
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Any

import uvloop

from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    RouterConfig,
    RouterMode,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def async_main() -> None:
    os.environ.setdefault("DYN_DISCOVERY_BACKEND", "mem")
    os.environ.setdefault("DYN_REQUEST_PLANE", "local")
    os.environ.setdefault("DYN_EVENT_PLANE", "zmq")

    from dynamo.vllm.args import parse_args as parse_worker_args
    from dynamo.vllm.main import setup_vllm_engine, register_vllm_model
    from dynamo.vllm.embedding_handler import EmbeddingWorkerHandler
    from dynamo.vllm.health_check import VllmEmbeddingHealthCheckPayload
    from dynamo.vllm.publisher import StatLoggerFactory
    from dynamo.llm import ModelInput, ModelType
    from dynamo import prometheus_names

    # This entrypoint is always an embedding worker; inject the flag before
    # parsing so update_engine_config_with_dynamo correctly sets
    # runner="pooling" and disables prefix caching on the vLLM engine args.
    # Without this, ModelConfig is built as a generative model and later
    # fails with: 'NoneType' object has no attribute 'seq_pooling_type'.
    if "--embedding-worker" not in sys.argv:
        sys.argv.append("--embedding-worker")

    config = parse_worker_args()
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(
        loop,
        os.environ["DYN_DISCOVERY_BACKEND"],
        os.environ["DYN_REQUEST_PLANE"],
        False,  # enable_nats=False for local mode
    )

    shutdown_event = asyncio.Event()

    def signal_handler():
        shutdown_event.set()
        runtime.shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # --- Embedding worker setup (same as init_embedding.py) ---
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    fpm_worker_id = str(generate_endpoint.connection_id())
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

    handler = EmbeddingWorkerHandler(
        engine_client, config, shutdown_event, vllm_config=vllm_config
    )

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

    # Choose the dispatch registration method based on DYN_EMBEDDING_DISPATCH.
    # Both variants also register the standard JSON engine with discovery so
    # the frontend watcher can still find the endpoint; the "typed" variant
    # additionally registers a direct-extraction typed engine that the
    # embedding branch of the watcher will prefer over the JSON path.
    dispatch_mode = os.environ.get("DYN_EMBEDDING_DISPATCH", "json").strip().lower()
    if dispatch_mode == "typed":
        serve_fn = generate_endpoint.serve_embedding_endpoint
        logger.info(
            "Local embedding server: using TYPED direct-extraction dispatch "
            "(DYN_EMBEDDING_DISPATCH=typed)"
        )
    else:
        serve_fn = generate_endpoint.serve_endpoint
        logger.info(
            "Local embedding server: using JSON dispatch "
            "(DYN_EMBEDDING_DISPATCH=json)"
        )

    # Note: serve_endpoint / serve_embedding_endpoint are PyO3-bridged
    # methods that return a Python Future (not a coroutine), so use
    # ensure_future instead of create_task.
    worker_task = asyncio.ensure_future(
        serve_fn(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=model_metrics_labels,
            health_check_payload=health_check_payload,
        )
    )

    # Small delay to let the worker register with discovery before the
    # frontend watcher starts scanning.
    await asyncio.sleep(0.5)

    # --- Frontend HTTP server setup ---
    http_port = int(os.environ.get("DYN_HTTP_PORT", "8000"))
    router_config = RouterConfig(RouterMode.RoundRobin)

    kwargs: dict[str, Any] = {
        "http_host": "0.0.0.0",
        "http_port": http_port,
        "router_config": router_config,
    }

    e = EntrypointArgs(EngineType.Dynamic, **kwargs)
    engine = await make_engine(runtime, e)

    logger.info(
        "Local embedding server ready — "
        f"model={config.model}, port={http_port}, "
        f"request_plane=local, discovery=mem, "
        f"dispatch={dispatch_mode}"
    )

    try:
        await run_input(runtime, "http", engine)
    except asyncio.CancelledError:
        pass
    finally:
        worker_task.cancel()
        if prometheus_temp_dir is not None:
            prometheus_temp_dir.cleanup()


def main() -> None:
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
