#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Hybrid embedding serving demo.
#
# Demonstrates the local-first hybrid request plane: the Dynamo frontend
# runs in `local` mode and the embedding worker runs in `tcp` mode, in
# separate processes (the worker holds the GPU, the frontend is CPU-only).
#
# This is the topology that motivates the hybrid behaviour — for example
# in an agentic-retrieval deployment where the frontend co-locates a
# cheap worker on its own process but still needs to reach a remote
# heavy worker (LLM) on a different GPU/process.  Here we exercise just
# the TCP-fallback half of the hybrid path: the frontend's local
# registry will be empty (the worker lives in another process), so all
# embedding requests must take the TCP fallback.
#
# A successful run proves that a frontend in `Local` mode can still
# dispatch to remote endpoints via TCP — i.e. that `Local` is a strict
# superset of `Tcp` from the client perspective.
#
# GPUs: 1
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="nvidia/llama-nemotron-embed-1b-v2"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Hybrid Embedding Server (frontend=local, worker=tcp)" "$MODEL" "$HTTP_PORT"

# Both processes share file-based discovery so the frontend can see the
# worker registered in the shared store.
export DYN_DISCOVERY_BACKEND=file

# Run frontend in local-first hybrid mode.  Empty local registry → all
# dispatch falls through to the TCP client, reaching the remote worker.
DYN_REQUEST_PLANE=local \
    python -m dynamo.frontend &

# Run embedding worker in standard TCP mode so it binds a TCP request-
# plane server reachable from the frontend.
DYN_REQUEST_PLANE=tcp \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --embedding-worker \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
