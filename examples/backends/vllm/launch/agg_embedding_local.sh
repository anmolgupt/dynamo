#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-process embedding serving (local dispatch, no TCP overhead).
#
# Runs the Dynamo frontend and vLLM embedding worker in one process using
# in-memory discovery and the LocalEndpointRegistry for dispatch.  This
# eliminates the TCP round-trip between frontend and worker, giving latency
# closer to standalone vLLM while keeping the Dynamo API layer.
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
print_launch_banner "Launching Local Embedding Server (1 GPU, single process)" "$MODEL" "$HTTP_PORT"

export DYN_DISCOVERY_BACKEND=mem
export DYN_REQUEST_PLANE=local
export DYN_HTTP_PORT="${HTTP_PORT}"

python -m dynamo.vllm.local_embedding \
    --model "$MODEL" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
