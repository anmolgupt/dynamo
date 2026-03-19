#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated text embedding model serving.
# GPUs: 1
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default: llama-nemotron-embed-1b-v2 (requires HuggingFace access or cached weights)
# For a smaller test model, use: intfloat/e5-mistral-7b-instruct or similar
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
print_launch_banner "Launching Embedding Worker (1 GPU)" "$MODEL" "$HTTP_PORT"

# Use file-based discovery and TCP request plane — no etcd/NATS needed
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp

# Run ingress (frontend)
python -m dynamo.frontend &

# Run embedding worker
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --embedding-worker \
    "${EXTRA_ARGS[@]}" &
#    --enforce-eager \
wait_any_exit
