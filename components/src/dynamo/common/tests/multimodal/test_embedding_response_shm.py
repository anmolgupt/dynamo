# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import multiprocessing.shared_memory as shm
import os
import uuid

import pytest
import torch

from dynamo.common.multimodal.embedding_response_shm import (
    apply_embedding_request_shm,
    maybe_write_embedding_response_shm,
    pooling_output_to_tensor,
)


def request_shm_name() -> str:
    return f"dyn_embed_req_pytest_{os.getpid()}_{uuid.uuid4().hex[:16]}"


def write_request_payload(payload: bytes) -> tuple[str, str]:
    name = request_shm_name()
    path = os.path.join("/dev/shm", name)
    with open(path, "wb") as f:
        f.write(payload)
    return name, path


def typed_token_payload(rows: list[list[int]]) -> bytes:
    total_tokens = sum(len(row) for row in rows)
    payload = bytearray(struct.pack("<8sIII", b"DYNEMBTK", 1, len(rows), total_tokens))
    offset = 0
    payload.extend(struct.pack("<I", offset))
    for row in rows:
        offset += len(row)
        payload.extend(struct.pack("<I", offset))
    for row in rows:
        for token_id in row:
            payload.extend(struct.pack("<I", token_id))
    return bytes(payload)


def test_pooling_output_to_tensor_flattens_tensor():
    data = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    out = pooling_output_to_tensor(data)

    assert out.dtype == torch.float32
    assert out.tolist() == [1.0, 2.0]


def test_apply_embedding_request_shm_roundtrip_input():
    payload = json.dumps(["hello", "world"]).encode("utf-8")
    name, path = write_request_payload(payload)
    request = {
        "model": "test-model",
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "text",
            "name": name,
            "size_bytes": len(payload),
            "payload_kind": "json",
            "field": "input",
        },
    }

    apply_embedding_request_shm(request)

    assert request["input"] == ["hello", "world"]
    assert "embedding_request_shm" not in request
    assert not os.path.exists(path)


def test_apply_embedding_request_shm_roundtrip_token_ids():
    payload = json.dumps([[1, 2, 3], [4, 5]]).encode("utf-8")
    name, path = write_request_payload(payload)
    request = {
        "model": "test-model",
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "tokens",
            "name": name,
            "size_bytes": len(payload),
            "payload_kind": "json",
            "field": "token_ids",
        },
    }

    apply_embedding_request_shm(request)

    assert request["token_ids"] == [[1, 2, 3], [4, 5]]
    assert "embedding_request_shm" not in request
    assert not os.path.exists(path)


def test_apply_embedding_request_shm_roundtrip_typed_token_ids():
    payload = typed_token_payload([[1, 2, 3], [4, 5]])
    name, path = write_request_payload(payload)
    request = {
        "model": "test-model",
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "tokens",
            "name": name,
            "size_bytes": len(payload),
            "payload_kind": "token_ids_u32_offsets_v1",
            "field": "token_ids",
        },
    }

    apply_embedding_request_shm(request)

    assert request["token_ids"] == [[1, 2, 3], [4, 5]]
    assert "embedding_request_shm" not in request
    assert not os.path.exists(path)


def test_apply_embedding_request_shm_rejects_typed_tokens_for_input_field():
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "tokens",
            "name": "dyn_embed_req_not_read",
            "size_bytes": 20,
            "payload_kind": "token_ids_u32_offsets_v1",
            "field": "input",
        },
    }

    with pytest.raises(ValueError, match="only valid for token_ids"):
        apply_embedding_request_shm(request)


def test_apply_embedding_request_shm_rejects_bad_typed_token_magic_and_unlinks():
    payload = bytearray(typed_token_payload([[1, 2]]))
    payload[0:8] = b"BADMAGIC"
    name, path = write_request_payload(bytes(payload))
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "tokens",
            "name": name,
            "size_bytes": len(payload),
            "payload_kind": "token_ids_u32_offsets_v1",
            "field": "token_ids",
        },
    }

    with pytest.raises(ValueError, match="invalid typed token SHM magic"):
        apply_embedding_request_shm(request)
    assert not os.path.exists(path)


def test_apply_embedding_request_shm_rejects_non_monotonic_typed_token_offsets_and_unlinks():
    payload = bytearray(struct.pack("<8sIII", b"DYNEMBTK", 1, 3, 3))
    for value in [0, 2, 1, 3, 1, 2, 3]:
        payload.extend(struct.pack("<I", value))
    name, path = write_request_payload(bytes(payload))
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "tokens",
            "name": name,
            "size_bytes": len(payload),
            "payload_kind": "token_ids_u32_offsets_v1",
            "field": "token_ids",
        },
    }

    with pytest.raises(ValueError, match="offsets must be monotonic"):
        apply_embedding_request_shm(request)
    assert not os.path.exists(path)


def test_apply_embedding_request_shm_rejects_bad_name():
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "text",
            "name": "../bad",
            "size_bytes": 2,
            "payload_kind": "json",
            "field": "input",
        },
    }

    with pytest.raises(ValueError, match="invalid embedding SHM name"):
        apply_embedding_request_shm(request)


def test_apply_embedding_request_shm_rejects_bad_prefix():
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "text",
            "name": "other_prefix",
            "size_bytes": 2,
            "payload_kind": "json",
            "field": "input",
        },
    }

    with pytest.raises(ValueError, match="invalid embedding SHM name"):
        apply_embedding_request_shm(request)


def test_apply_embedding_request_shm_invalid_json_unlinks():
    name, path = write_request_payload(b"not-json")
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "text",
            "name": name,
            "size_bytes": len(b"not-json"),
            "payload_kind": "json",
            "field": "input",
        },
    }

    with pytest.raises(json.JSONDecodeError):
        apply_embedding_request_shm(request)
    assert not os.path.exists(path)


def test_apply_embedding_request_shm_rejects_larger_segment_and_unlinks():
    payload = json.dumps(["hello"]).encode("utf-8")
    name, path = write_request_payload(payload + b" ")
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "text",
            "name": name,
            "size_bytes": len(payload),
            "payload_kind": "json",
            "field": "input",
        },
    }

    with pytest.raises(ValueError, match="larger than metadata"):
        apply_embedding_request_shm(request)
    assert not os.path.exists(path)


def test_apply_embedding_request_shm_rejects_payload_over_limit_and_unlinks(monkeypatch):
    payload = json.dumps(["hello"]).encode("utf-8")
    name, path = write_request_payload(payload)
    monkeypatch.setenv("DYN_EMBEDDING_SHM_REQUEST_MAX_BYTES", str(len(payload) - 1))
    request = {
        "embedding_request_shm": {
            "version": 1,
            "model": "test-model",
            "path": "text",
            "name": name,
            "size_bytes": len(payload),
            "payload_kind": "json",
            "field": "input",
        },
    }

    with pytest.raises(ValueError, match="payload too large"):
        apply_embedding_request_shm(request)
    assert not os.path.exists(path)


def test_write_embedding_response_shm_roundtrip(monkeypatch):
    monkeypatch.setenv("DYN_EMBEDDING_SHM_RESPONSE", "1")
    monkeypatch.setenv("DYN_EMBEDDING_SHM_MIN_BYTES", "0")

    result = maybe_write_embedding_response_shm(
        [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 4.0], dtype=torch.float32),
        ],
        model="test-model",
        path="tokens",
        encoding_format="float",
    )

    assert result is not None
    try:
        meta = result.metadata
        assert meta.version == 1
        assert meta.model == "test-model"
        assert meta.path == "tokens"
        assert meta.dtype == "float32"
        assert meta.shape == [2, 2]
        assert meta.size_bytes == 16

        reader = shm.SharedMemory(name=meta.name, create=False)
        try:
            assert bytes(reader.buf[: meta.size_bytes]) == (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
                .numpy()
                .tobytes()
            )
        finally:
            reader.close()
    finally:
        result.handle.close()
        result.handle.unlink()


def test_write_embedding_response_shm_disabled(monkeypatch):
    monkeypatch.delenv("DYN_EMBEDDING_SHM_RESPONSE", raising=False)

    result = maybe_write_embedding_response_shm(
        [torch.tensor([1.0, 2.0], dtype=torch.float32)],
        model="test-model",
        path="text",
        encoding_format="float",
    )

    assert result is None


def test_write_embedding_response_shm_rejects_base64(monkeypatch):
    monkeypatch.setenv("DYN_EMBEDDING_SHM_RESPONSE", "1")
    monkeypatch.setenv("DYN_EMBEDDING_SHM_MIN_BYTES", "0")

    result = maybe_write_embedding_response_shm(
        [torch.tensor([1.0, 2.0], dtype=torch.float32)],
        model="test-model",
        path="text",
        encoding_format="base64",
    )

    assert result is None


def test_write_embedding_response_shm_rejects_ragged_rows(monkeypatch):
    monkeypatch.setenv("DYN_EMBEDDING_SHM_RESPONSE", "1")
    monkeypatch.setenv("DYN_EMBEDDING_SHM_MIN_BYTES", "0")

    result = maybe_write_embedding_response_shm(
        [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
        ],
        model="test-model",
        path="text",
        encoding_format="float",
    )

    assert result is None


def test_write_embedding_response_shm_handles_create_failure(monkeypatch):
    monkeypatch.setenv("DYN_EMBEDDING_SHM_RESPONSE", "1")
    monkeypatch.setenv("DYN_EMBEDDING_SHM_MIN_BYTES", "0")

    def fail_shared_memory(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(shm, "SharedMemory", fail_shared_memory)

    result = maybe_write_embedding_response_shm(
        [torch.tensor([1.0, 2.0], dtype=torch.float32)],
        model="test-model",
        path="text",
        encoding_format="float",
    )

    assert result is None
