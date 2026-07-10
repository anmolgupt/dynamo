// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    env, fs,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
    sync::{
        LazyLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry};

use crate::protocols::common::{
    llm_backend::{EmbeddingRequestShmMetadata, EmbeddingResponseShmMetadata},
    preprocessor::PreprocessedEmbeddingRequest,
};

use super::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse};

const REQUEST_PAYLOAD_KIND_JSON: &str = "json";
const SUPPORTED_VERSION: u32 = 1;
const FLOAT32_SIZE: usize = 4;
const DEFAULT_RESPONSE_SHM_MAX_BYTES: usize = 256 * 1024 * 1024;

static REQUEST_SHM_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static SHM_SUCCESS_LOGGED: AtomicBool = AtomicBool::new(false);

static REQUEST_SHM_WRITES: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_frontend_embedding_request_shm_writes_total",
            "Embedding request SHM writes by status.",
        ),
        &["model", "path", "status"],
    )
    .expect("failed to create embedding request SHM write counter")
});

static REQUEST_SHM_BYTES_WRITTEN: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_frontend_embedding_request_shm_bytes_written_total",
            "Embedding request bytes written to shared memory.",
        ),
        &["model", "path"],
    )
    .expect("failed to create embedding request SHM bytes counter")
});

static REQUEST_SHM_WRITE_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_frontend_embedding_request_shm_write_seconds",
            "Time spent writing embedding requests to shared memory.",
        )
        .buckets(vec![
            0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        ]),
        &["model", "path"],
    )
    .expect("failed to create embedding request SHM write histogram")
});

static REQUEST_SHM_FALLBACKS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_frontend_embedding_request_shm_fallback_total",
            "Embedding request SHM fallback and failure reasons.",
        ),
        &["model", "path", "reason"],
    )
    .expect("failed to create embedding request SHM fallback counter")
});

static SHM_REQUESTS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_frontend_embedding_shm_requests_total",
            "Embedding SHM response reads by status.",
        ),
        &["model", "path", "status"],
    )
    .expect("failed to create embedding SHM request counter")
});

static SHM_BYTES_READ: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_frontend_embedding_shm_bytes_read_total",
            "Embedding response bytes read from shared memory.",
        ),
        &["model", "path"],
    )
    .expect("failed to create embedding SHM bytes counter")
});

static SHM_READ_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_frontend_embedding_shm_read_seconds",
            "Time spent reading embedding responses from shared memory.",
        )
        .buckets(vec![
            0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        ]),
        &["model", "path"],
    )
    .expect("failed to create embedding SHM read histogram")
});

static SHM_FALLBACKS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_frontend_embedding_shm_fallback_total",
            "Embedding SHM response fallback and failure reasons.",
        ),
        &["model", "path", "reason"],
    )
    .expect("failed to create embedding SHM fallback counter")
});

static SHM_CLEANUP_FAILURES: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_frontend_embedding_shm_cleanup_failures_total",
            "Embedding SHM unlink failures observed by the frontend.",
        ),
        &["model", "path"],
    )
    .expect("failed to create embedding SHM cleanup counter")
});

pub fn register_embedding_shm_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(REQUEST_SHM_WRITES.clone()))?;
    registry.register(Box::new(REQUEST_SHM_BYTES_WRITTEN.clone()))?;
    registry.register(Box::new(REQUEST_SHM_WRITE_SECONDS.clone()))?;
    registry.register(Box::new(REQUEST_SHM_FALLBACKS.clone()))?;
    registry.register(Box::new(SHM_REQUESTS.clone()))?;
    registry.register(Box::new(SHM_BYTES_READ.clone()))?;
    registry.register(Box::new(SHM_READ_SECONDS.clone()))?;
    registry.register(Box::new(SHM_FALLBACKS.clone()))?;
    registry.register(Box::new(SHM_CLEANUP_FAILURES.clone()))?;
    Ok(())
}

fn env_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no"
            )
        })
        .unwrap_or(false)
}

fn request_shm_enabled() -> bool {
    env_enabled("DYN_EMBEDDING_SHM_REQUEST")
}

fn frontend_tokenization_enabled() -> bool {
    env_enabled("DYN_EMBEDDING_FRONTEND_TOKENIZATION")
}

fn request_shm_min_bytes() -> usize {
    env::var("DYN_EMBEDDING_SHM_REQUEST_MIN_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(65536)
}

fn response_shm_max_bytes() -> usize {
    env::var("DYN_EMBEDDING_SHM_RESPONSE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_RESPONSE_SHM_MAX_BYTES)
}

fn record_request_fallback(model: &str, path: &str, reason: &str) {
    REQUEST_SHM_WRITES
        .with_label_values(&[model, path, "fallback"])
        .inc();
    REQUEST_SHM_FALLBACKS
        .with_label_values(&[model, path, reason])
        .inc();
}

/// Move the raw OpenAI `input` payload into SHM for Text+Embedding routing.
///
/// This must run only for the vLLM-tokenized topology. When frontend
/// tokenization is enabled, the Rust preprocessor needs the original text to
/// create token IDs and will write those token IDs to request SHM instead.
pub fn maybe_write_embedding_request_shm(
    request: &mut NvCreateEmbeddingRequest,
    path: &str,
    request_id: &str,
) -> Result<bool, String> {
    let model = request.inner.model.clone();
    if frontend_tokenization_enabled() {
        record_request_fallback(model.as_str(), path, "frontend_tokenization");
        return Ok(false);
    }
    if !request_shm_enabled() {
        record_request_fallback(model.as_str(), path, "disabled");
        return Ok(false);
    }
    let payload = serde_json::to_vec(&request.inner.input).map_err(|e| {
        REQUEST_SHM_WRITES
            .with_label_values(&[model.as_str(), path, "fallback"])
            .inc();
        REQUEST_SHM_FALLBACKS
            .with_label_values(&[model.as_str(), path, "serialize_failed"])
            .inc();
        format!("failed to serialize embedding request input for SHM: {e}")
    })?;
    let Some(meta) = write_request_payload(
        model.as_str(),
        path,
        REQUEST_PAYLOAD_KIND_JSON,
        "input",
        &payload,
        request_id,
    )?
    else {
        return Ok(false);
    };

    request.embedding_request_shm = Some(meta);
    // Once the internal SHM descriptor is set, downstream code restores the
    // original variant before inspecting input. This is only a compact placeholder.
    request.inner.input = dynamo_protocols::types::EmbeddingInput::StringArray(vec![]);
    Ok(true)
}

/// Move Rust-tokenized embedding request IDs into SHM for Tokens+Embedding routing.
pub fn maybe_write_preprocessed_embedding_request_shm(
    request: &mut PreprocessedEmbeddingRequest,
    path: &str,
    request_id: &str,
) -> Result<bool, String> {
    let model = request.model.clone();
    if !request_shm_enabled() {
        record_request_fallback(model.as_str(), path, "disabled");
        return Ok(false);
    }

    let payload = serde_json::to_vec(&request.token_ids).map_err(|e| {
        REQUEST_SHM_WRITES
            .with_label_values(&[model.as_str(), path, "fallback"])
            .inc();
        REQUEST_SHM_FALLBACKS
            .with_label_values(&[model.as_str(), path, "serialize_failed"])
            .inc();
        format!("failed to serialize embedding token_ids for SHM: {e}")
    })?;
    let Some(meta) = write_request_payload(
        model.as_str(),
        path,
        REQUEST_PAYLOAD_KIND_JSON,
        "token_ids",
        &payload,
        request_id,
    )?
    else {
        return Ok(false);
    };

    request.embedding_request_shm = Some(meta);
    request.token_ids.clear();
    Ok(true)
}

fn write_request_payload(
    model: &str,
    path: &str,
    payload_kind: &str,
    field: &str,
    payload: &[u8],
    request_id: &str,
) -> Result<Option<EmbeddingRequestShmMetadata>, String> {
    if payload.is_empty() {
        record_request_fallback(model, path, "empty");
        return Ok(None);
    }
    if payload.len() < request_shm_min_bytes() {
        record_request_fallback(model, path, "small_payload");
        return Ok(None);
    }

    let seq = REQUEST_SHM_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let name = format!("dyn_embed_req_{}_{}", std::process::id(), seq);
    let shm_path = shm_path(&name)?;
    let start = Instant::now();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&shm_path)
        .map_err(|e| {
            REQUEST_SHM_WRITES
                .with_label_values(&[model, path, "fallback"])
                .inc();
            REQUEST_SHM_FALLBACKS
                .with_label_values(&[model, path, "write_failed"])
                .inc();
            format!("failed to create embedding request SHM segment {name}: {e}")
        })?;
    if let Err(e) = file.write_all(payload) {
        let _ = fs::remove_file(&shm_path);
        REQUEST_SHM_WRITES
            .with_label_values(&[model, path, "fallback"])
            .inc();
        REQUEST_SHM_FALLBACKS
            .with_label_values(&[model, path, "write_failed"])
            .inc();
        return Err(format!(
            "failed to write embedding request SHM segment {name} for {request_id}: {e}"
        ));
    }

    REQUEST_SHM_WRITE_SECONDS
        .with_label_values(&[model, path])
        .observe(start.elapsed().as_secs_f64());
    REQUEST_SHM_BYTES_WRITTEN
        .with_label_values(&[model, path])
        .inc_by(payload.len() as u64);
    REQUEST_SHM_WRITES
        .with_label_values(&[model, path, "success"])
        .inc();

    tracing::debug!(
        shm_name = %name,
        bytes = payload.len(),
        request_id = request_id,
        field = field,
        "wrote embedding request to shared memory"
    );

    Ok(Some(EmbeddingRequestShmMetadata {
        version: SUPPORTED_VERSION,
        model: model.to_string(),
        path: path.to_string(),
        name,
        size_bytes: payload.len(),
        payload_kind: payload_kind.to_string(),
        field: field.to_string(),
    }))
}

/// Expand and clear the internal SHM descriptor on an OpenAI embedding response.
pub fn expand_embedding_response_shm(
    response: &mut NvCreateEmbeddingResponse,
) -> Result<(), String> {
    let Some(meta) = response.embedding_response_shm.take() else {
        return Ok(());
    };

    let embeddings = metadata_to_embeddings(&meta)?;
    response.inner.data = embeddings;
    Ok(())
}

/// Read a SHM segment and convert it to OpenAI embedding rows.
pub fn metadata_to_embeddings(
    meta: &EmbeddingResponseShmMetadata,
) -> Result<Vec<dynamo_protocols::types::Embedding>, String> {
    let bytes = read_embedding_response_shm(meta)?;
    float32_bytes_to_embeddings(meta, &bytes)
}

/// Read and unlink a POSIX shared-memory segment created by Python.
pub fn read_embedding_response_shm(meta: &EmbeddingResponseShmMetadata) -> Result<Vec<u8>, String> {
    if let Err(err) = validate_metadata(meta) {
        SHM_REQUESTS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "error"])
            .inc();
        SHM_FALLBACKS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "validation_failed"])
            .inc();
        return Err(err);
    }

    let path = shm_path(&meta.name)?;
    let start = Instant::now();
    let max_bytes = response_shm_max_bytes();
    if meta.size_bytes > max_bytes {
        let _ = fs::remove_file(&path);
        SHM_REQUESTS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "error"])
            .inc();
        SHM_FALLBACKS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "payload_too_large"])
            .inc();
        return Err(format!(
            "embedding SHM response payload is too large: {} bytes, max {}",
            meta.size_bytes, max_bytes
        ));
    }
    let file = File::open(&path).map_err(|e| {
        SHM_REQUESTS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "error"])
            .inc();
        SHM_FALLBACKS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "read_failed"])
            .inc();
        format!(
            "failed to open embedding SHM segment {}: {e}; ensure the frontend and worker share the same IPC namespace (/dev/shm)",
            meta.name
        )
    })?;
    let max_read = meta.size_bytes.saturating_add(1) as u64;
    let mut bytes = Vec::with_capacity(meta.size_bytes);
    let mut reader = file.take(max_read);
    reader.read_to_end(&mut bytes).map_err(|e| {
        SHM_REQUESTS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "error"])
            .inc();
        SHM_FALLBACKS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "read_failed"])
            .inc();
        format!("failed to read embedding SHM segment {}: {e}", meta.name)
    })?;

    if bytes.len() < meta.size_bytes {
        let _ = fs::remove_file(&path);
        SHM_REQUESTS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "error"])
            .inc();
        SHM_FALLBACKS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "validation_failed"])
            .inc();
        return Err(format!(
            "embedding SHM segment {} is too small: got {} bytes, expected {}",
            meta.name,
            bytes.len(),
            meta.size_bytes
        ));
    }
    if bytes.len() > meta.size_bytes {
        let _ = fs::remove_file(&path);
        SHM_REQUESTS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "error"])
            .inc();
        SHM_FALLBACKS
            .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "validation_failed"])
            .inc();
        return Err(format!(
            "embedding SHM segment {} is larger than metadata: got at least {} bytes, expected {}",
            meta.name,
            bytes.len(),
            meta.size_bytes
        ));
    }

    fs::remove_file(&path).map_err(|e| {
        SHM_CLEANUP_FAILURES
            .with_label_values(&[meta.model.as_str(), meta.path.as_str()])
            .inc();
        format!("failed to unlink embedding SHM segment {}: {e}", meta.name)
    })?;
    SHM_REQUESTS
        .with_label_values(&[meta.model.as_str(), meta.path.as_str(), "success"])
        .inc();
    SHM_BYTES_READ
        .with_label_values(&[meta.model.as_str(), meta.path.as_str()])
        .inc_by(meta.size_bytes as u64);
    SHM_READ_SECONDS
        .with_label_values(&[meta.model.as_str(), meta.path.as_str()])
        .observe(start.elapsed().as_secs_f64());

    if !SHM_SUCCESS_LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            shm_name = %meta.name,
            bytes = meta.size_bytes,
            "embedding SHM transport is working; frontend and worker share /dev/shm"
        );
    }

    tracing::debug!(
        shm_name = %meta.name,
        bytes = meta.size_bytes,
        elapsed_us = start.elapsed().as_micros(),
        "read embedding response from shared memory"
    );

    Ok(bytes[..meta.size_bytes].to_vec())
}

fn validate_metadata(meta: &EmbeddingResponseShmMetadata) -> Result<(), String> {
    if meta.version != SUPPORTED_VERSION {
        return Err(format!(
            "unsupported embedding SHM version {}, expected {}",
            meta.version, SUPPORTED_VERSION
        ));
    }
    if meta.dtype != "float32" {
        return Err(format!("unsupported embedding SHM dtype {}", meta.dtype));
    }
    if meta.endianness != "native" {
        return Err(format!(
            "unsupported embedding SHM endianness {}",
            meta.endianness
        ));
    }
    if meta.embedding_type != "float" {
        return Err(format!(
            "unsupported embedding_type {} for SHM response",
            meta.embedding_type
        ));
    }
    if meta.encoding_format != "float" {
        return Err(format!(
            "unsupported encoding_format {} for SHM response",
            meta.encoding_format
        ));
    }
    if meta.shape.len() != 2 || meta.shape[0] == 0 || meta.shape[1] == 0 {
        return Err(format!(
            "embedding SHM shape must be [batch, dim], got {:?}",
            meta.shape
        ));
    }
    let expected = meta.shape[0]
        .checked_mul(meta.shape[1])
        .and_then(|n| n.checked_mul(FLOAT32_SIZE))
        .ok_or_else(|| "embedding SHM shape overflows byte length".to_string())?;
    if expected != meta.size_bytes {
        return Err(format!(
            "embedding SHM size mismatch: metadata has {} bytes, shape implies {}",
            meta.size_bytes, expected
        ));
    }
    if !meta.name.starts_with("dyn_embed_resp_") {
        return Err(format!(
            "invalid embedding SHM response name {:?}",
            meta.name
        ));
    }
    let _ = shm_path(&meta.name)?;
    Ok(())
}

fn shm_path(name: &str) -> Result<PathBuf, String> {
    if name.is_empty()
        || name.contains('/')
        || name.contains("..")
        || name.as_bytes().contains(&0)
        || !(name.starts_with("dyn_embed_req_") || name.starts_with("dyn_embed_resp_"))
    {
        return Err(format!("invalid embedding SHM name {name:?}"));
    }
    Ok(PathBuf::from("/dev/shm").join(name))
}

fn float32_bytes_to_embeddings(
    meta: &EmbeddingResponseShmMetadata,
    bytes: &[u8],
) -> Result<Vec<dynamo_protocols::types::Embedding>, String> {
    validate_metadata(meta)?;
    let batch = meta.shape[0];
    let dim = meta.shape[1];
    if bytes.len() < meta.size_bytes {
        return Err(format!(
            "embedding SHM byte buffer is too small: got {} bytes, expected {}",
            bytes.len(),
            meta.size_bytes
        ));
    }

    let mut embeddings = Vec::with_capacity(batch);

    for row_idx in 0..batch {
        let mut row = Vec::with_capacity(dim);
        let row_start = row_idx * dim * FLOAT32_SIZE;
        for col_idx in 0..dim {
            let i = row_start + col_idx * FLOAT32_SIZE;
            let chunk: [u8; FLOAT32_SIZE] = bytes[i..i + FLOAT32_SIZE]
                .try_into()
                .map_err(|_| "failed to read float32 bytes".to_string())?;
            row.push(f32::from_ne_bytes(chunk));
        }
        embeddings.push(dynamo_protocols::types::Embedding {
            index: row_idx as u32,
            object: "embedding".to_string(),
            embedding: row,
        });
    }

    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> EmbeddingResponseShmMetadata {
        EmbeddingResponseShmMetadata {
            version: 1,
            model: "test-model".to_string(),
            path: "tokens".to_string(),
            name: "dyn_embed_resp_test".to_string(),
            size_bytes: 16,
            dtype: "float32".to_string(),
            endianness: "native".to_string(),
            shape: vec![2, 2],
            embedding_type: "float".to_string(),
            encoding_format: "float".to_string(),
        }
    }

    #[test]
    fn converts_float32_bytes_to_embeddings() {
        let meta = metadata();
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.5, -4.0] {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }

        let rows = float32_bytes_to_embeddings(&meta, &bytes).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].embedding, vec![1.0, 2.0]);
        assert_eq!(rows[1].embedding, vec![3.5, -4.0]);
    }

    #[test]
    fn rejects_short_byte_buffer_without_panic() {
        let meta = metadata();

        let err = float32_bytes_to_embeddings(&meta, &[0; 12]).unwrap_err();

        assert!(err.contains("byte buffer is too small"));
    }

    #[test]
    fn reads_exact_segment_and_unlinks() {
        let mut meta = metadata();
        meta.name = unique_name("dyn_embed_resp_exact_test");
        let path = shm_path(&meta.name).unwrap();
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.5, -4.0] {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        fs::write(&path, &bytes).unwrap();

        let read = read_embedding_response_shm(&meta).unwrap();

        assert_eq!(read, bytes);
        assert!(!path.exists());
    }

    #[test]
    fn rejects_response_payload_over_limit_and_unlinks() {
        let mut meta = metadata();
        meta.name = unique_name("dyn_embed_resp_over_limit_test");
        meta.shape = vec![1, DEFAULT_RESPONSE_SHM_MAX_BYTES / FLOAT32_SIZE + 1];
        meta.size_bytes = meta.shape[1] * FLOAT32_SIZE;
        let path = shm_path(&meta.name).unwrap();
        fs::write(&path, [0u8; 4]).unwrap();

        let err = read_embedding_response_shm(&meta).unwrap_err();

        assert!(err.contains("payload is too large"));
        assert!(!path.exists());
    }

    #[test]
    fn rejects_oversized_segment_and_unlinks() {
        let mut meta = metadata();
        meta.name = unique_name("dyn_embed_resp_oversized_test");
        let path = shm_path(&meta.name).unwrap();
        fs::write(&path, vec![0u8; meta.size_bytes + 4]).unwrap();

        let err = read_embedding_response_shm(&meta).unwrap_err();

        assert!(err.contains("larger than metadata"));
        assert!(!path.exists());
    }

    #[test]
    fn rejects_size_mismatch() {
        let mut meta = metadata();
        meta.size_bytes = 12;
        assert!(validate_metadata(&meta).is_err());
    }

    #[test]
    fn rejects_bad_name() {
        let mut meta = metadata();
        meta.name = "../bad".to_string();
        assert!(validate_metadata(&meta).is_err());
    }

    #[test]
    fn rejects_bad_prefix() {
        let mut meta = metadata();
        meta.name = "other_prefix".to_string();
        assert!(validate_metadata(&meta).is_err());
    }

    #[test]
    fn rejects_request_prefix_for_response_metadata() {
        let mut meta = metadata();
        meta.name = "dyn_embed_req_not_response".to_string();
        assert!(validate_metadata(&meta).is_err());
    }

    fn unique_name(prefix: &str) -> String {
        let seq = REQUEST_SHM_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}_{}", prefix, std::process::id(), seq)
    }
}
