// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local registry for **typed** embedding engines.
//!
//! Unlike the generic [`LocalEndpointRegistry`] in `dynamo_runtime`, which
//! stores engines working with untyped `serde_json::Value` payloads, this
//! registry stores engines specialised to the embedding request/response
//! types (`PreprocessedEmbeddingRequest` / `EmbeddingsEngineOutput`).
//!
//! The benefit is that the embedding pipeline can dispatch directly through
//! a typed engine without round-tripping requests and responses through
//! `serde_json::Value`, which is the dominant overhead for large-batch
//! embedding responses. See `LocalEngineAdapter` for the JSON-based path
//! that this registry supersedes (when available).
//!
//! Registration is typically performed from a Python binding that wraps a
//! Python async generator in a direct-extraction engine, then calls
//! [`register`].  Consumers (e.g. the embedding branch of `ModelWatcher`)
//! call [`get`] to resolve an endpoint.

use std::sync::{Arc, OnceLock};

use anyhow::Error;
use dashmap::DashMap;

use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{ManyOut, SingleIn},
    protocols::annotated::Annotated,
};

use crate::protocols::common::{
    llm_backend::EmbeddingsEngineOutput, preprocessor::PreprocessedEmbeddingRequest,
};

/// Boxed async engine with the specific embedding request/response types.
pub type TypedEmbeddingEngine = Arc<
    dyn AsyncEngine<
            SingleIn<PreprocessedEmbeddingRequest>,
            ManyOut<Annotated<EmbeddingsEngineOutput>>,
            Error,
        > + Send
        + Sync,
>;

static REGISTRY: OnceLock<DashMap<String, TypedEmbeddingEngine>> = OnceLock::new();

fn registry() -> &'static DashMap<String, TypedEmbeddingEngine> {
    REGISTRY.get_or_init(DashMap::new)
}

/// Register a typed embedding engine under `endpoint_name`.
///
/// If an entry already exists for that name it is replaced.  The registry
/// key should match the short endpoint name (e.g. `"generate"`) so consumer
/// lookups line up with the convention used by
/// `LocalEndpointRegistry::register_local_engine`.
pub fn register(endpoint_name: String, engine: TypedEmbeddingEngine) {
    tracing::debug!(
        endpoint = %endpoint_name,
        "Registering typed embedding engine in local registry"
    );
    registry().insert(endpoint_name, engine);
}

/// Look up a typed embedding engine by short endpoint name.
pub fn get(endpoint_name: &str) -> Option<TypedEmbeddingEngine> {
    registry().get(endpoint_name).map(|v| v.clone())
}
