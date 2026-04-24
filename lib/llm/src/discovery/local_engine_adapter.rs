// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapter that wraps a `LocalAsyncEngine` (JSON-based, in-process) so it can
//! be used as a typed `ServiceEngine` in embedding pipelines.
//!
//! The `LocalAsyncEngine` works with `serde_json::Value` payloads while the
//! embedding pipeline expects strongly-typed `PreprocessedEmbeddingRequest` /
//! `EmbeddingsEngineOutput`. This adapter bridges the two via ser/de.

use anyhow::Error;
use futures::StreamExt;
use serde::{Serialize, de::DeserializeOwned};

use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream, async_trait},
    local_endpoint_registry::LocalAsyncEngine,
    pipeline::{ManyOut, SingleIn},
    protocols::annotated::Annotated,
};

/// Wraps a [`LocalAsyncEngine`] to transparently convert between typed
/// request/response types and the `serde_json::Value` representation used by
/// the local endpoint registry.
pub struct LocalEngineAdapter<Req, Resp> {
    engine: LocalAsyncEngine,
    _phantom: std::marker::PhantomData<(Req, Resp)>,
}

impl<Req, Resp> LocalEngineAdapter<Req, Resp> {
    pub fn new(engine: LocalAsyncEngine) -> Self {
        Self {
            engine,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error>
    for LocalEngineAdapter<Req, Resp>
where
    Req: Serialize + Send + Sync + 'static,
    Resp: DeserializeOwned + Send + Sync + 'static,
{
    async fn generate(
        &self,
        request: SingleIn<Req>,
    ) -> Result<ManyOut<Annotated<Resp>>, Error> {
        let json_request = request.try_map(|req| serde_json::to_value(req))?;
        let json_stream = self.engine.generate(json_request).await?;

        let ctx = json_stream.context();
        let mapped = json_stream.map(|annotated: Annotated<serde_json::Value>| {
            annotated.map_data(|v| {
                serde_json::from_value::<Resp>(v)
                    .map_err(|e| format!("local engine adapter: response deser failed: {e}"))
            })
        });

        Ok(ResponseStream::new(Box::pin(mapped), ctx))
    }
}
