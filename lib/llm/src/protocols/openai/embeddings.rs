// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

mod aggregator;
mod nvext;
mod shm;

pub use nvext::{NvExt, NvExtProvider};
pub use shm::{
    expand_embedding_response_shm, maybe_write_embedding_request_shm,
    maybe_write_preprocessed_embedding_request_shm, metadata_to_embeddings,
    read_embedding_response_shm, register_embedding_shm_metrics,
};

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateEmbeddingRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateEmbeddingRequest,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,

    /// Internal frontend-to-worker SHM descriptor. Clients cannot set this
    /// field through HTTP deserialization; it is populated only after the
    /// request has entered Dynamo.
    #[serde(default, skip_serializing_if = "Option::is_none", skip_deserializing)]
    #[schema(ignore)]
    pub embedding_request_shm:
        Option<crate::protocols::common::llm_backend::EmbeddingRequestShmMetadata>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI unary chat completion response, embedded
///   using `serde(flatten)`.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateEmbeddingResponse {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateEmbeddingResponse,

    /// Internal worker-to-frontend SHM descriptor. This must be consumed and
    /// cleared before serializing the response to an HTTP client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(ignore)]
    pub embedding_response_shm:
        Option<crate::protocols::common::llm_backend::EmbeddingResponseShmMetadata>,
}

impl NvCreateEmbeddingResponse {
    pub fn empty() -> Self {
        Self {
            inner: dynamo_protocols::types::CreateEmbeddingResponse {
                object: "list".to_string(),
                model: "embedding".to_string(),
                data: vec![],
                usage: dynamo_protocols::types::EmbeddingUsage {
                    prompt_tokens: 0,
                    total_tokens: 0,
                },
            },
            embedding_response_shm: None,
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateEmbeddingRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateEmbeddingRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateEmbeddingRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateEmbeddingRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}
