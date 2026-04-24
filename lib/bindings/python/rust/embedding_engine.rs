// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed embedding engine — direct PyO3 extraction (no `serde_json::Value`
//! round trip) from a Python async generator handler.
//!
//! This is the fast in-process path used when the embedding worker and the
//! HTTP frontend live in the same Python/Rust process.  Compared to the
//! generic `PythonServerStreamingEngine` + `LocalEngineAdapter` path, which
//! goes through `pythonize`/`depythonize` (serde-based), this engine:
//!
//! * Builds the Python request dict directly from the typed
//!   [`PreprocessedEmbeddingRequest`] with PyO3 primitives — no intermediate
//!   `serde_json::Value`.
//! * Extracts `embeddings: Vec<Vec<f64>>`, `prompt_tokens: u32`,
//!   `total_tokens: u32` from the yielded Python dict via direct
//!   `FromPyObject` impls — no walk through a serde `Value` tree.
//!
//! For BS=16 embedding responses (~32k f64s), this avoids ~15–20 ms of
//! ser/de overhead per request.

use std::sync::Arc;

use anyhow::{Error, Result, anyhow};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_async_runtimes::TaskLocals;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream, async_trait},
    pipeline::{ManyOut, SingleIn},
    protocols::annotated::Annotated,
};

use dynamo_llm::protocols::common::{
    llm_backend::EmbeddingsEngineOutput, preprocessor::PreprocessedEmbeddingRequest,
};

use crate::context::{Context, callable_accepts_kwarg};

/// Direct-extraction async engine for embedding endpoints.
///
/// Wraps a Python async generator (the user's embedding handler) and
/// implements the typed `AsyncEngine` interface expected by the embedding
/// pipeline in `ModelWatcher`.
#[derive(Clone)]
pub struct PythonEmbeddingEngine {
    _cancel_token: CancellationToken,
    generator: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    has_context: bool,
}

impl PythonEmbeddingEngine {
    pub fn new(generator: Arc<PyObject>, event_loop: Arc<PyObject>) -> Self {
        let has_context = Python::with_gil(|py| {
            let callable = generator.bind(py);
            callable_accepts_kwarg(py, callable, "context").unwrap_or(false)
        });

        Self {
            _cancel_token: CancellationToken::new(),
            generator,
            event_loop,
            has_context,
        }
    }
}

/// Build a Python dict mirroring the JSON representation of
/// [`PreprocessedEmbeddingRequest`] that the existing embedding handler
/// expects: `{ "token_ids": [[int, ...], ...], "model": str, ... }`.
///
/// This uses only PyO3 primitives and therefore avoids the `pythonize`
/// serde round trip.
fn build_request_dict<'py>(
    py: Python<'py>,
    req: &PreprocessedEmbeddingRequest,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);

    let token_ids_outer = PyList::empty(py);
    for inner in &req.token_ids {
        // PyList::new allows a cheap construction from a slice of primitives
        // that implement IntoPyObject; u32 qualifies directly.
        let inner_list = PyList::new(py, inner.iter().copied())?;
        token_ids_outer.append(inner_list)?;
    }
    d.set_item("token_ids", token_ids_outer)?;
    d.set_item("model", &req.model)?;
    if let Some(enc) = &req.encoding_format {
        d.set_item("encoding_format", enc)?;
    } else {
        d.set_item("encoding_format", py.None())?;
    }
    if let Some(dim) = req.dimensions {
        d.set_item("dimensions", dim)?;
    } else {
        d.set_item("dimensions", py.None())?;
    }
    if let Some(mdc) = &req.mdc_sum {
        d.set_item("mdc_sum", mdc)?;
    } else {
        d.set_item("mdc_sum", py.None())?;
    }
    let anns = PyList::new(py, req.annotations.iter().map(String::as_str))?;
    d.set_item("annotations", anns)?;

    Ok(d)
}

/// Extract an [`EmbeddingsEngineOutput`] from a Python object that the
/// handler yielded — typically a plain dict with keys `embeddings`,
/// `prompt_tokens`, and `total_tokens`.
///
/// Implementation detail: for the heavy `embeddings` field we use PyO3's
/// optimised `FromPyObject` impl for `Vec<Vec<f64>>`, which iterates the
/// outer sequence once, then for each inner sequence iterates and extracts
/// `f64` values via `PyFloat::value` (a C-level call).  This is materially
/// faster than `depythonize` which routes every value through a serde
/// `Value` tree.
fn extract_embedding_output(obj: &Bound<'_, PyAny>) -> Result<EmbeddingsEngineOutput> {
    // Permit both "data" as a synonym for "embeddings" in case a custom
    // handler matches the OpenAI response shape more closely, but the
    // canonical key used by `embedding_handler.py` is `embeddings`.
    let embeddings_any = obj
        .get_item("embeddings")
        .or_else(|_| obj.get_item("data"))
        .map_err(|e| anyhow!("handler response missing 'embeddings' key: {e}"))?;

    let embeddings: Vec<Vec<f64>> = embeddings_any
        .extract()
        .map_err(|e| anyhow!("failed to extract embeddings as Vec<Vec<f64>>: {e}"))?;

    let prompt_tokens: u32 = obj
        .get_item("prompt_tokens")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(0);
    let total_tokens: u32 = obj
        .get_item("total_tokens")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(prompt_tokens);

    Ok(EmbeddingsEngineOutput {
        embeddings,
        prompt_tokens,
        total_tokens,
    })
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
        Error,
    > for PythonEmbeddingEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<EmbeddingsEngineOutput>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let (tx, rx) = mpsc::channel::<Annotated<EmbeddingsEngineOutput>>(8);

        let generator = self.generator.clone();
        let event_loop = self.event_loop.clone();
        let ctx_python = ctx.clone();
        let has_context = self.has_context;

        // Acquire the GIL on a blocking thread (same policy as
        // PythonServerStreamingEngine) to avoid stalling the tokio runtime
        // under contention.
        let stream = tokio::task::spawn_blocking(move || -> PyResult<_> {
            Python::with_gil(|py| {
                let py_request = build_request_dict(py, &request)?;

                let gen_result = if has_context {
                    let py_ctx = Py::new(py, Context::new(ctx_python.clone(), None))?;
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("context", &py_ctx)?;
                    generator.call(py, (py_request,), Some(&kwargs))
                } else {
                    generator.call1(py, (py_request,))
                }?;

                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::tokio::into_stream_with_locals_v1(
                    locals,
                    gen_result.into_bound(py),
                )
            })
        })
        .await
        .map_err(|e| anyhow!("failed to spawn blocking gil task: {e}"))??;

        tokio::spawn(async move {
            let mut stream = Box::pin(stream);
            while let Some(item) = stream.next().await {
                let response: Annotated<EmbeddingsEngineOutput> = match item {
                    Ok(py_obj) => {
                        let extracted = tokio::task::spawn_blocking(move || {
                            Python::with_gil(|py| {
                                let bound = py_obj.into_bound(py);
                                extract_embedding_output(&bound)
                            })
                        })
                        .await;
                        match extracted {
                            Ok(Ok(out)) => Annotated::from_data(out),
                            Ok(Err(e)) => Annotated::from_error(format!(
                                "typed embedding extract failed: {e}"
                            )),
                            Err(e) => Annotated::from_error(format!(
                                "typed embedding gil offload failed: {e}"
                            )),
                        }
                    }
                    Err(py_err) => {
                        let msg = Python::with_gil(|py| {
                            py_err.display(py);
                            py_err.to_string()
                        });
                        Annotated::from_error(format!(
                            "python exception in embedding handler: {msg}"
                        ))
                    }
                };

                if tx.send(response).await.is_err() {
                    break;
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}
