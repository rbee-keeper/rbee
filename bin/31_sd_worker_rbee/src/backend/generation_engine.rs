// Created by: TEAM-393
// TEAM-393: Asynchronous generation engine

use crate::backend::{
    inference::InferencePipeline,
    request_queue::{GenerationRequest, GenerationResponse, RequestQueue},
    image_utils::image_to_base64,
};
use crate::error::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Asynchronous generation engine
pub struct GenerationEngine {
    queue: Arc<RequestQueue>,
    shutdown: Arc<AtomicBool>,
}

impl GenerationEngine {
    /// Create a new generation engine
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            queue: Arc::new(RequestQueue::new(queue_capacity)),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the background generation task
    pub fn start(&mut self, pipeline: Arc<InferencePipeline>) {
        let mut rx = self.queue.take_receiver()
            .expect("Receiver already taken");
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            while \!shutdown.load(Ordering::Relaxed) {
                tokio::select\! {
                    Some((request, response_tx)) = rx.recv() => {
                        Self::process_request(
                            Arc::clone(&pipeline),
                            request,
                            response_tx,
                        ).await;
                    }
                    else => break,
                }
            }
        });
    }

    /// Submit a generation request
    pub async fn submit(
        &self,
        request: GenerationRequest,
        response_tx: mpsc::Sender<GenerationResponse>,
    ) -> Result<()> {
        self.queue
            .submit(request, response_tx)
            .await
            .map_err(|e| crate::error::Error::Generation(e))
    }

    /// Shutdown the engine
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Process a single generation request
    async fn process_request(
        pipeline: Arc<InferencePipeline>,
        request: GenerationRequest,
        response_tx: mpsc::Sender<GenerationResponse>,
    ) {
        // Progress callback
        let progress_tx = response_tx.clone();
        let progress_callback = move |step: usize, total: usize| {
            let _ = progress_tx.try_send(GenerationResponse::Progress { step, total });
        };

        // Generate image
        match pipeline.text_to_image(&request.config, progress_callback) {
            Ok(image) => {
                match image_to_base64(&image) {
                    Ok(base64) => {
                        // Send complete response with base64 image
                        let _ = response_tx.try_send(GenerationResponse::Complete {
                            image: image,
                        });
                    }
                    Err(e) => {
                        let _ = response_tx.try_send(GenerationResponse::Error {
                            message: format\!("Failed to encode image: {}", e),
                        });
                    }
                }
            }
            Err(e) => {
                let _ = response_tx.try_send(GenerationResponse::Error {
                    message: format\!("Generation failed: {}", e),
                });
            }
        }
    }

    /// Get queue sender for submitting requests
    pub fn queue_sender(&self) -> mpsc::Sender<(GenerationRequest, mpsc::Sender<GenerationResponse>)> {
        self.queue.sender()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::sampling::SamplingConfig;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = GenerationEngine::new(10);
        assert\!(\!engine.shutdown.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_engine_shutdown() {
        let engine = GenerationEngine::new(10);
        engine.shutdown().await;
        assert\!(engine.shutdown.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_submit_request() {
        let engine = GenerationEngine::new(10);
        let (response_tx, _response_rx) = mpsc::channel(10);
        
        let request = GenerationRequest {
            job_id: "test-123".to_string(),
            config: SamplingConfig {
                prompt: "test prompt".to_string(),
                ..Default::default()
            },
        };

        // Should succeed (even though no worker is processing)
        assert\!(engine.submit(request, response_tx).await.is_ok());
    }
}
