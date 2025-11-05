// Created by: TEAM-393
// TEAM-396: CRITICAL FIX - Rewritten to match LLM worker pattern
//
// FIXED ISSUES:
// 1. Dependency injection (rx passed in, not created internally)
// 2. spawn_blocking instead of tokio::spawn (CPU-intensive work)
// 3. start() consumes self (clean ownership)
// 4. Removed Arc<RequestQueue> complexity
// 5. Removed shutdown AtomicBool (not needed)
// 6. Matches bin/30_llm_worker_rbee/src/backend/generation_engine.rs

use crate::backend::{
    generation,  // TEAM-397: Direct Candle functions
    models::ModelComponents,  // TEAM-397: Direct Candle types
    request_queue::{GenerationRequest, GenerationResponse},
};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Generation engine that processes inference requests
///
/// TEAM-396: Fixed to match LLM worker pattern
/// TEAM-397: RULE ZERO - Uses direct Candle functions, not InferencePipeline wrapper
/// This runs in `spawn_blocking` to avoid blocking the async runtime.
/// It pulls requests from the queue and generates images one by one.
pub struct GenerationEngine {
    models: Arc<ModelComponents>,  // TEAM-397: Direct Candle types
    request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
}

impl GenerationEngine {
    /// Create a new generation engine
    ///
    /// # Arguments
    /// * `models` - The loaded model components (direct Candle types)
    /// * `request_rx` - Receiver for generation requests from HTTP handlers
    ///
    /// TEAM-396: Takes rx as parameter (dependency injection)
    /// TEAM-397: Takes ModelComponents instead of InferencePipeline
    pub fn new(
        models: Arc<ModelComponents>,
        request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
    ) -> Self {
        Self { models, request_rx }
    }

    /// Start the generation engine loop
    ///
    /// This spawns a blocking task that processes requests sequentially.
    /// The task runs until the request channel is closed.
    ///
    /// CRITICAL: Uses `spawn_blocking` to move CPU-intensive work
    /// off the async runtime, preventing it from blocking HTTP handlers.
    ///
    /// TEAM-396: Consumes self (clean ownership, matches LLM worker)
    pub fn start(mut self) {
        tokio::task::spawn_blocking(move || {
            // Get tokio runtime handle for async operations within blocking context
            let rt = tokio::runtime::Handle::current();

            tracing::info!("Generation engine started");

            loop {
                // Wait for next request (blocking is OK here, we're in spawn_blocking)
                let request = if let Some(req) = rt.block_on(self.request_rx.recv()) {
                    req
                } else {
                    tracing::info!("Request channel closed, stopping generation engine");
                    break;
                };

                tracing::info!(
                    request_id = %request.request_id,
                    prompt = %request.config.prompt,
                    steps = request.config.steps,
                    "Processing generation request"
                );

                // TEAM-397: Call Candle generation function directly
                // No lock needed - ModelComponents is immutable
                Self::generate_and_send(
                    &self.models,
                    &request.config,
                    request.response_tx,
                );

                tracing::debug!(
                    request_id = %request.request_id,
                    "Request completed"
                );
            }

            tracing::info!("Generation engine stopped");
        });
    }

    /// Generate image and send responses through channel
    ///
    /// TEAM-396: Simplified - response_tx is in the request
    /// TEAM-397: RULE ZERO - Calls generation::generate_image() directly
    fn generate_and_send(
        models: &ModelComponents,
        config: &crate::backend::sampling::SamplingConfig,
        response_tx: mpsc::UnboundedSender<GenerationResponse>,
    ) {
        // Progress callback
        let progress_tx = response_tx.clone();
        let progress_callback = move |step: usize, total: usize| {
            let _ = progress_tx.send(GenerationResponse::Progress { step, total });
        };

        // TEAM-397: Call Candle generation function (not struct method)
        match generation::generate_image(config, models, progress_callback) {
            Ok(image) => {
                // Send complete response with image
                let _ = response_tx.send(GenerationResponse::Complete { image });
            }
            Err(e) => {
                tracing::error!(error = %e, "Generation failed");
                let _ = response_tx.send(GenerationResponse::Error {
                    message: e.to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::sampling::SamplingConfig;

    #[test]
    fn test_engine_creation() {
        // Create mock pipeline and channel
        let (_tx, rx) = mpsc::unbounded_channel();
        let pipeline = Arc::new(Mutex::new(
            // Would need actual pipeline here, but this tests structure
            unsafe { std::mem::zeroed() }
        ));
        
        let _engine = GenerationEngine::new(pipeline, rx);
        // Engine created successfully
    }
}
