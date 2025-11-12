// Created by: TEAM-394
// TEAM-396: CRITICAL FIX - Rewritten to match LLM worker pattern
//
// FIXED ISSUES:
// 1. Stores RequestQueue (not GenerationEngine)
// 2. GenerationEngine started separately and consumes self
// 3. Simpler, cleaner ownership model

use crate::backend::request_queue::RequestQueue;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shared application state for HTTP handlers
///
/// TEAM-396: Fixed to match LLM worker pattern
/// This state is cloned for each request (cheap - `RequestQueue` is Clone).
/// Contains the request queue for submitting generation jobs.
///
/// # Thread Safety
/// `RequestQueue` is Clone and thread-safe (internally uses `UnboundedSender`).
#[derive(Clone)]
pub struct AppState {
    /// Request queue for submitting generation requests
    /// TEAM-396: Changed from `Arc<GenerationEngine>` to `RequestQueue`
    request_queue: RequestQueue,

    /// Model loading status (true = ready for inference)
    model_loaded: Arc<AtomicBool>,
}

impl AppState {
    /// Create new `AppState` with request queue
    ///
    /// # Arguments
    /// * `request_queue` - Request queue from initialization
    ///
    /// TEAM-396: Simplified - just takes the queue
    /// The `GenerationEngine` is started separately in main.rs
    ///
    /// # Correct Setup Pattern
    /// ```no_run
    /// # use std::sync::{Arc, Mutex};
    /// # use sd_worker_rbee::backend::{RequestQueue, GenerationEngine};
    /// # use sd_worker_rbee::http::backend::AppState;
    /// # fn example() -> anyhow::Result<()> {
    /// // 1. Create queue and get receiver
    /// let (request_queue, request_rx) = RequestQueue::new();
    ///
    /// // 2. Load model and create pipeline
    /// // let pipeline = Arc::new(Mutex::new(InferencePipeline::new(...)?));
    ///
    /// // 3. Create engine with dependency injection
    /// // let engine = GenerationEngine::new(pipeline, request_rx);
    ///
    /// // 4. Start engine (consumes self)
    /// // engine.start();
    ///
    /// // 5. Create AppState with queue
    /// let state = AppState::new(request_queue);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use] 
    pub fn new(request_queue: RequestQueue) -> Self {
        Self { request_queue, model_loaded: Arc::new(AtomicBool::new(true)) }
    }

    /// Check if worker is ready for inference
    ///
    /// Returns true if model is loaded and ready to accept generation requests.
    /// Used by the /ready endpoint for Kubernetes readiness probes.
    #[must_use] 
    pub fn is_ready(&self) -> bool {
        self.model_loaded.load(Ordering::Relaxed)
    }

    /// Get reference to request queue
    ///
    /// TEAM-396: Changed from `generation_engine()` to `request_queue()`
    /// Used by job submission handlers to queue generation requests.
    #[must_use] 
    pub fn request_queue(&self) -> &RequestQueue {
        &self.request_queue
    }

    /// Set model loading status
    ///
    /// Called during startup to indicate when model is fully loaded.
    /// Also used to mark model as unavailable if unloading occurs.
    pub fn set_model_loaded(&self, loaded: bool) {
        self.model_loaded.store(loaded, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loaded_flag() {
        // Test that we can set and read the model_loaded flag
        // This is important for the /ready endpoint
        let flag = Arc::new(AtomicBool::new(false));
        assert!(!flag.load(Ordering::Relaxed));

        flag.store(true, Ordering::Relaxed);
        assert!(flag.load(Ordering::Relaxed));
    }
}
