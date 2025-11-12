// Created by: TEAM-393
// TEAM-396: CRITICAL FIX - Rewritten to match LLM worker pattern
//
// FIXED ISSUES:
// 1. Removed Mutex anti-pattern (RequestQueue no longer owns receiver)
// 2. response_tx now IN GenerationRequest (cohesive structure)
// 3. Unbounded channels (simpler, no arbitrary capacity limits)
// 4. Clean ownership transfer (queue returns receiver)
// 5. Matches bin/30_llm_worker_rbee/src/backend/request_queue.rs

use crate::backend::sampling::SamplingConfig;
use image::DynamicImage;
use tokio::sync::mpsc;

/// Request for image generation
///
/// TEAM-396: `response_tx` is now PART of the request (not passed separately)
/// TEAM-487: Added `input_image`, mask, and strength for img2img and inpainting
#[derive(Debug)]
pub struct GenerationRequest {
    /// Unique request ID (`job_id` from HTTP request)
    pub request_id: String,

    /// Sampling configuration
    pub config: SamplingConfig,

    /// Optional input image for img2img/inpainting (None = text-to-image)
    /// TEAM-487: For image-to-image and inpainting generation
    pub input_image: Option<DynamicImage>,

    /// Optional mask for inpainting (white = inpaint, black = keep)
    /// TEAM-487: For inpainting generation
    /// If Some, this is inpainting. If None with `input_image`, this is img2img.
    pub mask: Option<DynamicImage>,

    /// Transformation strength for img2img (0.0-1.0)
    /// TEAM-487: Only used for img2img (not inpainting)
    /// Default: 0.8 (80% transformation)
    pub strength: f64,

    /// Channel to send responses back to HTTP handler
    /// TEAM-396: Moved from separate parameter to struct field
    pub response_tx: mpsc::UnboundedSender<GenerationResponse>,
}

/// Response events from generation
///
/// TEAM-487: Added Preview variant for intermediate images during generation
#[derive(Debug, Clone)]
pub enum GenerationResponse {
    /// Progress update during generation (step number only)
    Progress { step: usize, total: usize },
    /// Preview image during generation (intermediate result)
    /// TEAM-487: Sent every N steps to show generation progress
    Preview { step: usize, total: usize, image: DynamicImage },
    /// Generation complete with final image
    Complete { image: DynamicImage },
    /// Generation failed
    Error { message: String },
}

/// Request queue for adding generation requests
///
/// TEAM-396: Fixed to match LLM worker pattern
/// - Queue only holds SENDER (not receiver)
/// - Receiver returned from `new()` and given to `GenerationEngine`
/// - No Mutex needed (sender is Clone + Send)
/// - Unbounded channels (no capacity limits)
#[derive(Clone)]
pub struct RequestQueue {
    tx: mpsc::UnboundedSender<GenerationRequest>,
}

impl RequestQueue {
    /// Create a new request queue
    ///
    /// Returns the queue (for HTTP handlers) and receiver (for generation engine)
    /// TEAM-396: Clean separation - caller decides who gets what
    #[must_use] 
    pub fn new() -> (Self, mpsc::UnboundedReceiver<GenerationRequest>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Add a request to the queue
    ///
    /// TEAM-396: Now takes complete request (with `response_tx` inside)
    /// Returns Ok(()) if request was queued successfully.
    /// Returns Err if the generation engine has stopped.
    pub fn add_request(&self, request: GenerationRequest) -> Result<(), String> {
        self.tx
            .send(request)
            .map_err(|e| format!("Queue send failed (generation engine stopped): {e}"))
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_queue_submit() {
        let (queue, mut rx) = RequestQueue::new();
        let (response_tx, _response_rx) = mpsc::unbounded_channel();

        let request = GenerationRequest {
            request_id: "test-123".to_string(),
            config: SamplingConfig { prompt: "test".to_string(), ..Default::default() },
            input_image: None, // TEAM-487: Text-to-image
            mask: None,        // TEAM-487: No mask
            strength: 0.8,     // TEAM-487: Default strength
            response_tx,       // TEAM-396: Now part of request
        };

        assert!(queue.add_request(request).is_ok());

        // Verify we can receive it
        let received = rx.recv().await;
        assert!(received.is_some());
    }

    #[tokio::test]
    async fn test_queue_receive() {
        let (queue, mut rx) = RequestQueue::new();
        let (response_tx, _response_rx) = mpsc::unbounded_channel();

        let request = GenerationRequest {
            request_id: "test-123".to_string(),
            config: SamplingConfig { prompt: "test".to_string(), ..Default::default() },
            input_image: None, // TEAM-487: Text-to-image
            mask: None,        // TEAM-487: No mask
            strength: 0.8,     // TEAM-487: Default strength
            response_tx,       // TEAM-396: Now part of request
        };

        queue.add_request(request).unwrap();

        let received_req = rx.recv().await.unwrap();
        assert_eq!(received_req.request_id, "test-123");
    }
}
