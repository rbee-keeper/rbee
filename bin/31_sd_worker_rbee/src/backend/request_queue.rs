// Created by: TEAM-393
// TEAM-393: Request queue for async generation

use crate::backend::sampling::SamplingConfig;
use image::DynamicImage;
use tokio::sync::mpsc;

/// Request for image generation
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub job_id: String,
    pub config: SamplingConfig,
}

/// Response events from generation
#[derive(Debug, Clone)]
pub enum GenerationResponse {
    /// Progress update during generation
    Progress { step: usize, total: usize },
    /// Generation complete with image
    Complete { image: DynamicImage },
    /// Generation failed
    Error { message: String },
}

/// Request queue for managing generation requests
pub struct RequestQueue {
    tx: mpsc::Sender<QueueItem>,
    rx: Option<mpsc::Receiver<QueueItem>>,
}

type QueueItem = (GenerationRequest, mpsc::Sender<GenerationResponse>);

impl RequestQueue {
    /// Create a new request queue with given capacity
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self {
            tx,
            rx: Some(rx),
        }
    }

    /// Submit a generation request
    pub async fn submit(
        &self,
        request: GenerationRequest,
        response_tx: mpsc::Sender<GenerationResponse>,
    ) -> Result<(), String> {
        self.tx
            .send((request, response_tx))
            .await
            .map_err(|e| format\!("Failed to submit request: {}", e))
    }

    /// Take the receiver (can only be called once)
    pub fn take_receiver(&mut self) -> Option<mpsc::Receiver<QueueItem>> {
        self.rx.take()
    }

    /// Get a sender clone for submitting requests
    pub fn sender(&self) -> mpsc::Sender<QueueItem> {
        self.tx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_queue_submit() {
        let mut queue = RequestQueue::new(10);
        let (response_tx, _response_rx) = mpsc::channel(10);
        
        let request = GenerationRequest {
            job_id: "test-123".to_string(),
            config: SamplingConfig {
                prompt: "test".to_string(),
                ..Default::default()
            },
        };

        assert\!(queue.submit(request, response_tx).await.is_ok());
    }

    #[tokio::test]
    async fn test_queue_receive() {
        let mut queue = RequestQueue::new(10);
        let (response_tx, _response_rx) = mpsc::channel(10);
        
        let request = GenerationRequest {
            job_id: "test-123".to_string(),
            config: SamplingConfig {
                prompt: "test".to_string(),
                ..Default::default()
            },
        };

        queue.submit(request.clone(), response_tx).await.unwrap();
        
        let mut rx = queue.take_receiver().unwrap();
        let (received_req, _) = rx.recv().await.unwrap();
        assert_eq\!(received_req.job_id, "test-123");
    }
}
