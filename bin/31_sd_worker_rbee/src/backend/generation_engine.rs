// TEAM-488: Generation engine - processes request queue
// TEAM-481: Now uses Box<dyn ImageModel> instead of generic (object-safe trait)
//
// Receives GenerationRequest from queue, calls ImageModel trait methods,
// sends responses back via channels.

use crate::backend::request_queue::{GenerationRequest, GenerationResponse};
use crate::backend::traits::ImageModel;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Generation engine that processes requests from the queue
///
/// TEAM-488: Clean architecture
/// TEAM-481: Uses trait object (Box<dyn ImageModel>) for true polymorphism
/// - Receives requests from `RequestQueue`
/// - Calls `ImageModel` trait methods (works with SD or FLUX)
/// - Sends responses via channels
/// - Runs in `spawn_blocking` (CPU-intensive work)
pub struct GenerationEngine {
    model: Arc<Mutex<Box<dyn ImageModel>>>,
    request_rx: tokio::sync::mpsc::UnboundedReceiver<GenerationRequest>,
}

impl GenerationEngine {
    /// Create new generation engine
    ///
    /// # Arguments
    /// * `model` - Boxed `ImageModel` trait object (SD or FLUX)
    /// * `request_rx` - Receiver for generation requests
    pub fn new(
        model: Arc<Mutex<Box<dyn ImageModel>>>,
        request_rx: tokio::sync::mpsc::UnboundedReceiver<GenerationRequest>,
    ) -> Self {
        Self { model, request_rx }
    }

    /// Start the generation engine (consumes self)
    ///
    /// Spawns a tokio task that processes requests from the queue.
    /// Each generation runs in `spawn_blocking` (CPU/GPU intensive).
    pub fn start(mut self) {
        tokio::spawn(async move {
            tracing::info!("🚀 Generation engine started");

            while let Some(request) = self.request_rx.recv().await {
                let model = Arc::clone(&self.model);

                // Spawn blocking task for CPU/GPU intensive work
                tokio::task::spawn_blocking(move || {
                    Self::process_request(&model, &request);
                });
            }

            tracing::warn!("⚠️  Generation engine stopped (queue closed)");
        });
    }

    /// Process a single generation request
    ///
    /// TEAM-482: Takes parameters by reference to avoid needless_pass_by_value
    fn process_request(model: &Arc<Mutex<Box<dyn ImageModel>>>, request: &GenerationRequest) {
        let request_id = request.request_id.clone();
        let response_tx = request.response_tx.clone();

        tracing::info!("Processing request: {}", request_id);

        // Convert to trait request
        let trait_request = crate::backend::traits::GenerationRequest {
            request_id: request_id.clone(),
            prompt: request.config.prompt.clone(),
            negative_prompt: request.config.negative_prompt.clone(),
            width: request.config.width,
            height: request.config.height,
            steps: request.config.steps,
            guidance_scale: request.config.guidance_scale,
            seed: request.config.seed,
            input_image: request.input_image.clone(),
            mask: request.mask.clone(),
            strength: request.strength,
        };

        // Progress callback - TEAM-481: Box the closure for object safety
        let progress_tx = response_tx.clone();
        let progress_callback: Box<dyn FnMut(usize, usize, Option<image::DynamicImage>) + Send> =
            Box::new(move |step: usize, total: usize, preview: Option<image::DynamicImage>| {
                let response = if let Some(img) = preview {
                    GenerationResponse::Preview { step, total, image: img }
                } else {
                    GenerationResponse::Progress { step, total }
                };

                let _ = progress_tx.send(response);
            });

        // Generate image
        let result = {
            let mut model_guard = model.blocking_lock();
            model_guard.generate(&trait_request, progress_callback)
        };

        // Send final response
        match result {
            Ok(image) => {
                tracing::info!("✅ Generation complete: {}", request_id);
                let _ = response_tx.send(GenerationResponse::Complete { image });
            }
            Err(e) => {
                tracing::error!("❌ Generation failed: {} - {}", request_id, e);
                let _ = response_tx.send(GenerationResponse::Error { message: e.to_string() });
            }
        }
    }
}
