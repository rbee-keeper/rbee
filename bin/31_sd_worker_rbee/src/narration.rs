// TEAM-XXX: Narration utilities for SD worker
//
// Provides structured logging and progress reporting.

use observability_narration_core::n;

/// Log device initialization
pub fn log_device_init(device_name: &str) {
    n!("device_init", "Initializing device: {}", device_name);
}

/// Log model loading start
pub fn log_model_loading_start(model_version: &str) {
    n!("model_loading_start", "Loading Stable Diffusion model: {}", model_version);
}

/// Log model loading complete
pub fn log_model_loading_complete(model_version: &str, elapsed_ms: u64) {
    n!("model_loading_complete", "Model {} loaded in {}ms", model_version, elapsed_ms);
}

/// Log generation start
pub fn log_generation_start(prompt: &str, steps: usize) {
    n!("generation_start", "Starting generation: '{}' ({} steps)", prompt, steps);
}

/// Log generation progress
pub fn log_generation_progress(step: usize, total_steps: usize) {
    n!("generation_progress", "Step {}/{}", step, total_steps);
}

/// Log generation complete
pub fn log_generation_complete(elapsed_ms: u64) {
    n!("generation_complete", "Generation complete in {}ms", elapsed_ms);
}

/// Log error
pub fn log_error(context: &str, error: &str) {
    n!("error", "{}: {}", context, error);
}
