// TEAM-487: Model test fixtures for verification
//
// Defines test configurations for all 7 SD model variants

use sd_worker_rbee::backend::models::SDVersion;
use std::path::PathBuf;

/// Model test configuration
#[derive(Debug, Clone)]
pub struct ModelFixture {
    pub version: SDVersion,
    pub repo: &'static str,
    pub expected_size: (usize, usize),
    pub expected_steps: usize,
    pub test_prompt: &'static str,
}

/// All model variants to test
pub const TEST_MODELS: &[ModelFixture] = &[
    ModelFixture {
        version: SDVersion::V1_5,
        repo: "runwayml/stable-diffusion-v1-5",
        expected_size: (512, 512),
        expected_steps: 20,
        test_prompt: "a photo of a cat",
    },
    ModelFixture {
        version: SDVersion::V1_5Inpaint,
        repo: "stable-diffusion-v1-5/stable-diffusion-inpainting",
        expected_size: (512, 512),
        expected_steps: 20,
        test_prompt: "a photo of a dog",
    },
    ModelFixture {
        version: SDVersion::V2_1,
        repo: "stabilityai/stable-diffusion-2-1",
        expected_size: (768, 768),
        expected_steps: 20,
        test_prompt: "a landscape painting",
    },
    ModelFixture {
        version: SDVersion::V2Inpaint,
        repo: "stabilityai/stable-diffusion-2-inpainting",
        expected_size: (768, 768),
        expected_steps: 20,
        test_prompt: "a mountain scene",
    },
    ModelFixture {
        version: SDVersion::XL,
        repo: "stabilityai/stable-diffusion-xl-base-1.0",
        expected_size: (1024, 1024),
        expected_steps: 20,
        test_prompt: "a futuristic city",
    },
    ModelFixture {
        version: SDVersion::XLInpaint,
        repo: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        expected_size: (1024, 1024),
        expected_steps: 20,
        test_prompt: "a beach scene",
    },
    ModelFixture {
        version: SDVersion::Turbo,
        repo: "stabilityai/sdxl-turbo",
        expected_size: (1024, 1024),
        expected_steps: 4, // Turbo uses 4 steps
        test_prompt: "a portrait photo",
    },
];

/// Get model path from environment or default cache location
///
/// TEAM-487: Tries environment variable first, then default cache
pub fn get_model_path(version: SDVersion) -> Option<PathBuf> {
    // Try environment variable first (e.g., SD_MODEL_V1_5)
    let env_var = format!("SD_MODEL_{:?}", version).to_uppercase();
    if let Ok(path) = std::env::var(&env_var) {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return Some(path_buf);
        }
    }

    // Try default cache location
    let home = std::env::var("HOME").ok()?;
    let cache_path = PathBuf::from(home).join(".cache/rbee/models").join(version.repo());

    if cache_path.exists() {
        Some(cache_path)
    } else {
        None
    }
}

/// Check if a model is available for testing
pub fn is_model_available(version: SDVersion) -> bool {
    get_model_path(version).is_some()
}

/// Get list of available models for testing
pub fn available_models() -> Vec<SDVersion> {
    TEST_MODELS
        .iter()
        .filter(|fixture| is_model_available(fixture.version))
        .map(|fixture| fixture.version)
        .collect()
}
