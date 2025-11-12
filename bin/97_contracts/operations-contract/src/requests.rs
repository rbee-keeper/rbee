//! Request types for operations
//!
//! TEAM-284: Typed request structures for all operations
//!
//! These types provide compile-time guarantees that requests are well-formed.

use serde::{Deserialize, Serialize};

// ============================================================================
// Worker Operation Requests
// ============================================================================

/// Request to list available workers from catalog server
/// 
/// TEAM-388: Lists workers from Hono catalog (http://localhost:8787/workers)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerCatalogListRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
}

/// Request to get worker details from catalog server
/// 
/// TEAM-388: Gets worker details from Hono catalog (http://localhost:8787/workers/:id)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerCatalogGetRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
    /// Worker ID from catalog (e.g., "llm-worker-rbee-cpu")
    pub worker_id: String,
}

/// Request to install a worker binary from catalog
/// 
/// TEAM-388: Downloads PKGBUILD, builds, and installs worker
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerInstallRequest {
    /// Hive ID where worker should be installed
    pub hive_id: String,
    /// Worker ID from catalog (e.g., "llm-worker-rbee-cpu")
    pub worker_id: String,
}

/// Request to remove an installed worker binary
/// 
/// TEAM-388: Removes worker from ~/.cache/rbee/workers/
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerRemoveRequest {
    /// Hive ID where worker is installed
    pub hive_id: String,
    /// Worker ID to remove (e.g., "llm-worker-rbee-cpu")
    pub worker_id: String,
}

/// Request to list installed worker binaries on hive
/// 
/// TEAM-378: Lists workers from the worker catalog (installed binaries)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerListInstalledRequest {
    /// Hive ID to list installed workers from
    pub hive_id: String,
}

/// Request to spawn a worker process
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerSpawnRequest {
    /// Hive ID where worker should be spawned
    pub hive_id: String,
    /// Model to load
    pub model: String,
    /// Worker type (e.g., "cpu", "cuda", "metal")
    pub worker: String,
    /// Device index
    pub device: u32,
}

/// Request to list worker processes on a hive
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessListRequest {
    /// Hive ID to query
    pub hive_id: String,
}

/// Request to get worker process details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessGetRequest {
    /// Hive ID where worker is running
    pub hive_id: String,
    /// Process ID
    pub pid: u32,
}

/// Request to delete (kill) a worker process
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessDeleteRequest {
    /// Hive ID where worker is running
    pub hive_id: String,
    /// Process ID to kill
    pub pid: u32,
}

// ============================================================================
// Model Operation Requests
// ============================================================================

/// Request to download a model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDownloadRequest {
    /// Hive ID where model should be downloaded
    pub hive_id: String,
    /// Model identifier (e.g., "meta-llama/Llama-2-7b")
    pub model: String,
}

/// Request to list models on a hive
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelListRequest {
    /// Hive ID to query
    pub hive_id: String,
}

/// Request to get model details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelGetRequest {
    /// Hive ID where model is stored
    pub hive_id: String,
    /// Model ID
    pub id: String,
}

/// Request to delete a model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDeleteRequest {
    /// Hive ID where model is stored
    pub hive_id: String,
    /// Model ID to delete
    pub id: String,
}

/// Request to load a model into RAM
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelLoadRequest {
    /// Hive ID where model should be loaded
    pub hive_id: String,
    /// Model ID to load
    pub id: String,
    /// Device to load on (e.g., "cuda:0", "cpu")
    pub device: String,
}

/// Request to unload a model from RAM
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelUnloadRequest {
    /// Hive ID where model is loaded
    pub hive_id: String,
    /// Model ID to unload
    pub id: String,
}

// ============================================================================
// Inference Request
// ============================================================================

/// Request to perform inference
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
    /// Model to use
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Top-p sampling (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Device to use (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Specific worker ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
    /// Stream response
    #[serde(default = "default_stream")]
    pub stream: bool,
}

fn default_stream() -> bool {
    true
}

// ============================================================================
// Image Generation Requests (TEAM-397)
// ============================================================================

/// LoRA configuration for image generation
/// TEAM-488: Wire up LoRA support (TEAM-487 left it incomplete)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoRAConfig {
    /// Path to LoRA .safetensors file
    pub path: String,
    /// Strength multiplier (0.0-1.0, default: 1.0)
    #[serde(default = "default_lora_strength")]
    pub strength: f32,
}

fn default_lora_strength() -> f32 {
    1.0
}

/// Request to generate image from text prompt (Stable Diffusion)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageGenerationRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
    /// Model to use (e.g., "stable-diffusion-v1-5")
    pub model: String,
    /// Text prompt
    pub prompt: String,
    /// Negative prompt (optional)
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Number of inference steps (default: 20)
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Guidance scale (default: 7.5)
    #[serde(default = "default_guidance")]
    pub guidance_scale: f64,
    /// Random seed (optional)
    #[serde(default)]
    pub seed: Option<u64>,
    /// Image width (default: 512)
    #[serde(default = "default_width")]
    pub width: usize,
    /// Image height (default: 512)
    #[serde(default = "default_height")]
    pub height: usize,
    /// LoRA configurations (TEAM-488: Finally wired up!)
    #[serde(default)]
    pub loras: Vec<LoRAConfig>,
    /// Specific worker ID (optional, for direct routing)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
}

fn default_steps() -> usize { 20 }
fn default_guidance() -> f64 { 7.5 }
fn default_width() -> usize { 512 }
fn default_height() -> usize { 512 }

/// Request to transform image (img2img)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageTransformRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
    /// Model to use
    pub model: String,
    /// Text prompt
    pub prompt: String,
    /// Negative prompt (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Number of inference steps
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Guidance scale
    #[serde(default = "default_guidance")]
    pub guidance_scale: f64,
    /// Random seed (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Transformation strength (0.0-1.0)
    #[serde(default = "default_strength")]
    pub strength: f64,
    /// Input image (base64 encoded)
    pub input_image: String,
    /// LoRA configurations (TEAM-488: Finally wired up!)
    #[serde(default)]
    pub loras: Vec<LoRAConfig>,
    /// Specific worker ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
}

fn default_strength() -> f64 { 0.8 }

/// Request to inpaint image
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageInpaintRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
    /// Model to use
    pub model: String,
    /// Text prompt
    pub prompt: String,
    /// Negative prompt (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Base64-encoded input image
    pub init_image: String,
    /// Base64-encoded mask (white = inpaint, black = keep)
    pub mask_image: String,
    /// Number of inference steps
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Guidance scale
    #[serde(default = "default_guidance")]
    pub guidance_scale: f64,
    /// Random seed (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// LoRA configurations (TEAM-488: Finally wired up!)
    #[serde(default)]
    pub loras: Vec<LoRAConfig>,
    /// Specific worker ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_spawn_request_serialization() {
        let request = WorkerSpawnRequest {
            hive_id: "localhost".to_string(),
            model: "test-model".to_string(),
            worker: "cpu".to_string(),
            device: 0,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: WorkerSpawnRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request, deserialized);
    }

    #[test]
    fn test_infer_request_optional_fields() {
        let request = InferRequest {
            hive_id: "localhost".to_string(),
            model: "test-model".to_string(),
            prompt: "hello".to_string(),
            max_tokens: 20,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
            device: None,
            worker_id: None,
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();

        // top_p should be present
        assert!(json.contains("\"top_p\":0.9"));

        // top_k should be omitted
        assert!(!json.contains("\"top_k\""));
    }

    // TEAM-397/398: Tests for image generation requests
    #[test]
    fn test_image_generation_request_serialization() {
        let request = ImageGenerationRequest {
            hive_id: "localhost".to_string(),
            model: "stable-diffusion-v1-5".to_string(),
            prompt: "a beautiful sunset".to_string(),
            negative_prompt: Some("ugly, blurry".to_string()),
            steps: 20,
            guidance_scale: 7.5,
            width: 512,
            height: 512,
            seed: Some(42),
            loras: vec![],  // TEAM-488: LoRA support
            worker_id: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ImageGenerationRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request, deserialized);
        assert!(json.contains("\"prompt\":\"a beautiful sunset\""));
        assert!(json.contains("\"negative_prompt\":\"ugly, blurry\""));
    }

    #[test]
    fn test_image_generation_request_defaults() {
        let json = r#"{
            "hive_id": "localhost",
            "model": "stable-diffusion-v1-5",
            "prompt": "test"
        }"#;

        let request: ImageGenerationRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.steps, 20); // default
        assert_eq!(request.guidance_scale, 7.5); // default
        assert_eq!(request.width, 512); // default
        assert_eq!(request.height, 512); // default
    }

    #[test]
    fn test_image_transform_request_serialization() {
        let request = ImageTransformRequest {
            hive_id: "localhost".to_string(),
            model: "stable-diffusion-v1-5".to_string(),
            prompt: "make it artistic".to_string(),
            negative_prompt: None,
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
            strength: 0.8,
            input_image: "base64encodedimage".to_string(),
            loras: vec![],  // TEAM-488: LoRA support
            worker_id: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ImageTransformRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request, deserialized);
        assert!(json.contains("\"strength\":0.8"));
    }

    #[test]
    fn test_image_inpaint_request_serialization() {
        let request = ImageInpaintRequest {
            hive_id: "localhost".to_string(),
            model: "stable-diffusion-v1-5".to_string(),
            prompt: "a red car".to_string(),
            negative_prompt: None,
            init_image: "base64encodedimage".to_string(),
            mask_image: "base64encodedmask".to_string(),
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
            loras: vec![],  // TEAM-488: LoRA support
            worker_id: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ImageInpaintRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request, deserialized);
        assert!(json.contains("\"mask_image\":\"base64encodedmask\""));
    }
}
