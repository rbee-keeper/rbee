// TEAM-405: Image model configuration
//! Image-specific model configuration (CivitAI/Stable Diffusion)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::{ModelType, config::{ModelConfig, InferenceParams}};

/// Image model configuration (CivitAI/Stable Diffusion)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageConfig {
    // ========== CRITICAL for inference ==========
    /// Base model (e.g., "SD 1.5", "SDXL", "SD 2.1")
    pub base_model: String,
    
    /// Model type (checkpoint, lora, embedding, etc.)
    pub checkpoint_type: CheckpointType,
    
    /// VAE to use
    #[serde(default)]
    pub vae: Option<String>,
    
    /// CLIP skip value
    #[serde(default)]
    pub clip_skip: Option<u32>,
    
    /// Trained resolution
    pub resolution: (u32, u32),
    
    /// Trigger words/activation tokens
    #[serde(default)]
    pub trigger_words: Vec<String>,
    
    // ========== Inference settings ==========
    /// Recommended samplers
    #[serde(default)]
    pub samplers: Vec<String>,
    
    /// Recommended steps
    #[serde(default)]
    pub steps: Option<u32>,
    
    /// Recommended CFG scale
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    
    // ========== Metadata ==========
    /// NSFW flag
    #[serde(default)]
    pub nsfw: bool,
    
    /// License
    #[serde(default)]
    pub license: Option<String>,
    
    /// Tags
    #[serde(default)]
    pub tags: Vec<String>,
    
    // ========== CivitAI specific ==========
    /// Preview images
    #[serde(default)]
    pub preview_images: Vec<ImagePreview>,
    
    /// Download/like stats
    #[serde(default)]
    pub downloads: u64,
    
    /// Like count
    #[serde(default)]
    pub likes: u64,
}

/// Checkpoint type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum CheckpointType {
    /// Full model checkpoint
    Checkpoint,
    /// LoRA (Low-Rank Adaptation)
    Lora,
    /// Textual Inversion embedding
    Embedding,
    /// Hypernetwork
    Hypernetwork,
    /// VAE (Variational Autoencoder)
    VAE,
}

/// Image preview
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagePreview {
    /// Image URL
    pub url: String,
    
    /// Width in pixels
    pub width: u32,
    
    /// Height in pixels
    pub height: u32,
    
    /// NSFW flag
    #[serde(default)]
    pub nsfw: bool,
}

impl ImageConfig {
    /// Create ImageConfig from CivitAI API response (future)
    pub fn from_civitai(civitai_data: &serde_json::Value) -> Self {
        // Extract base model
        let base_model = civitai_data["baseModel"]
            .as_str()
            .unwrap_or("SD 1.5")
            .to_string();
        
        // Determine checkpoint type
        let checkpoint_type = match civitai_data["type"].as_str() {
            Some("LORA") => CheckpointType::Lora,
            Some("TextualInversion") => CheckpointType::Embedding,
            Some("Hypernetwork") => CheckpointType::Hypernetwork,
            Some("VAE") => CheckpointType::VAE,
            _ => CheckpointType::Checkpoint,
        };
        
        // Extract resolution
        let resolution = (
            civitai_data["resolution"][0].as_u64().unwrap_or(512) as u32,
            civitai_data["resolution"][1].as_u64().unwrap_or(512) as u32,
        );
        
        // Extract trigger words
        let trigger_words = civitai_data["trainedWords"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        
        // Extract preview images
        let preview_images = civitai_data["images"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        Some(ImagePreview {
                            url: v["url"].as_str()?.to_string(),
                            width: v["width"].as_u64().unwrap_or(512) as u32,
                            height: v["height"].as_u64().unwrap_or(512) as u32,
                            nsfw: v["nsfw"].as_bool().unwrap_or(false),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        
        Self {
            base_model,
            checkpoint_type,
            vae: civitai_data["vae"].as_str().map(|s| s.to_string()),
            clip_skip: civitai_data["clipSkip"].as_u64().map(|v| v as u32),
            resolution,
            trigger_words,
            samplers: civitai_data["samplers"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            steps: civitai_data["steps"].as_u64().map(|v| v as u32),
            cfg_scale: civitai_data["cfgScale"].as_f64().map(|v| v as f32),
            nsfw: civitai_data["nsfw"].as_bool().unwrap_or(false),
            license: civitai_data["license"].as_str().map(|s| s.to_string()),
            tags: civitai_data["tags"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            preview_images,
            downloads: civitai_data["downloads"].as_u64().unwrap_or(0),
            likes: civitai_data["likes"].as_u64().unwrap_or(0),
        }
    }
}

impl ModelConfig for ImageConfig {
    fn model_type(&self) -> ModelType {
        ModelType::Image
    }
    
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::json!({}))
    }
    
    fn is_compatible_with(&self, worker_type: &str) -> bool {
        match worker_type {
            "stable-diffusion" | "sd" => true,
            "comfyui" => true,
            "automatic1111" | "a1111" => true,
            _ => false
        }
    }
    
    fn inference_params(&self) -> InferenceParams {
        let mut additional = HashMap::new();
        
        additional.insert("resolution".to_string(), 
            serde_json::json!([self.resolution.0, self.resolution.1]));
        
        if let Some(cfg) = self.cfg_scale {
            additional.insert("cfg_scale".to_string(), serde_json::json!(cfg));
        }
        
        if let Some(steps) = self.steps {
            additional.insert("steps".to_string(), serde_json::json!(steps));
        }
        
        if let Some(clip_skip) = self.clip_skip {
            additional.insert("clip_skip".to_string(), serde_json::json!(clip_skip));
        }
        
        if !self.trigger_words.is_empty() {
            additional.insert("trigger_words".to_string(), serde_json::json!(self.trigger_words));
        }
        
        InferenceParams {
            context_length: None,
            batch_size: Some(1),
            additional,
        }
    }
}
