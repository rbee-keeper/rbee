// TEAM-405: Model module with trait system
// TEAM-407: Added metadata module for marketplace compatibility
//! Model types with trait-based configuration system
//!
//! Supports different model types (LLM, Image, Audio, Video) with
//! type-specific configurations while maintaining a common interface.
//!
//! TEAM-407: Added ModelMetadata for marketplace filtering

mod config;
mod llm;
mod image;
mod metadata;

pub use config::{ModelConfig, InferenceParams};
pub use llm::{LlmConfig, TokenizerConfig};
pub use image::{ImageConfig, CheckpointType, ImagePreview};
pub use metadata::{ModelArchitecture, ModelFormat, Quantization, ModelMetadata};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tsify::Tsify;

use crate::status::ArtifactStatus;

/// Model file information from repository
/// 
/// TEAM-463: Canonical type for model files (siblings) in repositories.
/// Used by marketplace SDK, catalog, and UI components.
/// 
/// This represents a single file in a model repository (e.g., HuggingFace).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct ModelFile {
    /// File name (relative path in repo)
    pub filename: String,
    /// File size in bytes (optional, using f64 for TypeScript compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
}

/// Model type enum
#[derive(Debug, Clone, Serialize, Deserialize, Tsify, PartialEq, Eq)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum ModelType {
    /// Large Language Model (text generation)
    Llm,
    /// Image generation model
    Image,
    /// Audio generation model (future)
    Audio,
    /// Video generation model (future)
    Video,
}

/// Model source/vendor
#[derive(Debug, Clone, Serialize, Deserialize, Tsify, PartialEq, Eq)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum ModelSource {
    /// HuggingFace model hub
    HuggingFace,
    /// CivitAI marketplace
    CivitAI,
    /// Local/custom model
    Local,
    /// Other source
    Other(String),
}

/// Model entry in the catalog
///
/// This is the base structure for all models, regardless of type.
/// Type-specific configuration is stored in the `config` field.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    // ========== Core Identity ==========
    /// Unique model ID (e.g., "meta-llama/Llama-2-7b", "civitai:12345")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Author/organization
    #[serde(default)]
    pub author: Option<String>,

    /// Model source (HuggingFace, CivitAI, Local, etc.)
    pub source: ModelSource,

    // ========== Local Storage ==========
    /// Filesystem path to model files
    #[tsify(type = "string")]
    pub path: PathBuf,

    /// Size in bytes
    pub size: u64,

    /// Current status
    pub status: ArtifactStatus,

    /// When the model was added to catalog
    #[serde(default = "chrono::Utc::now")]
    pub added_at: chrono::DateTime<chrono::Utc>,

    // ========== Model Type ==========
    /// Model type (Llm, Image, Audio, Video)
    pub model_type: ModelType,

    // ========== Type-Specific Configuration ==========
    /// Serialized model configuration
    /// 
    /// Deserialize based on model_type:
    /// - ModelType::Llm -> LlmConfig
    /// - ModelType::Image -> ImageConfig
    /// - etc.
    pub config: serde_json::Value,
}

impl ModelEntry {
    /// Create a new minimal model entry (for testing/backwards compatibility)
    pub fn new(id: String, name: String, path: PathBuf, size: u64) -> Self {
        Self {
            id,
            name,
            author: None,
            source: ModelSource::Local,
            path,
            size,
            status: ArtifactStatus::Available,
            added_at: chrono::Utc::now(),
            model_type: ModelType::Llm,
            config: serde_json::json!({}),
        }
    }

    /// Create a ModelEntry from HuggingFace API response
    /// 
    /// This creates an LLM model with full HuggingFace metadata.
    pub fn from_huggingface(hf_data: &serde_json::Value, path: PathBuf) -> Self {
        let id = hf_data["modelId"].as_str().unwrap_or("").to_string();
        
        // Extract name from model_id
        let name = id.split('/').last().unwrap_or(&id).to_string();
        
        // Extract author from model_id or author field
        let author = hf_data["author"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| id.split('/').next().map(|s| s.to_string()));
        
        // Create LLM config from HuggingFace data
        let llm_config = LlmConfig::from_huggingface(hf_data);
        
        Self {
            id,
            name,
            author,
            source: ModelSource::HuggingFace,
            path,
            size: hf_data["usedStorage"].as_u64().unwrap_or(0),
            status: ArtifactStatus::Available,
            added_at: chrono::Utc::now(),
            model_type: ModelType::Llm,
            config: serde_json::to_value(&llm_config).unwrap_or(serde_json::json!({})),
        }
    }

    /// Get LLM config (if this is an LLM model)
    pub fn as_llm_config(&self) -> Option<LlmConfig> {
        if self.model_type == ModelType::Llm {
            serde_json::from_value(self.config.clone()).ok()
        } else {
            None
        }
    }

    /// Get Image config (if this is an Image model)
    pub fn as_image_config(&self) -> Option<ImageConfig> {
        if self.model_type == ModelType::Image {
            serde_json::from_value(self.config.clone()).ok()
        } else {
            None
        }
    }

    /// Set the status (for Artifact trait compatibility)
    pub fn set_status(&mut self, status: ArtifactStatus) {
        self.status = status;
    }
}
