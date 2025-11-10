// TEAM-463: Canonical CivitAI model types
//! CivitAI model types - source of truth for marketplace and UI
//!
//! These types represent CivitAI models in our system.
//! They are used across:
//! - marketplace-sdk (Rust backend)
//! - marketplace-node (Node.js/WASM)
//! - UI components (TypeScript via tsify)

use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use tsify::Tsify;

/// CivitAI model
///
/// TEAM-463: Canonical type for CivitAI models in marketplace.
/// Source of truth for all CivitAI model representations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct CivitaiModel {
    /// Model ID
    pub id: i64,
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model type (Checkpoint, LORA, etc.)
    #[serde(rename = "type")]
    pub model_type: String,
    /// Whether model is NSFW
    pub nsfw: bool,
    /// Commercial use permission
    pub allow_commercial_use: String,
    /// Model statistics
    pub stats: CivitaiStats,
    /// Model creator
    pub creator: CivitaiCreator,
    /// Model tags
    pub tags: Vec<String>,
    /// Model versions
    pub model_versions: Vec<CivitaiModelVersion>,
}

/// CivitAI model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct CivitaiStats {
    /// Download count
    pub download_count: i64,
    /// Favorite count
    pub favorite_count: i64,
    /// Comment count
    pub comment_count: i64,
    /// Rating count
    pub rating_count: i64,
    /// Average rating
    pub rating: f64,
}

/// CivitAI creator information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct CivitaiCreator {
    /// Creator username
    pub username: String,
    /// Creator avatar image URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

/// CivitAI model version
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct CivitaiModelVersion {
    /// Version ID
    pub id: i64,
    /// Parent model ID
    pub model_id: i64,
    /// Version name
    pub name: String,
    /// Base model (SDXL, SD 1.5, etc.)
    pub base_model: String,
    /// Trained trigger words
    pub trained_words: Vec<String>,
    /// Model files
    pub files: Vec<CivitaiFile>,
    /// Preview images
    pub images: Vec<CivitaiImage>,
    /// Creation timestamp (optional, from API)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Last update timestamp (optional, from API)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

/// CivitAI file information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct CivitaiFile {
    /// File name
    pub name: String,
    /// File ID
    pub id: i64,
    /// File size in KB
    pub size_kb: f64,
    /// Download URL
    pub download_url: String,
    /// Whether this is the primary file
    pub primary: bool,
}

/// CivitAI image information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct CivitaiImage {
    /// Image URL
    pub url: String,
    /// Whether image is NSFW
    pub nsfw: bool,
    /// Image width
    pub width: i32,
    /// Image height
    pub height: i32,
}
