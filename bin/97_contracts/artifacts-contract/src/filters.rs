// TEAM-464: Shared filter types for marketplace providers
// Used by Civitai, HuggingFace, and UI components

use serde::{Deserialize, Serialize};

#[cfg(feature = "specta")]
use specta::Type;

#[cfg(target_arch = "wasm32")]
use tsify::Tsify;

use crate::nsfw::NsfwFilter;

// ============================================================================
// Common Filter Types
// ============================================================================

/// Time period for filtering models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub enum TimePeriod {
    /// All time
    AllTime,
    /// Past year
    Year,
    /// Past month
    Month,
    /// Past week
    Week,
    /// Past day
    Day,
}

impl Default for TimePeriod {
    fn default() -> Self {
        TimePeriod::AllTime
    }
}

impl TimePeriod {
    /// Get API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            TimePeriod::AllTime => "AllTime",
            TimePeriod::Year => "Year",
            TimePeriod::Month => "Month",
            TimePeriod::Week => "Week",
            TimePeriod::Day => "Day",
        }
    }
}

// ============================================================================
// Civitai Filter Types
// ============================================================================

/// Model type for Civitai
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub enum CivitaiModelType {
    /// All model types
    All,
    /// Stable Diffusion checkpoint
    Checkpoint,
    /// LoRA (Low-Rank Adaptation)
    #[serde(rename = "LORA")]
    Lora,
    /// Textual Inversion embedding
    TextualInversion,
    /// Hypernetwork
    Hypernetwork,
    /// Aesthetic Gradient
    AestheticGradient,
    /// ControlNet
    Controlnet,
    /// Upscaler
    Upscaler,
    /// Motion Module
    MotionModule,
    /// VAE (Variational Autoencoder)
    #[serde(rename = "VAE")]
    Vae,
    /// Pose models
    Poses,
    /// Wildcards
    Wildcards,
    /// Workflows
    Workflows,
    /// Detection models
    Detection,
    /// Other types
    Other,
}

impl Default for CivitaiModelType {
    fn default() -> Self {
        CivitaiModelType::All
    }
}

impl CivitaiModelType {
    /// Get API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CivitaiModelType::All => "All",
            CivitaiModelType::Checkpoint => "Checkpoint",
            CivitaiModelType::Lora => "LORA",
            CivitaiModelType::TextualInversion => "TextualInversion",
            CivitaiModelType::Hypernetwork => "Hypernetwork",
            CivitaiModelType::AestheticGradient => "AestheticGradient",
            CivitaiModelType::Controlnet => "Controlnet",
            CivitaiModelType::Upscaler => "Upscaler",
            CivitaiModelType::MotionModule => "MotionModule",
            CivitaiModelType::Vae => "VAE",
            CivitaiModelType::Poses => "Poses",
            CivitaiModelType::Wildcards => "Wildcards",
            CivitaiModelType::Workflows => "Workflows",
            CivitaiModelType::Detection => "Detection",
            CivitaiModelType::Other => "Other",
        }
    }

    /// Get display label
    pub fn label(&self) -> &'static str {
        match self {
            CivitaiModelType::All => "All Types",
            CivitaiModelType::Checkpoint => "Checkpoint",
            CivitaiModelType::Lora => "LoRA",
            CivitaiModelType::TextualInversion => "Textual Inversion",
            CivitaiModelType::Hypernetwork => "Hypernetwork",
            CivitaiModelType::AestheticGradient => "Aesthetic Gradient",
            CivitaiModelType::Controlnet => "ControlNet",
            CivitaiModelType::Upscaler => "Upscaler",
            CivitaiModelType::MotionModule => "Motion Module",
            CivitaiModelType::Vae => "VAE",
            CivitaiModelType::Poses => "Poses",
            CivitaiModelType::Wildcards => "Wildcards",
            CivitaiModelType::Workflows => "Workflows",
            CivitaiModelType::Detection => "Detection",
            CivitaiModelType::Other => "Other",
        }
    }
}

/// Base model for compatibility filtering
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub enum BaseModel {
    /// All base models
    All,
    /// SDXL 1.0
    #[serde(rename = "SDXL 1.0")]
    SdxlV1,
    /// SD 1.5
    #[serde(rename = "SD 1.5")]
    SdV15,
    /// SD 2.1
    #[serde(rename = "SD 2.1")]
    SdV21,
    /// Pony
    Pony,
    /// Flux
    Flux,
}

impl Default for BaseModel {
    fn default() -> Self {
        BaseModel::All
    }
}

impl BaseModel {
    /// Get API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            BaseModel::All => "All",
            BaseModel::SdxlV1 => "SDXL 1.0",
            BaseModel::SdV15 => "SD 1.5",
            BaseModel::SdV21 => "SD 2.1",
            BaseModel::Pony => "Pony",
            BaseModel::Flux => "Flux",
        }
    }
}

/// Sort options for Civitai
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub enum CivitaiSort {
    /// Most downloaded models
    #[serde(rename = "Most Downloaded")]
    MostDownloaded,
    /// Highest rated models
    #[serde(rename = "Highest Rated")]
    HighestRated,
    /// Newest models
    #[serde(rename = "Newest")]
    Newest,
}

impl Default for CivitaiSort {
    fn default() -> Self {
        CivitaiSort::MostDownloaded
    }
}

impl CivitaiSort {
    /// Get API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CivitaiSort::MostDownloaded => "Most Downloaded",
            CivitaiSort::HighestRated => "Highest Rated",
            CivitaiSort::Newest => "Newest",
        }
    }
}

/// Complete Civitai filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub struct CivitaiFilters {
    /// Time period filter
    pub time_period: TimePeriod,
    /// Model type filter
    pub model_type: CivitaiModelType,
    /// Base model filter
    pub base_model: BaseModel,
    /// Sort order
    pub sort: CivitaiSort,
    /// NSFW content filter
    pub nsfw: NsfwFilter,
    /// Page number for pagination
    pub page: Option<u32>,
    /// Number of results per page
    pub limit: u32,
}

impl Default for CivitaiFilters {
    fn default() -> Self {
        Self {
            time_period: TimePeriod::AllTime,
            model_type: CivitaiModelType::All,
            base_model: BaseModel::All,
            sort: CivitaiSort::MostDownloaded,
            nsfw: NsfwFilter::default(),
            page: None,
            limit: 100,
        }
    }
}

// ============================================================================
// HuggingFace Filter Types
// ============================================================================

/// Sort options for HuggingFace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub enum HuggingFaceSort {
    /// Sort by number of downloads
    Downloads,
    /// Sort by number of likes
    Likes,
    /// Sort by last modified date
    Recent,
    /// Sort by trending score
    Trending,
}

impl Default for HuggingFaceSort {
    fn default() -> Self {
        HuggingFaceSort::Downloads
    }
}

impl HuggingFaceSort {
    /// Get API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            HuggingFaceSort::Downloads => "downloads",
            HuggingFaceSort::Likes => "likes",
            HuggingFaceSort::Recent => "lastModified",
            HuggingFaceSort::Trending => "trending",
        }
    }
}

/// Complete HuggingFace filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub struct HuggingFaceFilters {
    /// Search query
    pub query: Option<String>,
    /// Sort order
    pub sort: HuggingFaceSort,
    /// Filter by tags (e.g., "text-generation", "llama")
    pub tags: Vec<String>,
    /// Number of results to return
    pub limit: u32,
}

impl Default for HuggingFaceFilters {
    fn default() -> Self {
        Self {
            query: None,
            sort: HuggingFaceSort::Downloads,
            tags: vec!["text-generation".to_string()],
            limit: 50,
        }
    }
}
