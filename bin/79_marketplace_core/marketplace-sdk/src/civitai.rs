// TEAM-460: Civitai API client for marketplace-sdk
//! Civitai API client for searching and listing Stable Diffusion models
//!
//! This is the NATIVE Rust implementation (not WASM) for use in Tauri/backend.

use crate::types::{Model, ModelFile, ModelProvider, ModelCategory};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Civitai API base URL
const CIVITAI_API_BASE: &str = "https://civitai.com/api/v1";

/// Civitai model type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CivitaiModelType {
    /// Stable Diffusion checkpoint model
    Checkpoint,
    /// Textual inversion embedding
    TextualInversion,
    /// Hypernetwork model
    Hypernetwork,
    /// Aesthetic gradient model
    AestheticGradient,
    /// LoRA (Low-Rank Adaptation) model
    #[serde(rename = "LORA")]
    Lora,
    /// ControlNet model for guided generation
    Controlnet,
    /// Pose/skeleton models
    Poses,
}

/// Civitai model response from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiModelResponse {
    /// Unique model ID
    pub id: i64,
    /// Model name
    pub name: String,
    /// Model description
    #[serde(default)]
    pub description: String,
    /// Type of model (Checkpoint, LoRA, etc.)
    #[serde(rename = "type")]
    pub model_type: CivitaiModelType,
    /// Whether model depicts a person of interest
    #[serde(default)]
    pub poi: bool,
    /// Whether model is NSFW
    #[serde(default)]
    pub nsfw: bool,
    /// Whether credit is required
    #[serde(rename = "allowNoCredit", default)]
    pub allow_no_credit: bool,
    /// Commercial use permission level
    #[serde(rename = "allowCommercialUse", default)]
    pub allow_commercial_use: String,
    /// Whether derivatives are allowed
    #[serde(rename = "allowDerivatives", default)]
    pub allow_derivatives: bool,
    /// Whether different licenses are allowed
    #[serde(rename = "allowDifferentLicense", default)]
    pub allow_different_license: bool,
    /// Model statistics
    pub stats: CivitaiStats,
    /// Model creator information
    pub creator: CivitaiCreator,
    /// Model tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// Available model versions
    #[serde(rename = "modelVersions", default)]
    pub model_versions: Vec<CivitaiModelVersion>,
}

/// Civitai model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiStats {
    /// Total number of downloads
    #[serde(rename = "downloadCount", default)]
    pub download_count: i64,
    /// Number of favorites
    #[serde(rename = "favoriteCount", default)]
    pub favorite_count: i64,
    /// Number of comments
    #[serde(rename = "commentCount", default)]
    pub comment_count: i64,
    /// Number of ratings
    #[serde(rename = "ratingCount", default)]
    pub rating_count: i64,
    /// Average rating score
    #[serde(default)]
    pub rating: f64,
}

/// Civitai creator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiCreator {
    /// Creator's username
    pub username: String,
    /// Creator's profile image URL
    #[serde(default)]
    pub image: Option<String>,
}

/// Civitai model version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiModelVersion {
    /// Version ID
    pub id: i64,
    /// Parent model ID
    #[serde(rename = "modelId")]
    pub model_id: i64,
    /// Version name
    pub name: String,
    /// Creation timestamp
    #[serde(rename = "createdAt")]
    pub created_at: String,
    /// Last update timestamp
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    /// Trigger words for this model
    #[serde(rename = "trainedWords", default)]
    pub trained_words: Vec<String>,
    /// Base model (e.g., SD 1.5, SDXL)
    #[serde(rename = "baseModel")]
    pub base_model: String,
    /// Version description
    #[serde(default)]
    pub description: Option<String>,
    /// Version statistics
    pub stats: CivitaiVersionStats,
    /// Available files for this version
    #[serde(default)]
    pub files: Vec<CivitaiFile>,
    /// Example images
    #[serde(default)]
    pub images: Vec<CivitaiImage>,
    /// Direct download URL
    #[serde(rename = "downloadUrl")]
    pub download_url: String,
}

/// Civitai version statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiVersionStats {
    #[serde(rename = "downloadCount", default)]
    pub download_count: i64,
    #[serde(rename = "ratingCount", default)]
    pub rating_count: i64,
    #[serde(default)]
    pub rating: f64,
}

/// Civitai file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiFile {
    /// File name
    pub name: String,
    /// File ID
    pub id: i64,
    /// File size in kilobytes
    #[serde(rename = "sizeKB")]
    pub size_kb: f64,
    /// File type (e.g., "Model", "VAE")
    #[serde(rename = "type")]
    pub file_type: String,
    /// File metadata (precision, size, format)
    pub metadata: CivitaiFileMetadata,
    /// Pickle scan result (security check)
    #[serde(rename = "pickleScanResult")]
    pub pickle_scan_result: String,
    /// Virus scan result
    #[serde(rename = "virusScanResult")]
    pub virus_scan_result: String,
    /// Scan timestamp
    #[serde(rename = "scannedAt")]
    pub scanned_at: String,
    /// File hashes for verification
    pub hashes: CivitaiHashes,
    /// Direct download URL
    #[serde(rename = "downloadUrl")]
    pub download_url: String,
    /// Whether this is the primary file
    #[serde(default)]
    pub primary: bool,
}

/// Civitai file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiFileMetadata {
    #[serde(default)]
    pub fp: Option<String>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
}

/// Civitai file hashes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiHashes {
    #[serde(rename = "AutoV2", default)]
    pub auto_v2: Option<String>,
    #[serde(rename = "SHA256", default)]
    pub sha256: Option<String>,
    #[serde(rename = "CRC32", default)]
    pub crc32: Option<String>,
    #[serde(rename = "BLAKE3", default)]
    pub blake3: Option<String>,
}

/// Civitai image information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiImage {
    /// Image URL
    pub url: String,
    /// Whether image is NSFW
    #[serde(default)]
    pub nsfw: bool,
    /// Image width in pixels
    pub width: i32,
    /// Image height in pixels
    pub height: i32,
    /// Image hash
    pub hash: String,
    /// Image generation metadata
    #[serde(default)]
    pub meta: Option<serde_json::Value>,
}

/// Civitai list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiListResponse {
    /// List of models
    pub items: Vec<CivitaiModelResponse>,
    /// Pagination metadata
    pub metadata: CivitaiMetadata,
}

/// Civitai pagination metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivitaiMetadata {
    /// Total number of items across all pages
    #[serde(rename = "totalItems")]
    pub total_items: i64,
    /// Current page number
    #[serde(rename = "currentPage")]
    pub current_page: i64,
    /// Number of items per page
    #[serde(rename = "pageSize")]
    pub page_size: i64,
    /// Total number of pages
    #[serde(rename = "totalPages")]
    pub total_pages: i64,
    /// URL for the next page (if available)
    #[serde(rename = "nextPage", default)]
    pub next_page: Option<String>,
}

/// Civitai client for searching models
pub struct CivitaiClient {
    client: reqwest::Client,
}

impl CivitaiClient {
    /// Create a new Civitai client
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// List models from Civitai API
    ///
    /// # Arguments
    /// * `limit` - Maximum number of models to return (default: 20)
    /// * `page` - Page number for pagination (default: 1)
    /// * `types` - Filter by model types (e.g., "Checkpoint,LORA")
    /// * `sort` - Sort order ("Highest Rated", "Most Downloaded", "Newest")
    /// * `nsfw` - Include NSFW models (default: false)
    ///
    /// # Example
    /// ```no_run
    /// use marketplace_sdk::CivitaiClient;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = CivitaiClient::new();
    /// let models = client.list_models(
    ///     Some(100),
    ///     None,
    ///     Some("Checkpoint,LORA"),
    ///     Some("Most Downloaded"),
    ///     Some(false),
    ///     Some("Sell"),
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_models(
        &self,
        limit: Option<i32>,
        page: Option<i32>,
        types: Option<&str>,
        sort: Option<&str>,
        nsfw: Option<bool>,
        allow_commercial_use: Option<&str>,
    ) -> Result<CivitaiListResponse> {
        let mut url = format!("{}/models", CIVITAI_API_BASE);
        let mut params = Vec::new();

        if let Some(limit) = limit {
            params.push(format!("limit={}", limit));
        }
        if let Some(page) = page {
            params.push(format!("page={}", page));
        }
        if let Some(types) = types {
            params.push(format!("types={}", urlencoding::encode(types)));
        }
        if let Some(sort) = sort {
            params.push(format!("sort={}", urlencoding::encode(sort)));
        }
        if let Some(nsfw) = nsfw {
            params.push(format!("nsfw={}", nsfw));
        }
        if let Some(allow_commercial_use) = allow_commercial_use {
            params.push(format!("allowCommercialUse={}", urlencoding::encode(allow_commercial_use)));
        }

        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }

        let response = self
            .client
            .get(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .context("Failed to send request to Civitai API")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Civitai API error: {} {}",
                response.status(),
                response.text().await.unwrap_or_default()
            );
        }

        let list_response: CivitaiListResponse = response
            .json()
            .await
            .context("Failed to parse Civitai API response")?;

        Ok(list_response)
    }

    /// Get a specific model by ID
    ///
    /// # Arguments
    /// * `model_id` - Civitai model ID
    ///
    /// # Example
    /// ```no_run
    /// use marketplace_sdk::CivitaiClient;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = CivitaiClient::new();
    /// let model = client.get_model(1102).await?;
    /// println!("Model: {}", model.name);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_model(&self, model_id: i64) -> Result<CivitaiModelResponse> {
        let url = format!("{}/models/{}", CIVITAI_API_BASE, model_id);

        let response = self
            .client
            .get(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .context("Failed to send request to Civitai API")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Civitai API error: {} {}",
                response.status(),
                response.text().await.unwrap_or_default()
            );
        }

        let model: CivitaiModelResponse = response
            .json()
            .await
            .context("Failed to parse Civitai API response")?;

        Ok(model)
    }

    /// Get compatible Stable Diffusion models for rbee
    /// Filters for safe, high-quality checkpoints and LORAs
    ///
    /// # Example
    /// ```no_run
    /// use marketplace_sdk::CivitaiClient;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = CivitaiClient::new();
    /// let models = client.get_compatible_models().await?;
    /// println!("Found {} compatible models", models.items.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_compatible_models(&self) -> Result<CivitaiListResponse> {
        self.list_models(
            Some(100),                // limit
            None,                     // page
            Some("Checkpoint,LORA"),  // types
            Some("Most Downloaded"),  // sort
            Some(false),              // nsfw
            Some("Sell"),             // allow_commercial_use (most permissive)
        )
        .await
    }

    /// Convert Civitai model to marketplace Model type
    pub fn to_marketplace_model(&self, civitai_model: &CivitaiModelResponse) -> Model {
        let latest_version = civitai_model.model_versions.first();

        Model {
            id: format!("civitai-{}", civitai_model.id),  // TEAM-460: Consistent prefix
            name: civitai_model.name.clone(),
            description: civitai_model.description.clone(),
            author: Some(civitai_model.creator.username.clone()),
            image_url: latest_version
                .and_then(|v| v.images.first())
                .map(|img| img.url.clone()),
            tags: civitai_model.tags.clone(),
            downloads: civitai_model.stats.download_count as f64,
            likes: civitai_model.stats.favorite_count as f64,
            size: latest_version
                .and_then(|v| v.files.first())
                .map(|f| format!("{:.2} GB", f.size_kb / 1024.0 / 1024.0))
                .unwrap_or_else(|| "Unknown".to_string()),
            provider: ModelProvider::Civitai,
            category: ModelCategory::Image,
            siblings: latest_version
                .map(|v| {
                    v.files
                        .iter()
                        .map(|f| ModelFile {
                            filename: f.name.clone(),
                            size: Some((f.size_kb * 1024.0) as f64),
                        })
                        .collect()
                }),
        }
    }
}

impl Default for CivitaiClient {
    fn default() -> Self {
        Self::new()
    }
}
