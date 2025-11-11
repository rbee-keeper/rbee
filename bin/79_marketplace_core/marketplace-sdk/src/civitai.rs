// TEAM-460: Civitai API client for marketplace-sdk
// TEAM-463: Now uses canonical CivitAI types from artifacts-contract
//! Civitai API client for searching and listing Stable Diffusion models
//!
//! This is the NATIVE Rust implementation (not WASM) for use in Tauri/backend.

use crate::types::{Model, ModelFile, ModelProvider, ModelCategory};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// TEAM-463: Import canonical CivitAI types from artifacts-contract
// TEAM-464: Import shared filter types
use artifacts_contract::{
    CivitaiStats, CivitaiCreator,
    CivitaiFilters, CivitaiModelType, TimePeriod, BaseModel,
};

/// Civitai API base URL
const CIVITAI_API_BASE: &str = "https://civitai.com/api/v1";

/// Civitai model response from API (internal parsing type)
/// TEAM-463: Keep for API parsing. Convert to marketplace Model via to_marketplace_model().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CivitaiModelResponse {
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
    /// Commercial use permission level (e.g., ["Image", "RentCivit", "Sell"])
    #[serde(rename = "allowCommercialUse", default)]
    pub allow_commercial_use: Vec<String>,
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
    pub model_versions: Vec<CivitaiModelVersionResponse>,
}

// TEAM-463: CivitaiStats now imported from artifacts-contract (deleted duplicate)

// TEAM-463: CivitaiCreator now imported from artifacts-contract (deleted duplicate)

/// Civitai model version response from API (internal parsing type)
/// TEAM-463: Keep this for API parsing. Has extra fields not in contract type.
/// TEAM-XXX: Made fields optional to handle inconsistent API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CivitaiModelVersionResponse {
    /// Version ID
    pub id: i64,
    /// Parent model ID (optional - not always included when nested in model response)
    #[serde(rename = "modelId", default)]
    pub model_id: Option<i64>,
    /// Version name
    pub name: String,
    /// Creation timestamp (optional - not always included)
    #[serde(rename = "createdAt", default)]
    pub created_at: Option<String>,
    /// Last update timestamp (optional - not always included)
    #[serde(rename = "updatedAt", default)]
    pub updated_at: Option<String>,
    /// Trigger words for this model
    #[serde(rename = "trainedWords", default)]
    pub trained_words: Vec<String>,
    /// Base model (e.g., SD 1.5, SDXL) (optional - not always included)
    #[serde(rename = "baseModel", default)]
    pub base_model: Option<String>,
    /// Version description
    #[serde(default)]
    pub description: Option<String>,
    /// Version statistics (optional - use defaults if missing)
    #[serde(default)]
    pub stats: Option<CivitaiVersionStats>,
    /// Available files for this version
    #[serde(default)]
    pub files: Vec<CivitaiFileResponse>,
    /// Example images
    #[serde(default)]
    pub images: Vec<CivitaiImageResponse>,
    /// Direct download URL (optional - not always included)
    #[serde(rename = "downloadUrl", default)]
    pub download_url: Option<String>,
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

/// Civitai file response from API (internal parsing type)
/// TEAM-463: Keep for API parsing. Has extra security/metadata fields not in contract type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CivitaiFileResponse {
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

/// Civitai image response from API (internal parsing type)
/// TEAM-463: Keep for API parsing. Has extra metadata fields not in contract type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CivitaiImageResponse {
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
    /// * `filters` - Filter configuration for the query
    ///
    /// # Example
    /// ```no_run
    /// use marketplace_sdk::CivitaiClient;
    /// use artifacts_contract::CivitaiFilters;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = CivitaiClient::new();
    /// let filters = CivitaiFilters::default();
    /// let models = client.list_models(&filters).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_models(&self, filters: &CivitaiFilters) -> Result<CivitaiListResponse> {
        use observability_narration_core::n;
        
        let url = format!("{}/models", CIVITAI_API_BASE);
        let mut query_params: Vec<(&str, String)> = Vec::new();

        // Limit and page
        query_params.push(("limit", filters.limit.to_string()));
        n!("civitai_list_models", "📋 limit={}", filters.limit);
        
        if let Some(page) = filters.page {
            query_params.push(("page", page.to_string()));
            n!("civitai_list_models", "📋 page={}", page);
        }

        // Model types
        if filters.model_type != CivitaiModelType::All {
            query_params.push(("types", filters.model_type.as_str().to_string()));
            n!("civitai_list_models", "📋 types={}", filters.model_type.as_str());
        } else {
            query_params.push(("types", "Checkpoint".to_string()));
            query_params.push(("types", "LORA".to_string()));
            n!("civitai_list_models", "📋 types=Checkpoint,LORA (default)");
        }

        // Sort
        query_params.push(("sort", filters.sort.as_str().to_string()));
        n!("civitai_list_models", "📋 sort={}", filters.sort.as_str());

        // Time period
        if filters.time_period != TimePeriod::AllTime {
            query_params.push(("period", filters.time_period.as_str().to_string()));
            n!("civitai_list_models", "📋 period={}", filters.time_period.as_str());
        }

        // Base model
        if filters.base_model != BaseModel::All {
            query_params.push(("baseModel", filters.base_model.as_str().to_string()));
            n!("civitai_list_models", "📋 baseModel={}", filters.base_model.as_str());
        }

        // NSFW filtering
        let nsfw_levels = filters.nsfw.max_level.allowed_levels();
        for level in &nsfw_levels {
            query_params.push(("nsfwLevel", level.as_number().to_string()));
        }
        n!("civitai_list_models", "📋 nsfwLevel={:?}", nsfw_levels.iter().map(|l| l.as_number()).collect::<Vec<_>>());

        // TEAM-464: Debug logging - build URL for display
        let debug_url = if query_params.is_empty() {
            url.clone()
        } else {
            let params_str = query_params
                .iter()
                .map(|(k, v)| format!("{}={}", k, urlencoding::encode(v)))
                .collect::<Vec<_>>()
                .join("&");
            format!("{}?{}", url, params_str)
        };
        n!("civitai_list_models", "🌐 API Request URL: {}", debug_url);
        n!("civitai_list_models", "🔢 Total query params: {}", query_params.len());

        let response = self
            .client
            .get(&url)
            .query(&query_params)
            .header("Content-Type", "application/json")
            .send()
            .await
            .context("Failed to send request to Civitai API")?;

        let status = response.status();
        n!("civitai_list_models", "📡 Response status: {}", status);

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            
            // TEAM-463: Pretty-print JSON errors for better readability
            let formatted_error = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&error_text) {
                serde_json::to_string_pretty(&json).unwrap_or(error_text)
            } else {
                error_text
            };
            
            n!("civitai_list_models", "❌ API Error Response: {}", formatted_error);
            anyhow::bail!(
                "Civitai API error: {}\n{}",
                status,
                formatted_error
            );
        }

        // TEAM-464: Get response text first for better error messages
        let response_text = response.text().await
            .context("Failed to read response body")?;
        
        n!("civitai_list_models", "📦 Response size: {} bytes", response_text.len());
        
        // Try to parse and show detailed error if it fails
        let list_response: CivitaiListResponse = match serde_json::from_str(&response_text) {
            Ok(parsed) => {
                n!("civitai_list_models", "✅ Successfully parsed response");
                parsed
            }
            Err(e) => {
                n!("civitai_list_models", "❌ Parse error: {}", e);
                n!("civitai_list_models", "📄 Response preview (first 500 chars): {}", 
                    &response_text.chars().take(500).collect::<String>());
                anyhow::bail!("Failed to parse Civitai API response: {}\nResponse: {}", e, 
                    &response_text.chars().take(1000).collect::<String>());
            }
        };

        Ok(list_response)
    }

    /// Get a specific model by ID (returns internal response type)
    /// TEAM-463: Internal method. Use get_marketplace_model() for public API.
    pub(crate) async fn get_model(&self, model_id: i64) -> Result<CivitaiModelResponse> {
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

    /// Get a specific model by ID and convert to marketplace Model
    /// TEAM-463: Public API that returns marketplace Model type
    ///
    /// # Example
    /// ```no_run
    /// use marketplace_sdk::CivitaiClient;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = CivitaiClient::new();
    /// let model = client.get_marketplace_model(1102).await?;
    /// println!("Model: {}", model.name);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_marketplace_model(&self, model_id: i64) -> Result<Model> {
        let civitai_model = self.get_model(model_id).await?;
        Ok(self.to_marketplace_model(&civitai_model))
    }

    /// Get compatible Stable Diffusion models for rbee
    /// Shows checkpoints and LORAs, sorted by downloads
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
    pub(crate) async fn get_compatible_models(&self) -> Result<CivitaiListResponse> {
        let filters = CivitaiFilters::default();
        self.list_models(&filters).await
    }

    /// Get compatible models and convert to marketplace Model types
    /// TEAM-463: Public API that returns Vec<Model>
    pub async fn get_compatible_marketplace_models(&self) -> Result<Vec<Model>> {
        let response = self.get_compatible_models().await?;
        Ok(response.items
            .iter()
            .map(|civitai_model| self.to_marketplace_model(civitai_model))
            .collect())
    }

    /// List models and convert to marketplace Model types
    /// TEAM-429: Public API for filtered model listing
    pub async fn list_marketplace_models(&self, filters: &CivitaiFilters) -> Result<Vec<Model>> {
        let response = self.list_models(filters).await?;
        Ok(response.items
            .iter()
            .map(|civitai_model| self.to_marketplace_model(civitai_model))
            .collect())
    }

    /// Convert Civitai model to marketplace Model type
    /// TEAM-463: Internal conversion function
    /// TEAM-429: Made public for Tauri command usage
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
