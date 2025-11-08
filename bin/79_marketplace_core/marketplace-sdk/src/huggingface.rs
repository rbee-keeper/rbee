// TEAM-405: HuggingFace API client for marketplace-sdk
//! HuggingFace API client for searching and listing models
//!
//! This is the NATIVE Rust implementation (not WASM) for use in Tauri/backend.

use crate::types::{Model, ModelFile, ModelProvider, ModelCategory};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// HuggingFace API base URL
const HF_API_BASE: &str = "https://huggingface.co/api";

/// HuggingFace model response from API
/// TEAM-405: Using serde_json::Value to capture ALL fields for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFModelResponse {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(default, alias = "author")]
    author: Option<String>,
    #[serde(default)]
    downloads: f64,
    #[serde(default)]
    likes: f64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    private: bool,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default, rename = "createdAt")]
    created_at: Option<String>,
    // TEAM-405: Capture all other fields for debugging
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

/// HuggingFace client for searching models
pub struct HuggingFaceClient {
    client: reqwest::Client,
}

impl HuggingFaceClient {
    /// Create a new HuggingFace client
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// List models from HuggingFace
    ///
    /// # Arguments
    /// * `query` - Search query (optional)
    /// * `sort` - Sort order: "downloads", "likes", "recent", "trending" (optional)
    /// * `filter_tags` - Filter by tags (optional)
    /// * `limit` - Maximum number of results (default: 50)
    pub async fn list_models(
        &self,
        query: Option<String>,
        sort: Option<String>,
        filter_tags: Option<Vec<String>>,
        limit: Option<u32>,
    ) -> Result<Vec<Model>> {
        let limit = limit.unwrap_or(50);
        let mut url = format!("{}/models?limit={}", HF_API_BASE, limit);

        // Add search query
        if let Some(q) = query {
            url.push_str(&format!("&search={}", urlencoding::encode(&q)));
        }

        // Add sort parameter
        if let Some(s) = sort {
            match s.as_str() {
                "downloads" => url.push_str("&sort=downloads"),
                "likes" => url.push_str("&sort=likes"),
                "recent" => url.push_str("&sort=lastModified"),
                "trending" => url.push_str("&sort=trending"),
                _ => url.push_str("&sort=downloads"), // default
            }
            url.push_str("&direction=-1"); // descending
        }

        // Add tag filters
        if let Some(tags) = filter_tags {
            for tag in tags {
                url.push_str(&format!("&filter={}", urlencoding::encode(&tag)));
            }
        } else {
            // Default: only text-generation models
            url.push_str("&filter=text-generation");
        }

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch models from HuggingFace")?;

        // TEAM-405: Get RAW JSON text first so we can print it
        let raw_json = response
            .text()
            .await
            .context("Failed to get response text")?;

        // Parse response
        let hf_models: Vec<HFModelResponse> = serde_json::from_str(&raw_json)
            .context("Failed to parse HuggingFace response")?;

        // Convert HF models to our Model type
        let models = hf_models
            .into_iter()
            .filter(|m| !m.private) // Filter out private models
            .map(|m| self.convert_hf_model(m))
            .collect();

        Ok(models)
    }

    /// Search models by query
    pub async fn search_models(&self, query: &str, limit: Option<u32>) -> Result<Vec<Model>> {
        self.list_models(Some(query.to_string()), None, None, limit).await
    }

    /// Get a specific model by ID
    ///
    /// # Arguments
    /// * `model_id` - Model ID (e.g., "meta-llama/Llama-3.2-1B")
    pub async fn get_model(&self, model_id: &str) -> Result<Model> {
        let url = format!("{}/models/{}", HF_API_BASE, model_id);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch model from HuggingFace")?;

        let raw_json = response
            .text()
            .await
            .context("Failed to get response text")?;

        let hf_model: HFModelResponse = serde_json::from_str(&raw_json)
            .context("Failed to parse HuggingFace model response")?;

        if hf_model.private {
            anyhow::bail!("Model is private");
        }

        Ok(self.convert_hf_model(hf_model))
    }

    /// Convert HuggingFace model to our Model type
    fn convert_hf_model(&self, hf_model: HFModelResponse) -> Model {
        // Extract author and model name from model_id (format: "author/model-name")
        let parts: Vec<&str> = hf_model.model_id.split('/').collect();
        let (author, name) = if parts.len() >= 2 {
            (Some(parts[0].to_string()), parts[1].to_string())
        } else {
            (None, hf_model.model_id.clone())
        };

        // Use author from API if available, otherwise use extracted from model_id
        let author = hf_model.author.or(author);

        // Generate description from pipeline_tag and tags
        let description = if let Some(pipeline) = &hf_model.pipeline_tag {
            format!("{} model", pipeline.replace('-', " "))
        } else {
            "Language model".to_string()
        };

        // Estimate size (placeholder - would need to fetch from model card)
        let size = "Unknown".to_string();

        // TEAM-421: Parse siblings (model files) from extra fields
        let siblings = hf_model.extra.get("siblings")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|file_obj| {
                        let filename = file_obj.get("rfilename")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())?;
                        
                        // Convert u64 to f64 for TypeScript compatibility
                        let size = file_obj.get("size")
                            .and_then(|v| v.as_u64())
                            .map(|s| s as f64);
                        
                        Some(ModelFile { filename, size })
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty()); // Only include if there are files

        Model {
            id: format!("huggingface-{}", hf_model.model_id),  // TEAM-460: Consistent prefix
            name,
            description,
            author,
            image_url: None, // HF API doesn't provide image URLs directly
            tags: hf_model.tags,
            downloads: hf_model.downloads,
            likes: hf_model.likes,
            size,
            provider: ModelProvider::HuggingFace,
            category: ModelCategory::Llm,
            siblings, // TEAM-421: Include model files
        }
    }
}

impl Default for HuggingFaceClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_models() {
        let client = HuggingFaceClient::new();
        let models = client.list_models(None, None, None, Some(5)).await;
        assert!(models.is_ok());
        let models = models.unwrap();
        assert!(!models.is_empty());
        assert!(models.len() <= 5);
    }

    #[tokio::test]
    async fn test_search_models() {
        let client = HuggingFaceClient::new();
        let models = client.search_models("llama", Some(5)).await;
        assert!(models.is_ok());
        let models = models.unwrap();
        assert!(!models.is_empty());
        // Check that results contain "llama" in id or tags
        assert!(models.iter().any(|m| 
            m.id.to_lowercase().contains("llama") || 
            m.tags.iter().any(|t| t.to_lowercase().contains("llama"))
        ));
    }
}
