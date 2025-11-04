// TEAM-405: HuggingFace API client for marketplace-sdk
//! HuggingFace API client for searching and listing models
//!
//! This is the NATIVE Rust implementation (not WASM) for use in Tauri/backend.

use crate::types::{Model, ModelSource};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// HuggingFace API base URL
const HF_API_BASE: &str = "https://huggingface.co/api";

/// HuggingFace model response from API
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFModelResponse {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(default)]
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
    /// * `limit` - Maximum number of results (default: 50)
    pub async fn list_models(&self, query: Option<String>, limit: Option<u32>) -> Result<Vec<Model>> {
        let limit = limit.unwrap_or(50);
        let mut url = format!("{}/models?limit={}", HF_API_BASE, limit);

        if let Some(q) = query {
            url.push_str(&format!("&search={}", urlencoding::encode(&q)));
        }

        // Add filter for text-generation models
        url.push_str("&filter=text-generation");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch models from HuggingFace")?;

        let hf_models: Vec<HFModelResponse> = response
            .json()
            .await
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
        self.list_models(Some(query.to_string()), limit).await
    }

    /// Convert HuggingFace model to our Model type
    fn convert_hf_model(&self, hf_model: HFModelResponse) -> Model {
        // Extract model name from model_id (format: "author/model-name")
        let name = hf_model
            .model_id
            .split('/')
            .last()
            .unwrap_or(&hf_model.model_id)
            .to_string();

        // Generate description from pipeline_tag and tags
        let description = if let Some(pipeline) = &hf_model.pipeline_tag {
            format!("{} model", pipeline.replace('-', " "))
        } else {
            "Language model".to_string()
        };

        // Estimate size (placeholder - would need to fetch from model card)
        let size = "Unknown".to_string();

        Model {
            id: hf_model.model_id.clone(),
            name,
            description,
            author: hf_model.author,
            image_url: None, // HF API doesn't provide image URLs directly
            tags: hf_model.tags,
            downloads: hf_model.downloads,
            likes: hf_model.likes,
            size,
            source: ModelSource::HuggingFace,
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
        let models = client.list_models(None, Some(5)).await;
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
