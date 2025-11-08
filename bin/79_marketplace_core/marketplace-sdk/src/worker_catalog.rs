// TEAM-408: Worker catalog client for marketplace-sdk
//! Worker catalog client
//!
//! Provides access to the worker catalog API for listing and filtering
//! available workers by type, platform, and capabilities.
//!
//! **IMPORTANT:** This works with `WorkerCatalogEntry` (available workers)
//! NOT `WorkerBinary` (installed workers). These are different types!

use anyhow::{Context, Result};
use artifacts_contract::{
    Architecture, Platform, WorkerCatalogEntry, WorkerType,
};
use serde::{Deserialize, Serialize};

/// Response wrapper from Hono API
#[derive(Debug, Deserialize)]
struct WorkersListResponse {
    workers: Vec<WorkerCatalogEntry>,
}

/// Worker catalog client
///
/// Fetches worker catalog entries from the Hono worker catalog API.
/// 
/// **Note:** Returns `WorkerCatalogEntry` (available workers for download),
/// not `WorkerBinary` (installed workers on filesystem).
pub struct WorkerCatalogClient {
    base_url: String,
    client: reqwest::Client,
}

/// Worker filter options
///
/// Used to filter workers by type, platform, architecture, and capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(tsify::Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct WorkerFilter {
    /// Filter by worker type (cpu, cuda, metal)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<WorkerType>,
    
    /// Filter by platform (linux, macos, windows)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform: Option<Platform>,
    
    /// Filter by CPU architecture (x86_64, aarch64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<Architecture>,
    
    /// Minimum context length required
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_context_length: Option<u32>,
    
    /// Filter by supported model architecture (llama, mistral, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_architecture: Option<String>,
    
    /// Filter by supported model format (safetensors, gguf)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_format: Option<String>,
}

impl WorkerCatalogClient {
    /// Create a new worker catalog client
    ///
    /// # Arguments
    /// * `base_url` - Base URL of the worker catalog API (default: http://localhost:8787)
    ///
    /// # Default Port
    /// The global worker catalog (Hono/Cloudflare Worker) runs on port 8787 by default.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
        }
    }
    
    /// Create a client with default localhost URL
    ///
    /// TEAM-421: Changed from port 3000 to 8787 (Hono worker catalog default port)
    pub fn default() -> Self {
        Self::new("http://localhost:8787")
    }
    
    /// List all available workers
    ///
    /// Fetches the complete list of worker catalog entries.
    ///
    /// # Returns
    /// Vector of `WorkerCatalogEntry` (available workers)
    ///
    /// # Errors
    /// Returns error if network request fails or response cannot be parsed
    pub async fn list_workers(&self) -> Result<Vec<WorkerCatalogEntry>> {
        let url = format!("{}/workers", self.base_url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch workers from catalog")?;
        
        if !response.status().is_success() {
            anyhow::bail!(
                "Worker catalog API returned error: {}",
                response.status()
            );
        }
        
        // TEAM-423: Parse response (removed verbose logging)
        let response_text = response
            .text()
            .await
            .context("Failed to read response body as text")?;
        
        // TEAM-408: Hono API returns { workers: [...] } wrapper
        let response_data: WorkersListResponse = serde_json::from_str(&response_text)
            .context("Failed to parse worker catalog response")?;
        
        Ok(response_data.workers)
    }
    
    /// Get a specific worker by ID
    ///
    /// # Arguments
    /// * `id` - Worker ID (e.g., "llm-worker-rbee-cuda")
    ///
    /// # Returns
    /// `Some(WorkerCatalogEntry)` if found, `None` if not found
    ///
    /// # Errors
    /// Returns error if network request fails
    pub async fn get_worker(&self, id: &str) -> Result<Option<WorkerCatalogEntry>> {
        let url = format!("{}/workers/{}", self.base_url, id);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch worker from catalog")?;
        
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        
        if !response.status().is_success() {
            anyhow::bail!(
                "Worker catalog API returned error: {}",
                response.status()
            );
        }
        
        let worker: WorkerCatalogEntry = response
            .json()
            .await
            .context("Failed to parse worker response")?;
        
        Ok(Some(worker))
    }
    
    /// Filter workers by criteria
    ///
    /// Filters the worker list based on the provided filter options.
    /// All filters are AND-ed together (worker must match all specified criteria).
    ///
    /// # Arguments
    /// * `filter` - Filter criteria
    ///
    /// # Returns
    /// Vector of matching `WorkerCatalogEntry` entries
    ///
    /// # Errors
    /// Returns error if network request fails
    pub async fn filter_workers(&self, filter: WorkerFilter) -> Result<Vec<WorkerCatalogEntry>> {
        let workers = self.list_workers().await?;
        
        let filtered = workers
            .into_iter()
            .filter(|worker| {
                // Filter by worker type
                if let Some(ref wtype) = filter.worker_type {
                    if worker.worker_type != *wtype {
                        return false;
                    }
                }
                
                // Filter by platform
                if let Some(ref platform) = filter.platform {
                    if !worker.supports_platform(*platform) {
                        return false;
                    }
                }
                
                // TEAM-408: Filter by CPU architecture (x86_64, aarch64)
                if let Some(arch) = filter.architecture {
                    if !worker.supports_architecture(arch) {
                        return false;
                    }
                }
                
                // Filter by minimum context length
                if let Some(min_context) = filter.min_context_length {
                    if let Some(max_context) = worker.max_context_length {
                        if max_context < min_context {
                            return false;
                        }
                    } else {
                        // No max_context_length specified, skip this worker
                        return false;
                    }
                }
                
                // Filter by model architecture support (llama, mistral, etc.)
                if let Some(ref arch) = filter.model_architecture {
                    if !worker.supports_format(arch) {
                        return false;
                    }
                }
                
                // Filter by model format support (safetensors, gguf)
                if let Some(ref format) = filter.model_format {
                    if !worker.supports_format(format) {
                        return false;
                    }
                }
                
                true
            })
            .collect();
        
        Ok(filtered)
    }
    
    /// Find workers compatible with a specific model
    ///
    /// Checks both model architecture and format compatibility.
    ///
    /// # Arguments
    /// * `architecture` - Model architecture (e.g., "llama", "mistral")
    /// * `format` - Model format (e.g., "safetensors", "gguf")
    ///
    /// # Returns
    /// Vector of compatible `WorkerCatalogEntry` entries
    ///
    /// # Errors
    /// Returns error if network request fails
    pub async fn find_compatible_workers(
        &self,
        architecture: &str,
        format: &str,
    ) -> Result<Vec<WorkerCatalogEntry>> {
        let filter = WorkerFilter {
            model_architecture: Some(architecture.to_string()),
            model_format: Some(format.to_string()),
            ..Default::default()
        };
        
        self.filter_workers(filter).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_worker_filter_default() {
        let filter = WorkerFilter::default();
        assert!(filter.worker_type.is_none());
        assert!(filter.platform.is_none());
        assert!(filter.architecture.is_none());
        assert!(filter.min_context_length.is_none());
    }
    
    #[test]
    fn test_client_creation() {
        let client = WorkerCatalogClient::new("http://localhost:8787");
        assert_eq!(client.base_url, "http://localhost:8787");
        
        let default_client = WorkerCatalogClient::default();
        assert_eq!(default_client.base_url, "http://localhost:8787");
    }
}
