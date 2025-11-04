//! Worker catalog HTTP client
//!
//! TEAM-402: Extracted from worker_install.rs
//!
//! Fetches worker metadata and PKGBUILDs from the worker catalog service.

use anyhow::{Context, Result};
use observability_narration_core::n;
use std::time::Duration;

/// Worker metadata from catalog
#[derive(Debug, serde::Deserialize)]
pub struct WorkerMetadata {
    /// Worker ID (e.g., "llm-worker-rbee-cpu")
    pub id: String,
    
    /// Display name
    pub name: String,
    
    /// Version
    pub version: String,
    
    /// Supported platforms (linux, macos, windows)
    pub platforms: Vec<String>,
    
    /// Supported architectures (x86_64, aarch64)
    pub architectures: Vec<String>,
    
    /// Runtime dependencies
    #[serde(default)]
    pub depends: Vec<String>,
}

/// Worker catalog HTTP client
pub struct CatalogClient {
    base_url: String,
    client: reqwest::Client,
}

impl CatalogClient {
    /// Create a new catalog client
    ///
    /// Uses `WORKER_CATALOG_URL` environment variable or defaults to `http://localhost:8787`
    pub fn new() -> Result<Self> {
        let base_url = std::env::var("WORKER_CATALOG_URL")
            .unwrap_or_else(|_| "http://localhost:8787".to_string());
        
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;
        
        Ok(Self { base_url, client })
    }
    
    /// Create client with custom base URL
    pub fn with_url(base_url: String) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;
        
        Ok(Self { base_url, client })
    }
    
    /// Fetch worker metadata from catalog
    pub async fn fetch_metadata(&self, worker_id: &str) -> Result<WorkerMetadata> {
        let url = format!("{}/workers/{}", self.base_url, worker_id);
        n!("fetch_url", "📡 Fetching from: {}", url);
        
        n!("send_request", "📤 Sending GET request to catalog...");
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch worker metadata")?;
        
        n!("send_request_ok", "✓ Response received: {}", response.status());
        
        if !response.status().is_success() {
            anyhow::bail!(
                "Worker '{}' not found in catalog (HTTP {})",
                worker_id,
                response.status()
            );
        }
        
        n!("parse_json", "📄 Parsing JSON response...");
        let metadata: WorkerMetadata = response
            .json()
            .await
            .context("Failed to parse worker metadata")?;
        
        n!("parse_json_ok", "✓ Metadata parsed: {} v{}", metadata.name, metadata.version);
        
        Ok(metadata)
    }
    
    /// Download PKGBUILD from catalog
    pub async fn download_pkgbuild(&self, worker_id: &str) -> Result<String> {
        let url = format!("{}/workers/{}/PKGBUILD", self.base_url, worker_id);
        n!("pkgbuild_url", "📡 Fetching from: {}", url);
        
        n!("pkgbuild_http_get", "📤 Sending GET request...");
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to download PKGBUILD")?;
        
        n!("pkgbuild_http_response", "📥 Response status: {}", response.status());
        
        if !response.status().is_success() {
            anyhow::bail!(
                "PKGBUILD not found for worker '{}' (HTTP {})",
                worker_id,
                response.status()
            );
        }
        
        n!("pkgbuild_read_body", "📖 Reading response body...");
        let content = response
            .text()
            .await
            .context("Failed to read PKGBUILD content")?;
        
        n!("pkgbuild_read_ok", "✓ PKGBUILD content received ({} bytes)", content.len());
        
        Ok(content)
    }
}

impl Default for CatalogClient {
    fn default() -> Self {
        Self::new().expect("Failed to create catalog client")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation_with_default_url() {
        let client = CatalogClient::new().unwrap();
        assert_eq!(client.base_url, "http://localhost:8787");
    }
    
    #[test]
    fn test_client_creation_with_custom_url() {
        let client = CatalogClient::with_url("http://custom:9999".to_string()).unwrap();
        assert_eq!(client.base_url, "http://custom:9999");
    }
}
