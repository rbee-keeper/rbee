# Part 2: HuggingFace Client Implementation

**Status:** ⏳ Pending  
**Estimated Time:** 3-4 days  
**Prerequisites:** Part 1 complete, HuggingFace API knowledge

---

## Overview

Implement a complete HuggingFace API client for searching, listing, and discovering models. This client will be the primary source for LLM models in the marketplace.

---

## Required Reading

### HuggingFace API Documentation
- **[HuggingFace Hub API](https://huggingface.co/docs/hub/api)** (Full read - 30 min)
  - Authentication
  - Rate limiting
  - Pagination
  - Error handling

- **[Models API](https://huggingface.co/docs/hub/api#models)** (Full read - 20 min)
  - List models endpoint
  - Search models endpoint
  - Model details endpoint
  - Model files endpoint

- **[GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** (Sections 1-3 - 15 min)
  - Understanding GGUF structure
  - Quantization types
  - File naming conventions

### HTTP Client in WASM
- **[reqwest WASM](https://docs.rs/reqwest/latest/reqwest/#wasm)** (Full read - 10 min)
  - WASM-specific features
  - CORS handling
  - Async in WASM

- **[wasm-bindgen-futures](https://docs.rs/wasm-bindgen-futures/)** (Examples - 10 min)
  - Converting Rust futures to JS Promises
  - Error handling across boundary

### Async Rust
- **[Async Book](https://rust-lang.github.io/async-book/)** (Chapters 1-3 - 45 min)
  - Understanding async/await
  - Futures and executors
  - Pinning

**Total Reading Time:** ~2 hours

---

## Architecture

```
src/huggingface/
├── mod.rs              # Module exports
├── client.rs           # Main client implementation
├── types.rs            # HuggingFace-specific types
├── endpoints.rs        # API endpoint builders
└── filters.rs          # Filter/search logic
```

---

## Tasks

### 1. Create Module Structure

**Create `src/huggingface/mod.rs`:**
```rust
// TEAM-XXX: HuggingFace API client module

mod client;
mod types;
mod endpoints;
mod filters;

pub use client::HuggingFaceClient;
pub use types::{HFModel, HFModelFile, HFSearchResponse};
pub use filters::HFFilters;
```

**Verification:**
- [ ] Module compiles
- [ ] Exports are public
- [ ] No warnings

---

### 2. Define HuggingFace-Specific Types

**Create `src/huggingface/types.rs`:**
```rust
// TEAM-XXX: HuggingFace API response types

use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

/// HuggingFace model response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HFModel {
    /// Model ID (e.g., "meta-llama/Llama-2-7b-hf")
    pub id: String,
    
    /// Model name
    #[serde(default)]
    pub model_id: String,
    
    /// Author/organization
    #[serde(default)]
    pub author: String,
    
    /// Last modified timestamp
    #[serde(default)]
    pub last_modified: String,
    
    /// Download count
    #[serde(default)]
    pub downloads: u64,
    
    /// Like count
    #[serde(default)]
    pub likes: u64,
    
    /// Tags (e.g., ["llama", "text-generation"])
    #[serde(default)]
    pub tags: Vec<String>,
    
    /// Private model flag
    #[serde(default)]
    pub private: bool,
    
    /// Gated model flag (requires approval)
    #[serde(default)]
    pub gated: bool,
}

/// HuggingFace model file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFModelFile {
    /// Filename (e.g., "model-q4_0.gguf")
    pub rfilename: String,
    
    /// File size in bytes
    pub size: u64,
    
    /// LFS OID (for large files)
    #[serde(default)]
    pub lfs: Option<HFLfsInfo>,
}

/// LFS (Large File Storage) info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFLfsInfo {
    pub oid: String,
    pub size: u64,
    pub pointer_size: u64,
}

/// Search response with pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFSearchResponse {
    /// List of models
    pub models: Vec<HFModel>,
    
    /// Total count (for pagination)
    #[serde(default)]
    pub num_total_items: Option<u64>,
    
    /// Next page token
    #[serde(default)]
    pub next: Option<String>,
}

/// Convert HFModel to our Model type
impl From<HFModel> for crate::types::Model {
    fn from(hf: HFModel) -> Self {
        Self {
            id: hf.id.clone(),
            name: hf.model_id.clone(),
            description: format!("HuggingFace model: {}", hf.id),
            author: Some(hf.author),
            image_url: None, // HF doesn't provide images in list API
            tags: hf.tags,
            downloads: hf.downloads,
            likes: hf.likes,
            size: "Unknown".to_string(), // Need to fetch files for size
            source: crate::types::ModelSource::HuggingFace,
        }
    }
}
```

**Verification:**
- [ ] All types derive Serialize/Deserialize
- [ ] Conversion to Model type works
- [ ] Handles optional fields correctly

---

### 3. Implement API Endpoints

**Create `src/huggingface/endpoints.rs`:**
```rust
// TEAM-XXX: HuggingFace API endpoint builders

/// Base URL for HuggingFace API
pub const HF_API_BASE: &str = "https://huggingface.co/api";

/// Build models list URL
pub fn models_list_url(limit: Option<u32>, offset: Option<u32>) -> String {
    let mut url = format!("{}/models", HF_API_BASE);
    let mut params = Vec::new();
    
    if let Some(limit) = limit {
        params.push(format!("limit={}", limit));
    }
    if let Some(offset) = offset {
        params.push(format!("skip={}", offset));
    }
    
    if !params.is_empty() {
        url.push('?');
        url.push_str(&params.join("&"));
    }
    
    url
}

/// Build model search URL
pub fn models_search_url(query: &str, limit: Option<u32>) -> String {
    let mut url = format!("{}/models", HF_API_BASE);
    let mut params = vec![format!("search={}", urlencoding::encode(query))];
    
    if let Some(limit) = limit {
        params.push(format!("limit={}", limit));
    }
    
    url.push('?');
    url.push_str(&params.join("&"));
    url
}

/// Build model details URL
pub fn model_details_url(model_id: &str) -> String {
    format!("{}/models/{}", HF_API_BASE, model_id)
}

/// Build model files URL
pub fn model_files_url(model_id: &str) -> String {
    format!("https://huggingface.co/{}/tree/main", model_id)
}

/// Build download URL for a file
pub fn download_url(model_id: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model_id, filename
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_list_url() {
        let url = models_list_url(Some(10), Some(20));
        assert!(url.contains("limit=10"));
        assert!(url.contains("skip=20"));
    }

    #[test]
    fn test_search_url() {
        let url = models_search_url("llama", Some(5));
        assert!(url.contains("search=llama"));
        assert!(url.contains("limit=5"));
    }
}
```

**Add dependency to `Cargo.toml`:**
```toml
urlencoding = "2.1"
```

**Verification:**
- [ ] All URL builders work correctly
- [ ] Query parameters are properly encoded
- [ ] Tests pass

---

### 4. Implement Main Client

**Create `src/huggingface/client.rs`:**
```rust
// TEAM-XXX: HuggingFace API client implementation

use anyhow::{Context, Result};
use reqwest::Client;
use wasm_bindgen::prelude::*;

use super::endpoints;
use super::types::{HFModel, HFSearchResponse};
use crate::types::Model;

/// HuggingFace API client
#[wasm_bindgen]
pub struct HuggingFaceClient {
    client: Client,
    api_token: Option<String>,
}

#[wasm_bindgen]
impl HuggingFaceClient {
    /// Create new client without authentication
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_token: None,
        }
    }
    
    /// Create client with API token
    #[wasm_bindgen(js_name = withToken)]
    pub fn with_token(token: String) -> Self {
        Self {
            client: Client::new(),
            api_token: Some(token),
        }
    }
    
    /// List models with pagination
    #[wasm_bindgen(js_name = listModels)]
    pub async fn list_models(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<JsValue>, JsValue> {
        let url = endpoints::models_list_url(limit, offset);
        let models = self.fetch_models(&url).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        // Convert to JS values
        models.into_iter()
            .map(|m| serde_wasm_bindgen::to_value(&Model::from(m)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    /// Search models by query
    #[wasm_bindgen(js_name = searchModels)]
    pub async fn search_models(
        &self,
        query: String,
        limit: Option<u32>,
    ) -> Result<Vec<JsValue>, JsValue> {
        let url = endpoints::models_search_url(&query, limit);
        let models = self.fetch_models(&url).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        models.into_iter()
            .map(|m| serde_wasm_bindgen::to_value(&Model::from(m)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    /// Get model details
    #[wasm_bindgen(js_name = getModel)]
    pub async fn get_model(&self, model_id: String) -> Result<JsValue, JsValue> {
        let url = endpoints::model_details_url(&model_id);
        let model = self.fetch_model(&url).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        serde_wasm_bindgen::to_value(&Model::from(model))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// Internal methods (not exported to WASM)
impl HuggingFaceClient {
    /// Fetch models from URL
    async fn fetch_models(&self, url: &str) -> Result<Vec<HFModel>> {
        let mut request = self.client.get(url);
        
        // Add authorization header if token provided
        if let Some(token) = &self.api_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }
        
        let response = request
            .send()
            .await
            .context("Failed to send request to HuggingFace API")?;
        
        if !response.status().is_success() {
            anyhow::bail!(
                "HuggingFace API error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            );
        }
        
        let models: Vec<HFModel> = response
            .json()
            .await
            .context("Failed to parse HuggingFace API response")?;
        
        Ok(models)
    }
    
    /// Fetch single model
    async fn fetch_model(&self, url: &str) -> Result<HFModel> {
        let mut request = self.client.get(url);
        
        if let Some(token) = &self.api_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }
        
        let response = request
            .send()
            .await
            .context("Failed to send request")?;
        
        if !response.status().is_success() {
            anyhow::bail!("API error: {}", response.status());
        }
        
        let model: HFModel = response
            .json()
            .await
            .context("Failed to parse response")?;
        
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = HuggingFaceClient::new();
        assert!(client.api_token.is_none());
    }

    #[test]
    fn test_client_with_token() {
        let client = HuggingFaceClient::with_token("test_token".to_string());
        assert_eq!(client.api_token, Some("test_token".to_string()));
    }
}
```

**Verification:**
- [ ] Client compiles
- [ ] WASM bindings export correctly
- [ ] Error handling works
- [ ] Tests pass

---

### 5. Implement Filtering

**Create `src/huggingface/filters.rs`:**
```rust
// TEAM-XXX: HuggingFace filtering logic

use super::types::HFModel;
use crate::types::ModelFilters;

/// Apply filters to HuggingFace models
pub fn apply_filters(models: Vec<HFModel>, filters: &ModelFilters) -> Vec<HFModel> {
    models
        .into_iter()
        .filter(|model| {
            // Filter by tags
            if !filters.tags.is_empty() {
                let has_tag = filters.tags.iter().any(|tag| {
                    model.tags.iter().any(|model_tag| {
                        model_tag.to_lowercase().contains(&tag.to_lowercase())
                    })
                });
                if !has_tag {
                    return false;
                }
            }
            
            // Filter by minimum downloads
            if let Some(min_downloads) = filters.min_downloads {
                if model.downloads < min_downloads {
                    return false;
                }
            }
            
            // Filter by minimum likes
            if let Some(min_likes) = filters.min_likes {
                if model.likes < min_likes {
                    return false;
                }
            }
            
            // Filter out gated models if requested
            if !filters.include_gated && model.gated {
                return false;
            }
            
            true
        })
        .collect()
}

/// Sort models by criteria
pub fn sort_models(mut models: Vec<HFModel>, sort_by: &str) -> Vec<HFModel> {
    match sort_by {
        "downloads" => {
            models.sort_by(|a, b| b.downloads.cmp(&a.downloads));
        }
        "likes" => {
            models.sort_by(|a, b| b.likes.cmp(&a.likes));
        }
        "recent" => {
            models.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
        }
        _ => {} // No sorting
    }
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model(downloads: u64, likes: u64, tags: Vec<String>) -> HFModel {
        HFModel {
            id: "test/model".to_string(),
            model_id: "test-model".to_string(),
            author: "test".to_string(),
            last_modified: "2024-01-01".to_string(),
            downloads,
            likes,
            tags,
            private: false,
            gated: false,
        }
    }

    #[test]
    fn test_filter_by_downloads() {
        let models = vec![
            create_test_model(100, 10, vec![]),
            create_test_model(200, 20, vec![]),
        ];
        
        let filters = ModelFilters {
            tags: vec![],
            min_downloads: Some(150),
            min_likes: None,
            include_gated: true,
        };
        
        let filtered = apply_filters(models, &filters);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].downloads, 200);
    }

    #[test]
    fn test_sort_by_downloads() {
        let models = vec![
            create_test_model(100, 10, vec![]),
            create_test_model(300, 30, vec![]),
            create_test_model(200, 20, vec![]),
        ];
        
        let sorted = sort_models(models, "downloads");
        assert_eq!(sorted[0].downloads, 300);
        assert_eq!(sorted[1].downloads, 200);
        assert_eq!(sorted[2].downloads, 100);
    }
}
```

**Verification:**
- [ ] Filters work correctly
- [ ] Sorting works correctly
- [ ] Tests pass

---

### 6. Update Main Library

**Update `src/lib.rs`:**
```rust
// Add to modules
mod huggingface;

// Add to re-exports
pub use huggingface::HuggingFaceClient;
```

**Verification:**
- [ ] Module exports correctly
- [ ] No compilation errors

---

### 7. Write Integration Tests

**Create `tests/huggingface_tests.rs`:**
```rust
// TEAM-XXX: HuggingFace client integration tests

use marketplace_sdk::HuggingFaceClient;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_list_models() {
    let client = HuggingFaceClient::new();
    let result = client.list_models(Some(5), None).await;
    
    assert!(result.is_ok());
    let models = result.unwrap();
    assert!(models.len() <= 5);
}

#[wasm_bindgen_test]
async fn test_search_models() {
    let client = HuggingFaceClient::new();
    let result = client.search_models("llama".to_string(), Some(10)).await;
    
    assert!(result.is_ok());
    let models = result.unwrap();
    assert!(models.len() <= 10);
}

#[wasm_bindgen_test]
async fn test_get_model() {
    let client = HuggingFaceClient::new();
    let result = client.get_model("gpt2".to_string()).await;
    
    assert!(result.is_ok());
}
```

**Run tests:**
```bash
wasm-pack test --headless --firefox
wasm-pack test --headless --chrome
```

**Verification:**
- [ ] All tests pass in Firefox
- [ ] All tests pass in Chrome
- [ ] No console errors

---

## Acceptance Criteria

### Functionality
- [ ] Client can list models
- [ ] Client can search models
- [ ] Client can get model details
- [ ] Pagination works correctly
- [ ] Authentication with token works
- [ ] Error handling is robust

### Code Quality
- [ ] All code compiles without warnings
- [ ] All tests pass
- [ ] No `TODO` markers
- [ ] TEAM-XXX signatures on all files
- [ ] Follows Rust conventions (rustfmt, clippy)

### WASM Integration
- [ ] Client exports to JavaScript
- [ ] Methods are callable from JS
- [ ] Promises work correctly
- [ ] Error messages are helpful

### Performance
- [ ] API calls are efficient
- [ ] No unnecessary cloning
- [ ] WASM bundle size impact <500KB

---

## Testing Checklist

### Unit Tests
- [ ] Endpoint URL builders
- [ ] Type conversions
- [ ] Filtering logic
- [ ] Sorting logic

### Integration Tests
- [ ] List models (real API)
- [ ] Search models (real API)
- [ ] Get model details (real API)
- [ ] Error handling (invalid model ID)
- [ ] Rate limiting handling

### WASM Tests
- [ ] Client creation in browser
- [ ] Async methods work
- [ ] Error propagation to JS
- [ ] Type conversions work

---

## Common Issues & Solutions

### Issue: CORS errors in browser
**Solution:** HuggingFace API supports CORS. Check browser console for specific error.

### Issue: Rate limiting (429 errors)
**Solution:** 
- Implement exponential backoff
- Use API token for higher limits
- Cache responses

### Issue: Large response payloads
**Solution:**
- Use pagination (limit parameter)
- Filter on server side when possible
- Stream responses if needed

### Issue: Gated models return 401
**Solution:**
- Require API token for gated models
- Show clear error message to user
- Filter out gated models by default

---

## Next Steps

After completing Part 2:
- **Part 3:** Implement CivitAI client (similar structure)
- Test HuggingFace client in browser console
- Measure API response times
- Document rate limits

---

## References

- [HuggingFace Hub API](https://huggingface.co/docs/hub/api)
- [HuggingFace Models](https://huggingface.co/models)
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [reqwest WASM](https://docs.rs/reqwest/latest/reqwest/#wasm)
