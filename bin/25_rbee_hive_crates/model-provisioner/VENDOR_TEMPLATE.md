# Vendor Implementation Template

## Copy-Paste Template for New Vendors

All future vendors follow this exact pattern. Just copy, rename, and implement.

## Example: GitHubVendor

```rust
//! GitHub vendor implementation
//!
//! Downloads GGUF models from GitHub releases with:
//! - Cancellation support
//! - Progress tracking via DownloadTracker
//! - Proper narration context propagation

use crate::download_tracker::DownloadTracker;
use anyhow::Result;
use observability_narration_core::{context, n};
use rbee_hive_artifact_catalog::VendorSource;
use std::path::Path;
use tokio_util::sync::CancellationToken;

/// GitHub vendor for downloading models from releases
///
/// # Supported ID Formats
///
/// - `owner/repo@v1.0.0` - Specific release
/// - `owner/repo@latest` - Latest release
/// - `owner/repo@v1.0.0:model.gguf` - Specific asset
pub struct GitHubVendor {
    client: reqwest::Client,
}

impl std::fmt::Debug for GitHubVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GitHubVendor").field("client", &"<reqwest::Client>").finish()
    }
}

impl GitHubVendor {
    /// Create a new GitHub vendor
    ///
    /// # Errors
    ///
    /// Returns error if HTTP client initialization fails.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::new();
        Ok(Self { client })
    }

    /// Find GGUF asset in release
    async fn find_gguf_asset(
        &self,
        owner: &str,
        repo: &str,
        tag: &str,
        cancel_token: &CancellationToken,
    ) -> Result<String> {
        // Check for cancellation
        if cancel_token.is_cancelled() {
            return Err(anyhow::anyhow!("Cancelled"));
        }

        // Fetch release info from GitHub API
        let url = format!(
            "https://api.github.com/repos/{}/{}/releases/tags/{}",
            owner, repo, tag
        );

        let response = self.client
            .get(&url)
            .header("User-Agent", "rbee-hive")
            .send()
            .await?;

        // Parse assets and find .gguf file
        // ... implementation ...

        Ok("model.gguf".to_string())
    }
}

impl Default for GitHubVendor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| panic!("Failed to create GitHub vendor: {}", e))
    }
}

#[async_trait::async_trait]
impl VendorSource for GitHubVendor {
    async fn download(
        &self,
        id: &str,
        dest: &Path,
        job_id: &str,
        cancel_token: CancellationToken,
    ) -> Result<u64> {
        n!("github_download_start", "📥 Downloading model '{}' from GitHub", id);

        // Parse ID: "owner/repo@tag" or "owner/repo@tag:asset.gguf"
        let (repo_part, asset_name) = if let Some((repo, asset)) = id.split_once(':') {
            (repo, Some(asset.to_string()))
        } else {
            (id, None)
        };

        let parts: Vec<&str> = repo_part.split('@').collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!("Invalid GitHub ID format. Expected: owner/repo@tag"));
        }

        let repo_parts: Vec<&str> = parts[0].split('/').collect();
        if repo_parts.len() != 2 {
            return Err(anyhow::anyhow!("Invalid repo format. Expected: owner/repo"));
        }

        let owner = repo_parts[0];
        let repo = repo_parts[1];
        let tag = parts[1];

        // Determine asset name
        let asset_name = if let Some(name) = asset_name {
            name
        } else {
            n!("github_find_asset", "🔍 Looking for GGUF asset in release...");
            self.find_gguf_asset(owner, repo, tag, &cancel_token).await?
        };

        n!("github_download_file", "📥 Downloading asset: {}", asset_name);

        // Create download tracker for progress reporting
        let (tracker, _progress_rx) = DownloadTracker::new(job_id.to_string(), None);

        // Start heartbeat task with proper narration context
        let heartbeat_handle = {
            let job_id_clone = job_id.to_string();
            let asset_clone = asset_name.clone();
            let cancel_token_clone = cancel_token.clone();

            tokio::spawn(async move {
                // CRITICAL: Set up narration context for spawned task
                let ctx = context::NarrationContext::new().with_job_id(&job_id_clone);
                context::with_narration_context(ctx, async move {
                    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
                    interval.tick().await; // Skip first immediate tick

                    loop {
                        tokio::select! {
                            _ = cancel_token_clone.cancelled() => {
                                n!("github_download_cancelled", "❌ Download cancelled");
                                break;
                            }
                            _ = interval.tick() => {
                                n!(
                                    "github_download_heartbeat",
                                    "⏳ Still downloading {} from GitHub...",
                                    asset_clone
                                );
                            }
                        }
                    }
                })
                .await
            })
        };

        // Download the asset with cancellation support
        let download_url = format!(
            "https://github.com/{}/{}/releases/download/{}/{}",
            owner, repo, tag, asset_name
        );

        let download_result = tokio::select! {
            _ = cancel_token.cancelled() => {
                Err(anyhow::anyhow!("Download cancelled by user"))
            }
            result = self.download_file(&download_url, dest) => {
                result
            }
        };

        // Stop heartbeat
        heartbeat_handle.abort();

        let size = download_result?;

        #[allow(clippy::cast_precision_loss)]
        let size_gb = size as f64 / 1_000_000_000.0_f64;
        n!("github_download_complete", "✅ Model ready: {} ({:.2} GB)", asset_name, size_gb);

        Ok(size)
    }

    fn supports(&self, id: &str) -> bool {
        // GitHub IDs must have format: "owner/repo@tag" or "owner/repo@tag:asset.gguf"
        if id.is_empty() || id.trim().is_empty() {
            return false;
        }

        let base_id = id.split(':').next().unwrap_or(id);

        // Must contain '/' and '@'
        base_id.contains('/') && base_id.contains('@') && !base_id.contains("://")
    }

    fn name(&self) -> &str {
        "GitHub"
    }
}

impl GitHubVendor {
    /// Download file from URL
    async fn download_file(&self, url: &str, dest: &Path) -> Result<u64> {
        let response = self.client.get(url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("GitHub download failed: {}", response.status()));
        }

        // Create destination directory
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Stream to file
        let bytes = response.bytes().await?;
        tokio::fs::write(dest, &bytes).await?;

        Ok(bytes.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_supports_valid_ids() {
        let vendor = GitHubVendor::new().expect("Failed to create vendor");

        assert!(vendor.supports("ggerganov/llama.cpp@b1234"));
        assert!(vendor.supports("owner/repo@v1.0.0"));
        assert!(vendor.supports("owner/repo@latest:model.gguf"));
    }

    #[test]
    fn test_vendor_rejects_invalid_ids() {
        let vendor = GitHubVendor::new().expect("Failed to create vendor");

        assert!(!vendor.supports(""));
        assert!(!vendor.supports("   "));
        assert!(!vendor.supports("no-slash"));
        assert!(!vendor.supports("owner/repo")); // No tag
        assert!(!vendor.supports("https://github.com/owner/repo")); // URL
    }
}
```

## Key Points (Copy These Patterns)

### 1. ✅ Always Create DownloadTracker
```rust
let (tracker, _progress_rx) = DownloadTracker::new(job_id.to_string(), total_size);
```

### 2. ✅ Always Start Heartbeat with Proper Context
```rust
let heartbeat_handle = {
    let job_id_clone = job_id.to_string();
    let name_clone = name.clone();
    let cancel_token_clone = cancel_token.clone();

    tokio::spawn(async move {
        let ctx = context::NarrationContext::new().with_job_id(&job_id_clone);
        context::with_narration_context(ctx, async move {
            // Heartbeat loop with cancellation
        }).await
    })
};
```

### 3. ✅ Always Use tokio::select! for Cancellation
```rust
let result = tokio::select! {
    _ = cancel_token.cancelled() => {
        Err(anyhow::anyhow!("Cancelled by user"))
    }
    result = actual_download() => {
        result
    }
};
```

### 4. ✅ Always Stop Heartbeat
```rust
heartbeat_handle.abort();
```

### 5. ✅ Always Check Cancellation in Loops
```rust
for item in items {
    if cancel_token.is_cancelled() {
        return Err(anyhow::anyhow!("Cancelled"));
    }
    // ... process item ...
}
```

## Checklist for New Vendors

- [ ] Implement `VendorSource` trait
- [ ] Add `CancellationToken` parameter to `download()`
- [ ] Create `DownloadTracker` for progress
- [ ] Start heartbeat with proper narration context
- [ ] Use `tokio::select!` for cancellable operations
- [ ] Stop heartbeat when done
- [ ] Check cancellation in loops
- [ ] Add `supports()` validation
- [ ] Add unit tests for `supports()`
- [ ] Document supported ID formats

## Benefits of This Pattern

✅ **Reusable** - Same pattern for all vendors  
✅ **Cancellable** - Works out of the box  
✅ **Progress** - DownloadTracker ready for real-time updates  
✅ **SSE Routing** - Proper narration context  
✅ **Testable** - Easy to mock CancellationToken  
✅ **Maintainable** - One pattern to learn  

**Just copy, rename, and implement!**
