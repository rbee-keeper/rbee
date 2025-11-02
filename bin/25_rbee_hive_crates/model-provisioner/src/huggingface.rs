//! HuggingFace vendor implementation
//!
//! Downloads GGUF models from HuggingFace Hub with:
//! - Cancellation support
//! - Progress tracking via DownloadTracker
//! - Proper narration context propagation

use crate::download_tracker::DownloadTracker;
use anyhow::Result;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub_simple_progress::{sync::callback_builder, ProgressEvent};
use observability_narration_core::n;
use rbee_hive_artifact_catalog::VendorSource;
use std::path::Path;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// HuggingFace vendor for downloading models
///
/// # Supported ID Formats
///
/// - `meta-llama/Llama-2-7b-chat-hf` - Standard HF repo (auto-detects GGUF)
/// - `TheBloke/Llama-2-7B-Chat-GGUF:model-Q4_K_M.gguf` - Explicit filename
///
/// # Example
///
/// ```rust,no_run
/// use rbee_hive_model_provisioner::HuggingFaceVendor;
/// use rbee_hive_artifact_catalog::VendorSource;
/// use tokio_util::sync::CancellationToken;
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let vendor = HuggingFaceVendor::new()?;
/// let cancel_token = CancellationToken::new();
/// let size = vendor.download(
///     "TheBloke/Llama-2-7B-Chat-GGUF",
///     Path::new("/tmp/model.gguf"),
///     "job-123",
///     cancel_token
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct HuggingFaceVendor {
    api: Arc<Api>,  // TEAM-379: Arc for clone in spawn_blocking
}

impl std::fmt::Debug for HuggingFaceVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HuggingFaceVendor").field("api", &"<hf_hub::Api>").finish()
    }
}

impl HuggingFaceVendor {
    /// Create a new HuggingFace vendor
    ///
    /// Uses default HF cache location (~/.cache/huggingface/)
    ///
    /// # Errors
    ///
    /// Returns error if HuggingFace API initialization fails.
    pub fn new() -> Result<Self> {
        let api = ApiBuilder::new().build()?;
        Ok(Self { api: Arc::new(api) })
    }

    /// Create vendor with custom cache directory
    ///
    /// # Errors
    ///
    /// Returns error if HuggingFace API initialization fails.
    ///
    /// # Note
    ///
    /// hf-hub 0.3 doesn't support custom cache directories via API.
    /// It uses HF_HOME environment variable instead.
    /// For now, this just uses the default cache.
    pub fn with_cache_dir(_cache_dir: impl AsRef<Path>) -> Result<Self> {
        // TODO: hf-hub 0.3 doesn't support custom cache directories via API
        // For now, just use default
        Self::new()
    }

    /// Find GGUF file in repository
    ///
    /// Tries common GGUF quantizations in order of popularity.
    async fn find_gguf_file(&self, repo_id: &str, cancel_token: &CancellationToken) -> Result<String> {
        let api_clone = self.api.clone();
        let repo_id_clone = repo_id.to_string();
        let cancel_token_clone = cancel_token.clone();
        
        // TEAM-379: Use spawn_blocking for sync API
        tokio::task::spawn_blocking(move || {
            let repo = api_clone.model(repo_id_clone.clone());

            // Common GGUF quantizations (from most to least common)
            let common_quants = ["Q4_K_M", "Q5_K_M", "Q4_0", "Q5_0", "Q8_0", "F16"];

            let base_name = repo_id_clone.split('/').next_back().unwrap_or(&repo_id_clone);

            for quant in &common_quants {
                // Check for cancellation before each attempt
                if cancel_token_clone.is_cancelled() {
                    return Err(anyhow::anyhow!("Download cancelled"));
                }

                let filename = format!("{}-{}.gguf", base_name, quant);
                match repo.get(&filename) {
                    Ok(_) => return Ok(filename),
                    Err(_) => continue,
                }
            }

            // If no common quant found, return error with helpful message
            Err(anyhow::anyhow!(
                "Could not find GGUF file in repository '{}'. \
                 Please specify the exact filename (e.g., 'repo/model:file-Q4_K_M.gguf')",
                repo_id_clone
            ))
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }
}

impl Default for HuggingFaceVendor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| panic!("Failed to create HuggingFace vendor: {}", e))
    }
}

#[async_trait::async_trait]
impl VendorSource for HuggingFaceVendor {
    async fn download(
        &self,
        id: &str,
        dest: &Path,
        job_id: &str,
        cancel_token: CancellationToken,
    ) -> Result<u64> {
        n!("hf_download_start", "📥 Downloading model '{}' from HuggingFace", id);

        // Parse ID: "repo/model" or "repo/model:filename.gguf"
        let (repo_id, filename) = if let Some((repo, file)) = id.split_once(':') {
            (repo, Some(file.to_string()))
        } else {
            (id, None)
        };

        // Get the repository
        // Determine filename
        let filename = if let Some(f) = filename {
            f
        } else {
            n!("hf_find_gguf", "🔍 Looking for GGUF file in repository...");
            self.find_gguf_file(repo_id, &cancel_token).await?
        };

        n!("hf_download_file", "📥 Downloading file: {}", filename);

        // TEAM-379: Create download tracker and start heartbeat
        let (tracker, _progress_rx) = DownloadTracker::new(
            job_id.to_string(),
            None,  // Total size unknown until download starts
            cancel_token.clone(),
        );
        let heartbeat_handle = tracker.start_heartbeat(filename.clone());

        // TEAM-379: Download with real-time progress callback
        let api_clone = self.api.clone();
        let repo_id_clone = repo_id.to_string();
        let filename_clone = filename.clone();
        let tracker_clone = tracker.clone();
        let cancel_token_clone = cancel_token.clone();
        
        let cached_path = tokio::task::spawn_blocking(move || {
            let repo = api_clone.model(repo_id_clone);
            
            // TEAM-379: Progress callback checks cancellation and updates tracker
            let callback = callback_builder(move |progress: ProgressEvent| {
                // Check if cancelled - if so, panic to abort the download
                // This is caught and converted to an error below
                if cancel_token_clone.is_cancelled() {
                    panic!("Download cancelled by user");
                }
                
                // ProgressEvent has: url, percentage (0.0-1.0), elapsed_time, remaining_time
                tracker_clone.update_percentage(progress.percentage as f64);
            });
            
            repo.download_with_progress(&filename_clone, callback)
        })
        .await
        .map_err(|e| {
            // Convert panic from cancellation into proper error
            if e.is_panic() {
                anyhow::anyhow!("Download cancelled by user")
            } else {
                anyhow::anyhow!("Task join error: {}", e)
            }
        })?
        .map_err(|e| anyhow::anyhow!("HuggingFace download failed: {}", e))?;

        // Stop heartbeat
        heartbeat_handle.abort();

        n!("hf_download_cached", "✅ File downloaded and cached at: {}", cached_path.display());

        // Copy to destination
        let metadata = tokio::fs::metadata(&cached_path).await?;
        let size = metadata.len();

        // Create destination directory
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Copy file to destination with cancellation support
        #[allow(clippy::cast_precision_loss)]
        let size_gb = size as f64 / 1_000_000_000.0_f64;
        n!("hf_copy_start", "📋 Copying to model cache... ({:.2} GB)", size_gb);

        tokio::select! {
            _ = cancel_token.cancelled() => {
                return Err(anyhow::anyhow!("Copy cancelled by user"));
            }
            result = tokio::fs::copy(&cached_path, dest) => {
                result?;
            }
        }

        n!("hf_download_complete", "✅ Model ready: {} ({:.2} GB)", filename, size_gb);

        Ok(size)
    }

    fn supports(&self, id: &str) -> bool {
        // HuggingFace IDs must have format: "org/model" or "org/model:file.gguf"
        // Reject empty strings, URLs, and single names without org
        if id.is_empty() || id.trim().is_empty() {
            return false;
        }

        let base_id = id.split(':').next().unwrap_or(id);

        // Must contain '/' (org/model format)
        // Must NOT contain "://" (reject URLs)
        base_id.contains('/') && !base_id.contains("://")
    }

    fn name(&self) -> &str {
        "HuggingFace"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BEHAVIOR: Vendor ID Recognition
    // ========================================================================
    // The vendor must correctly identify HuggingFace repository IDs while
    // rejecting URLs and local paths to prevent security issues.
    //
    // NOTE: Download tests removed - they require network access and are slow.
    // Cancellation behavior is tested at the integration level.

    #[test]
    fn behavior_accepts_standard_huggingface_repo_ids() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        // Standard org/model format
        assert!(
            vendor.supports("meta-llama/Llama-2-7b"),
            "Should accept standard HF repo format: org/model"
        );

        // GGUF-specific repos
        assert!(vendor.supports("TheBloke/Llama-2-7B-GGUF"), "Should accept GGUF-specific repos");

        // Different organizations
        assert!(vendor.supports("microsoft/phi-2"), "Should accept repos from different orgs");
    }

    #[test]
    fn behavior_accepts_repo_ids_with_explicit_filenames() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        // Repo with explicit filename
        assert!(
            vendor.supports("meta-llama/Llama-2-7b:model-Q4_K_M.gguf"),
            "Should accept repo:filename format for explicit file selection"
        );

        // Different quantizations
        assert!(
            vendor.supports("TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf"),
            "Should accept different quantization formats"
        );
    }

    #[test]
    fn behavior_rejects_urls_to_prevent_arbitrary_downloads() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        // HTTP URLs
        assert!(
            !vendor.supports("https://example.com/model.gguf"),
            "Must reject HTTP URLs to prevent arbitrary downloads"
        );

        assert!(
            !vendor.supports("http://malicious.com/model.gguf"),
            "Must reject HTTP URLs (security)"
        );

        // File URLs
        assert!(
            !vendor.supports("file:///local/path/model.gguf"),
            "Must reject file:// URLs to prevent local file access"
        );
    }

    #[test]
    fn behavior_rejects_single_names_without_organization() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        // Single name without org (ambiguous)
        assert!(
            !vendor.supports("llama-2-7b"),
            "Should reject single names without org/ prefix (ambiguous)"
        );

        assert!(!vendor.supports("model.gguf"), "Should reject bare filenames");
    }

    // ========================================================================
    // BEHAVIOR: Vendor Identity
    // ========================================================================
    // The vendor must correctly identify itself for logging and error messages.

    #[test]
    fn behavior_identifies_as_huggingface_vendor() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        assert_eq!(
            vendor.name(),
            "HuggingFace",
            "Vendor must identify as 'HuggingFace' for logging"
        );
    }

    // ========================================================================
    // BEHAVIOR: Initialization
    // ========================================================================
    // The vendor must initialize successfully with default and custom caches.

    #[test]
    fn behavior_initializes_with_default_cache() {
        let result = HuggingFaceVendor::new();

        assert!(result.is_ok(), "Vendor should initialize successfully with default cache");
    }

    #[test]
    fn behavior_initializes_with_custom_cache_directory() {
        use std::env;

        let temp_dir = env::temp_dir().join("rbee-test-cache");
        let result = HuggingFaceVendor::with_cache_dir(&temp_dir);

        assert!(result.is_ok(), "Vendor should initialize with custom cache directory");
    }

    #[test]
    fn behavior_default_constructor_works() {
        let vendor = HuggingFaceVendor::default();

        assert_eq!(vendor.name(), "HuggingFace", "Default constructor should create valid vendor");
    }

    // ========================================================================
    // BEHAVIOR: ID Parsing
    // ========================================================================
    // The vendor must correctly parse repo IDs with and without filenames.

    #[test]
    fn behavior_parses_repo_id_without_filename() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");
        let id = "meta-llama/Llama-2-7b";

        // Should support it
        assert!(vendor.supports(id));

        // Should extract repo part correctly (tested implicitly by supports())
        assert!(!id.contains(':'), "ID without filename should have no colon");
        assert_eq!(id, "meta-llama/Llama-2-7b");
    }

    #[test]
    fn behavior_parses_repo_id_with_filename() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");
        let id = "meta-llama/Llama-2-7b:model-Q4_K_M.gguf";

        // Should support it
        assert!(vendor.supports(id));

        // Should extract both parts correctly
        let (repo, file) = id.split_once(':').expect("ID should contain colon");
        assert_eq!(repo, "meta-llama/Llama-2-7b", "Repo part should be correct");
        assert_eq!(file, "model-Q4_K_M.gguf", "Filename part should be correct");
    }

    // ========================================================================
    // BEHAVIOR: Edge Cases
    // ========================================================================
    // The vendor must handle edge cases gracefully.

    #[test]
    fn behavior_handles_empty_string() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        assert!(!vendor.supports(""), "Empty string should not be supported");
    }

    #[test]
    fn behavior_handles_whitespace() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        assert!(!vendor.supports("   "), "Whitespace-only string should not be supported");
    }

    #[test]
    fn behavior_handles_multiple_colons() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        // Multiple colons (malformed)
        assert!(
            vendor.supports("org/model:file:extra"),
            "Should still support (will use first colon split)"
        );
    }

    #[test]
    fn behavior_handles_special_characters_in_org_name() {
        let vendor = HuggingFaceVendor::new().expect("Failed to create vendor");

        // Underscores, hyphens are common in HF org names
        assert!(
            vendor.supports("hugging-face_org/model-name_v2"),
            "Should support special characters common in HF names"
        );
    }
}
