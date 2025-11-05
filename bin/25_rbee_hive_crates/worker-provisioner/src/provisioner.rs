//! Worker provisioner implementation
//!
//! TEAM-402: Refactored from worker_install.rs to follow ArtifactProvisioner pattern
//!
//! Coordinates catalog client, PKGBUILD parsing, source fetching, and building
//! to provision WorkerBinary artifacts.

use anyhow::{Context, Result};
use observability_narration_core::n;
use rbee_hive_artifact_catalog::ArtifactProvisioner;
// TEAM-402: Import types from artifacts-contract instead of worker-catalog
use artifacts_contract::{WorkerBinary, WorkerType, Platform};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

use crate::catalog_client::{CatalogClient, WorkerMetadata};
use crate::pkgbuild::{PkgBuild, PkgBuildExecutor, fetch_sources};

/// Worker provisioner
///
/// Downloads/builds workers from PKGBUILDs and creates WorkerBinary artifacts.
///
/// # Example
///
/// ```rust,no_run
/// use rbee_hive_worker_provisioner::WorkerProvisioner;
/// use rbee_hive_artifact_catalog::ArtifactProvisioner;
/// use tokio_util::sync::CancellationToken;
///
/// # async fn example() -> anyhow::Result<()> {
/// let provisioner = WorkerProvisioner::new()?;
/// let cancel_token = CancellationToken::new();
/// let worker = provisioner.provision(
///     "llm-worker-rbee-cpu",
///     "job-123",
///     cancel_token
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct WorkerProvisioner {
    catalog_client: CatalogClient,
    cache_dir: PathBuf,
}

impl std::fmt::Debug for WorkerProvisioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerProvisioner")
            .field("cache_dir", &self.cache_dir)
            .finish()
    }
}

impl WorkerProvisioner {
    /// Create a new worker provisioner
    ///
    /// Workers are stored in:
    /// - Linux/Mac: ~/.cache/rbee/workers/
    /// - Windows: %LOCALAPPDATA%\rbee\workers\
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("workers");
        
        let catalog_client = CatalogClient::new()?;
        
        Ok(Self {
            catalog_client,
            cache_dir,
        })
    }
    
    /// Create provisioner with custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        let catalog_client = CatalogClient::new()?;
        let workers_cache = cache_dir.join("workers");
        Ok(Self {
            catalog_client,
            cache_dir: workers_cache,
        })
    }
    
    /// Get worker directory path
    #[allow(dead_code)]  // TEAM-420: Reserved for future use
    fn worker_dir(&self, worker_id: &str) -> PathBuf {
        // Sanitize worker ID for filesystem
        let safe_id = worker_id.replace('/', "-").replace(':', "-");
        self.cache_dir.join(safe_id)
    }
    
    /// Check platform compatibility
    fn check_platform_compatibility(&self, worker: &WorkerMetadata) -> Result<()> {
        let current_os = std::env::consts::OS;
        let current_arch = std::env::consts::ARCH;
        
        // Check OS
        if !worker.platforms.iter().any(|p| p == current_os) {
            anyhow::bail!(
                "Platform incompatible. Worker requires: {:?}, Current platform: {}",
                worker.platforms,
                current_os
            );
        }
        
        // Check architecture
        if !worker.architectures.iter().any(|a| a == current_arch) {
            anyhow::bail!(
                "Architecture incompatible. Worker requires: {:?}, Current architecture: {}",
                worker.architectures,
                current_arch
            );
        }
        
        n!("platform_check", "✓ Platform: {}", current_os);
        n!("arch_check", "✓ Architecture: {}", current_arch);
        
        Ok(())
    }
    
    /// Create temp directories for build
    fn create_temp_directories(&self, worker_id: &str) -> Result<PathBuf> {
        let temp_base = std::env::temp_dir().join("worker-install").join(worker_id);
        
        // Create directories
        std::fs::create_dir_all(&temp_base)?;
        std::fs::create_dir_all(temp_base.join("src"))?;
        std::fs::create_dir_all(temp_base.join("pkg"))?;
        
        Ok(temp_base)
    }
    
    /// Determine where to install the binary
    ///
    /// TEAM-388: Try /usr/local/bin first, fall back to ~/.local/bin
    fn determine_install_directory(&self) -> Result<PathBuf> {
        let system_dir = PathBuf::from("/usr/local/bin");
        
        // Check if we can write to /usr/local/bin
        if system_dir.exists() {
            // Try to create a test file
            let test_file = system_dir.join(".rbee-write-test");
            if std::fs::write(&test_file, b"test").is_ok() {
                let _ = std::fs::remove_file(&test_file);
                return Ok(system_dir);
            }
        }
        
        // Fall back to user-local directory
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("Could not determine home directory")?;
        
        let user_dir = PathBuf::from(home).join(".local").join("bin");
        
        // Create if it doesn't exist
        if !user_dir.exists() {
            std::fs::create_dir_all(&user_dir)
                .context(format!("Failed to create {}", user_dir.display()))?;
        }
        
        Ok(user_dir)
    }
    
    /// Install binary to determined directory
    fn install_binary(
        &self,
        temp_dir: &Path,
        pkgbuild: &PkgBuild,
        install_dir: &Path,
    ) -> Result<PathBuf> {
        let pkg_dir = temp_dir.join("pkg");
        let binary_name = &pkgbuild.pkgname;
        
        // Find binary in pkg directory
        let binary_src = pkg_dir
            .join("usr")
            .join("local")
            .join("bin")
            .join(binary_name);
        
        if !binary_src.exists() {
            anyhow::bail!(
                "Binary '{}' not found in package directory",
                binary_src.display()
            );
        }
        
        let binary_dest = install_dir.join(binary_name);
        
        // Copy binary
        std::fs::copy(&binary_src, &binary_dest).context(format!(
            "Failed to install binary to {}",
            binary_dest.display()
        ))?;
        
        // Set executable permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&binary_dest)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&binary_dest, perms)?;
        }
        
        n!(
            "install_path",
            "✓ Binary installed: {}",
            binary_dest.display()
        );
        
        Ok(binary_dest)
    }
    
    /// Cleanup temp directories
    fn cleanup_temp_directories(&self, temp_dir: &PathBuf) -> Result<()> {
        if temp_dir.exists() {
            std::fs::remove_dir_all(temp_dir).context("Failed to cleanup temp directories")?;
        }
        Ok(())
    }
    
    /// Determine worker type from worker_id
    /// TEAM-404: Updated to use simplified WorkerType enum
    fn determine_worker_type(&self, worker_id: &str) -> Result<WorkerType> {
        if worker_id.contains("cpu") {
            Ok(WorkerType::Cpu)
        } else if worker_id.contains("cuda") {
            Ok(WorkerType::Cuda)
        } else if worker_id.contains("metal") {
            Ok(WorkerType::Metal)
        } else {
            anyhow::bail!("Unknown worker type for worker_id: {}", worker_id);
        }
    }
}

impl Default for WorkerProvisioner {
    fn default() -> Self {
        Self::new().expect("Failed to create worker provisioner")
    }
}

#[async_trait::async_trait]
impl ArtifactProvisioner<WorkerBinary> for WorkerProvisioner {
    async fn provision(
        &self,
        id: &str,
        _job_id: &str,
        cancel_token: CancellationToken,
    ) -> Result<WorkerBinary> {
        n!("provision_start", "🚀 Starting worker provisioning: {}", id);
        
        // TEAM-388: Check cancellation before starting
        if cancel_token.is_cancelled() {
            n!("install_cancelled", "❌ Installation cancelled before start");
            anyhow::bail!("Installation cancelled");
        }
        
        // 1. Fetch worker metadata from catalog
        n!("fetch_metadata", "📦 Fetching worker metadata from catalog...");
        let worker = self.catalog_client.fetch_metadata(id).await?;
        n!(
            "fetch_metadata_ok",
            "✓ Worker: {} v{}",
            worker.name,
            worker.version
        );
        
        // 2. Check platform compatibility
        n!("check_platform", "🔍 Checking platform compatibility...");
        self.check_platform_compatibility(&worker)?;
        n!("check_platform_ok", "✓ Platform compatible");
        
        // 3. Download PKGBUILD
        n!("download_pkgbuild", "📄 Downloading PKGBUILD...");
        let pkgbuild_content = self.catalog_client.download_pkgbuild(id).await?;
        n!(
            "download_pkgbuild_ok",
            "✓ PKGBUILD downloaded ({} bytes)",
            pkgbuild_content.len()
        );
        
        // 4. Parse PKGBUILD
        n!("parse_pkgbuild", "🔍 Parsing PKGBUILD...");
        let pkgbuild = PkgBuild::parse(&pkgbuild_content)?;
        n!(
            "parse_pkgbuild_ok",
            "✓ Parsed: pkgname={}, pkgver={}",
            pkgbuild.pkgname,
            pkgbuild.pkgver
        );
        
        // 5. Create temp directories
        n!("create_temp", "📁 Creating temporary directories...");
        let temp_dir = self.create_temp_directories(id)?;
        n!("create_temp_ok", "✓ Temp directory: {}", temp_dir.display());
        
        // 6. Preflight check - determine install directory before building
        n!("preflight_check", "🔍 Checking installation permissions...");
        let install_dir = self.determine_install_directory()?;
        n!("preflight_ok", "✓ Will install to: {}", install_dir.display());
        
        // 7. Fetch sources (git clone, etc.)
        n!("fetch_sources", "📦 Fetching sources from PKGBUILD...");
        let srcdir = temp_dir.join("src");
        // TEAM-402: Use architecture-specific sources if available
        let sources = pkgbuild.get_sources_for_arch();
        fetch_sources(&sources, &srcdir).await?;
        n!("fetch_sources_ok", "✓ Sources fetched to: {}", srcdir.display());
        
        // 8. Execute build() - THIS IS THE LONG-RUNNING OPERATION
        n!("build_start", "🏗️  Starting build phase (cancellable)...");
        let executor = PkgBuildExecutor::new(
            temp_dir.join("src"),
            temp_dir.join("pkg"),
            temp_dir.clone(),
        );
        
        // TEAM-388: Build with cancellation support
        if let Err(e) = executor
            .build_with_cancellation(&pkgbuild, cancel_token.clone(), |line| {
                n!("build_output", "{}", line);
            })
            .await
        {
            // Check if it was cancelled
            if cancel_token.is_cancelled() {
                n!("build_cancelled", "❌ Build cancelled by user");
                self.cleanup_temp_directories(&temp_dir).ok();
                anyhow::bail!("Build cancelled");
            }
            n!("build_failed", "❌ Build failed: {}", e);
            return Err(e.into());
        }
        n!("build_complete", "✓ Build complete");
        
        // 9. Execute package()
        n!("package_start", "📦 Starting package phase...");
        if let Err(e) = executor
            .package(&pkgbuild, |line| {
                n!("package_output", "{}", line);
            })
            .await
        {
            n!("package_failed", "❌ Package failed: {}", e);
            return Err(e.into());
        }
        n!("package_complete", "✓ Package complete");
        
        // 10. Install binary
        n!("install_binary", "💾 Installing binary...");
        let binary_path = self.install_binary(&temp_dir, &pkgbuild, &install_dir)?;
        n!("install_binary_ok", "✓ Binary installed to: {}", binary_path.display());
        
        // 11. Create WorkerBinary artifact
        n!("create_artifact", "📝 Creating WorkerBinary artifact...");
        let worker_type = self.determine_worker_type(id)?;
        let platform = Platform::current();
        let size = std::fs::metadata(&binary_path)?.len();
        let artifact_id = format!("{}-{}-{:?}", id, pkgbuild.pkgver, platform).to_lowercase();
        
        let worker_binary = WorkerBinary::new(
            artifact_id,
            worker_type,
            platform,
            binary_path.clone(),
            size,
            pkgbuild.pkgver.clone(),
        );
        n!("create_artifact_ok", "✓ WorkerBinary artifact created");
        
        // 12. Cleanup
        n!("cleanup", "🧹 Cleaning up temp files...");
        self.cleanup_temp_directories(&temp_dir)?;
        n!("cleanup_ok", "✓ Cleanup complete");
        
        n!("provision_complete", "✅ Worker provisioning complete!");
        Ok(worker_binary)
    }
    
    fn supports(&self, id: &str) -> bool {
        // Support all worker IDs that contain known worker types
        id.contains("cpu") || id.contains("cuda") || id.contains("metal")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_provisioner_creation() {
        let provisioner = WorkerProvisioner::new().unwrap();
        assert!(provisioner.cache_dir.ends_with("workers"));
    }
    
    #[test]
    fn test_supports_cpu_worker() {
        let provisioner = WorkerProvisioner::new().unwrap();
        assert!(provisioner.supports("llm-worker-rbee-cpu"));
    }
    
    #[test]
    fn test_supports_cuda_worker() {
        let provisioner = WorkerProvisioner::new().unwrap();
        assert!(provisioner.supports("llm-worker-rbee-cuda"));
    }
    
    #[test]
    fn test_supports_metal_worker() {
        let provisioner = WorkerProvisioner::new().unwrap();
        assert!(provisioner.supports("llm-worker-rbee-metal"));
    }
    
    #[test]
    fn test_does_not_support_unknown_worker() {
        let provisioner = WorkerProvisioner::new().unwrap();
        assert!(!provisioner.supports("unknown-worker"));
    }
    
    #[test]
    fn test_determine_worker_type_cpu() {
        let provisioner = WorkerProvisioner::new().unwrap();
        let worker_type = provisioner.determine_worker_type("llm-worker-rbee-cpu").unwrap();
        assert_eq!(worker_type, WorkerType::Cpu);
    }
    
    #[test]
    fn test_determine_worker_type_cuda() {
        let provisioner = WorkerProvisioner::new().unwrap();
        let worker_type = provisioner.determine_worker_type("llm-worker-rbee-cuda").unwrap();
        assert_eq!(worker_type, WorkerType::Cuda);
    }
    
    #[test]
    fn test_determine_worker_type_metal() {
        let provisioner = WorkerProvisioner::new().unwrap();
        let worker_type = provisioner.determine_worker_type("llm-worker-rbee-metal").unwrap();
        assert_eq!(worker_type, WorkerType::Metal);
    }
}
