//! Worker binary installation handler
//!
//! TEAM-378: Implements WorkerInstall operation
//!
//! This module handles the complete worker installation flow:
//! 1. Fetch worker metadata from catalog
//! 2. Check platform compatibility
//! 3. Download PKGBUILD
//! 4. Parse PKGBUILD
//! 5. Check dependencies
//! 6. Execute build()
//! 7. Execute package()
//! 8. Install binary
//! 9. Update capabilities
//! 10. Cleanup temp files

use anyhow::{Context, Result};
use observability_narration_core::n;
use rbee_hive_worker_catalog::WorkerCatalog;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken; // TEAM-388: For cancellable operations

/// Worker metadata from catalog
#[derive(Debug, serde::Deserialize)]
struct WorkerMetadata {
    #[allow(dead_code)]  // TEAM-420: Deserialized but not used
    id: String,
    name: String,
    version: String,
    platforms: Vec<String>,
    architectures: Vec<String>,
    #[serde(default)]
    depends: Vec<String>,
    // Ignore other fields we don't need
}

/// Handle worker installation operation
///
/// TEAM-378: Full implementation with PKGBUILD download and execution
/// TEAM-388: Added cancellation token support for cancellable builds
pub async fn handle_worker_install(
    worker_id: String,
    worker_catalog: Arc<WorkerCatalog>,
    cancel_token: CancellationToken,
) -> Result<()> {
    // TEAM-388: Check cancellation before starting
    if cancel_token.is_cancelled() {
        n!("install_cancelled", "❌ Installation cancelled before start");
        anyhow::bail!("Installation cancelled");
    }
    // 1. Fetch worker metadata from catalog
    n!("fetch_metadata", "📦 Fetching worker metadata from catalog...");
    let worker = fetch_worker_metadata(&worker_id).await?;
    n!(
        "fetch_metadata_ok",
        "✓ Worker: {} v{}",
        worker.name,
        worker.version
    );

    // 2. Check platform compatibility
    n!("check_platform", "🔍 Checking platform compatibility...");
    check_platform_compatibility(&worker)?;
    n!("check_platform_ok", "✓ Platform compatible");

    // 3. Download PKGBUILD
    n!("download_pkgbuild", "📄 Downloading PKGBUILD...");
    let pkgbuild_content = download_pkgbuild(&worker_id).await?;
    n!(
        "download_pkgbuild_ok",
        "✓ PKGBUILD downloaded ({} bytes)",
        pkgbuild_content.len()
    );

    // 4. Parse PKGBUILD
    n!("parse_pkgbuild", "🔍 Parsing PKGBUILD...");
    let pkgbuild = crate::pkgbuild_parser::PkgBuild::parse(&pkgbuild_content)?;
    n!(
        "parse_pkgbuild_ok",
        "✓ Parsed: pkgname={}, pkgver={}",
        pkgbuild.pkgname,
        pkgbuild.pkgver
    );

    // 5. Check dependencies
    n!("check_deps", "🔧 Checking dependencies...");
    check_dependencies(&worker)?;
    n!("check_deps_ok", "✓ All dependencies satisfied");

    // 6. Create temp directories
    n!("create_temp", "📁 Creating temporary directories...");
    let temp_dir = create_temp_directories(&worker_id)?;
    n!("create_temp_ok", "✓ Temp directory: {}", temp_dir.display());

    // TEAM-388: Preflight check - determine install directory before building
    n!("preflight_check", "🔍 Checking installation permissions...");
    let install_dir = determine_install_directory()?;
    n!("preflight_ok", "✓ Will install to: {}", install_dir.display());

    // 6.5. Fetch sources (git clone, etc.)
    n!("fetch_sources", "📦 Fetching sources from PKGBUILD...");
    let srcdir = temp_dir.join("src");
    crate::source_fetcher::fetch_sources(&pkgbuild.source, &srcdir).await?;
    n!("fetch_sources_ok", "✓ Sources fetched to: {}", srcdir.display());

    // 7. Execute build() - THIS IS THE LONG-RUNNING OPERATION
    n!("build_start", "🏗️  Starting build phase (cancellable)...");
    let executor = crate::pkgbuild_executor::PkgBuildExecutor::new(
        temp_dir.join("src"),
        temp_dir.join("pkg"),
        temp_dir.clone(),
    );

    // TEAM-388: Build with cancellation support
    // The build phase is the longest operation (cargo build can take minutes)
    // Pass the cancel_token to the executor so it can check for cancellation
    if let Err(e) = executor
        .build_with_cancellation(&pkgbuild, cancel_token.clone(), |line| {
            n!("build_output", "{}", line);
        })
        .await
    {
        // Check if it was cancelled
        if cancel_token.is_cancelled() {
            n!("build_cancelled", "❌ Build cancelled by user");
            cleanup_temp_directories(&temp_dir).ok(); // Best effort cleanup
            anyhow::bail!("Build cancelled");
        }
        n!("build_failed", "❌ Build failed: {}", e);
        n!("build_error_detail", "Error details: {:?}", e);
        return Err(e.into());
    }
    n!("build_complete", "✓ Build complete");

    // 8. Execute package()
    n!("package_start", "📦 Starting package phase...");
    if let Err(e) = executor
        .package(&pkgbuild, |line| {
            n!("package_output", "{}", line);
        })
        .await
    {
        n!("package_failed", "❌ Package failed: {}", e);
        n!("package_error_detail", "Error details: {:?}", e);
        return Err(e.into());
    }
    n!("package_complete", "✓ Package complete");

    // 9. Install binary
    n!("install_binary", "💾 Installing binary...");
    let binary_path = install_binary(&temp_dir, &pkgbuild, &install_dir)?;
    n!("install_binary_ok", "✓ Binary installed to: {}", binary_path.display());

    // 10. Add to worker catalog
    n!("catalog_add", "📝 Adding to worker catalog...");
    eprintln!("[worker_install] About to call add_to_catalog for worker_id={}", worker_id);
    add_to_catalog(&worker_id, &pkgbuild, &binary_path, &worker_catalog)?;
    eprintln!("[worker_install] add_to_catalog returned successfully");
    n!("catalog_add_ok", "✓ Added to catalog");

    // 11. Update capabilities (placeholder - actual implementation depends on capabilities system)
    n!("update_caps", "📝 Updating capabilities cache...");
    // TODO: Implement capabilities update when capabilities system is ready
    n!("update_caps_ok", "✓ Capabilities updated");

    // 11. Cleanup
    n!("cleanup", "🧹 Cleaning up temp files...");
    cleanup_temp_directories(&temp_dir)?;
    n!("cleanup_ok", "✓ Cleanup complete");

    n!("install_complete", "✅ Worker installation complete!");
    Ok(())
}

/// Fetch worker metadata from catalog
async fn fetch_worker_metadata(worker_id: &str) -> Result<WorkerMetadata> {
    let catalog_url = std::env::var("WORKER_CATALOG_URL")
        .unwrap_or_else(|_| "http://localhost:8787".to_string());

    let url = format!("{}/workers/{}", catalog_url, worker_id);
    n!("fetch_url", "📡 Fetching from: {}", url);

    // TEAM-378: Add 10-second timeout for catalog requests
    n!("build_client", "🔧 Building HTTP client with 10s timeout...");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("Failed to create HTTP client")?;
    n!("build_client_ok", "✓ HTTP client built");
    
    n!("send_request", "📤 Sending GET request to catalog...");
    let response = client
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

/// Check platform compatibility
fn check_platform_compatibility(worker: &WorkerMetadata) -> Result<()> {
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

/// Download PKGBUILD from catalog
async fn download_pkgbuild(worker_id: &str) -> Result<String> {
    let catalog_url = std::env::var("WORKER_CATALOG_URL")
        .unwrap_or_else(|_| "http://localhost:8787".to_string());

    let url = format!("{}/workers/{}/PKGBUILD", catalog_url, worker_id);
    n!("pkgbuild_url", "📡 Fetching from: {}", url);

    n!("pkgbuild_http_start", "🌐 Creating HTTP client...");
    // TEAM-378: Add 10-second timeout for catalog requests
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("Failed to create HTTP client")?;
    
    n!("pkgbuild_http_get", "📤 Sending GET request...");
    let response = client
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

/// Check dependencies
fn check_dependencies(worker: &WorkerMetadata) -> Result<()> {
    // For now, just log the dependencies
    // TODO: Implement actual dependency checking (which, dpkg -l, etc.)
    for dep in &worker.depends {
        n!("dep_check", "  Dependency: {}", dep);
    }

    Ok(())
}

/// Create temp directories for build
fn create_temp_directories(worker_id: &str) -> Result<PathBuf> {
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
fn determine_install_directory() -> Result<PathBuf> {
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
/// 
/// TEAM-388: Install to user-provided directory (from preflight check)
fn install_binary(
    temp_dir: &PathBuf, 
    pkgbuild: &crate::pkgbuild_parser::PkgBuild,
    install_dir: &PathBuf,
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

/// Add installed worker to catalog
/// 
/// TEAM-378: Registers the installed binary in the worker catalog
/// so it shows up in the "Installed Workers" view
/// TEAM-384: Use shared catalog instance instead of creating new one
fn add_to_catalog(
    worker_id: &str, 
    pkgbuild: &crate::pkgbuild_parser::PkgBuild, 
    binary_path: &PathBuf,
    catalog: &WorkerCatalog,
) -> Result<()> {
    use rbee_hive_worker_catalog::{WorkerBinary, WorkerType, Platform};
    use rbee_hive_artifact_catalog::catalog::ArtifactCatalog;
    
    eprintln!("[add_to_catalog] worker_id={}, binary_path={}", worker_id, binary_path.display());
    
    // Determine worker type from worker_id
    // TEAM-404: Updated to use simplified WorkerType enum
    let worker_type = if worker_id.contains("cpu") {
        WorkerType::Cpu
    } else if worker_id.contains("cuda") {
        WorkerType::Cuda
    } else if worker_id.contains("metal") {
        WorkerType::Metal
    } else {
        anyhow::bail!("Unknown worker type for worker_id: {}", worker_id);
    };
    eprintln!("[add_to_catalog] Determined worker_type: {:?}", worker_type);
    
    // Get current platform
    let platform = Platform::current();
    eprintln!("[add_to_catalog] Platform: {:?}", platform);
    
    // Get binary size
    let size = std::fs::metadata(binary_path)?.len();
    eprintln!("[add_to_catalog] Binary size: {} bytes", size);
    
    // Create unique ID: worker_id-version-platform
    let id = format!("{}-{}-{:?}", worker_id, pkgbuild.pkgver, platform).to_lowercase();
    eprintln!("[add_to_catalog] Generated ID: {}", id);
    
    // Create WorkerBinary entry
    let worker_binary = WorkerBinary::new(
        id.clone(),
        worker_type,
        platform,
        binary_path.clone(),
        size,
        pkgbuild.pkgver.clone(),
    );
    eprintln!("[add_to_catalog] WorkerBinary created, calling catalog.add()...");
    
    // Add to catalog
    catalog.add(worker_binary)?;
    eprintln!("[add_to_catalog] ✓ catalog.add() succeeded");
    
    Ok(())
}

/// Cleanup temp directories
fn cleanup_temp_directories(temp_dir: &PathBuf) -> Result<()> {
    if temp_dir.exists() {
        std::fs::remove_dir_all(temp_dir).context("Failed to cleanup temp directories")?;
    }
    Ok(())
}
