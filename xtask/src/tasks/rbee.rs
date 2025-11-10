//! Smart wrapper for rbee-keeper - DEVELOPMENT BUILD ONLY
//!
//! WORKFLOW:
//! 1. `cargo build` -> creates target/debug/rbee-keeper
//! 2. `./rbee` -> auto-rebuilds if deps changed, then launches debug binary
//!
//! For production builds, use a different command (not this).

use anyhow::{Context, Result};
use auto_update::AutoUpdater;
use std::path::PathBuf;
use std::process::Command;

const RBEE_KEEPER_BIN: &str = "bin/00_rbee_keeper";
const TARGET_BINARY_DEBUG: &str = "target/debug/rbee-keeper";
const TARGET_BINARY_RELEASE: &str = "target/release/rbee-keeper";

/// Check if rbee-keeper binary needs rebuilding
/// Uses AutoUpdater to check ALL dependencies (including shared crates)
fn needs_rebuild(_workspace_root: &PathBuf) -> Result<bool> {
    let updater = AutoUpdater::new("rbee-keeper", RBEE_KEEPER_BIN)?;
    updater.needs_rebuild()
}

/// Build rbee-keeper in development mode (fast builds, debug symbols)
fn build_rbee_keeper(workspace_root: &PathBuf) -> Result<()> {
    println!("🔨 Building rbee-keeper (development build)...");

    let status = Command::new("cargo")
        .arg("build")
        .arg("--bin")
        .arg("rbee-keeper")
        .current_dir(workspace_root)
        .status()
        .context("Failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("Failed to build rbee-keeper");
    }

    println!("✅ Build complete\n");
    Ok(())
}

/// Main entry point: auto-rebuild if needed, then launch development binary
pub fn run_rbee_keeper(args: Vec<String>) -> Result<()> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Failed to get workspace root")?
        .to_path_buf();

    // Auto-rebuild if dependencies changed
    if needs_rebuild(&workspace_root)? {
        build_rbee_keeper(&workspace_root)?;
    }

    // Use debug binary (development build)
    let binary_path = {
        let debug_path = workspace_root.join(TARGET_BINARY_DEBUG);
        let release_path = workspace_root.join(TARGET_BINARY_RELEASE);
        
        if debug_path.exists() {
            debug_path
        } else if release_path.exists() {
            println!("⚠️  Using release binary (expected development build)");
            release_path
        } else {
            anyhow::bail!("rbee-keeper binary not found. Run: cargo build --bin rbee-keeper");
        }
    };

    let status = Command::new(&binary_path)
        .args(&args)
        .current_dir(&workspace_root)
        .status()
        .context("Failed to execute rbee-keeper")?;

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}
