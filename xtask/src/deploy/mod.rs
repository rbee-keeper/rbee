// Deployment commands
// Created by: TEAM-451
// Individual deployment commands for each app and binary

pub mod binaries;
pub mod commercial;
pub mod docs;
pub mod gates;
pub mod marketplace;
pub mod worker_catalog;

use anyhow::{Context, Result};
use std::path::PathBuf;

fn bump_version(app: &str, bump_type: &str, dry_run: bool) -> Result<()> {
    // Map app name to package directory
    let package_dir = match app {
        "worker" | "gwc" | "worker-catalog" => "bin/80-hono-worker-catalog",
        "commercial" => "frontend/apps/commercial",
        "marketplace" => "frontend/apps/marketplace",
        "docs" | "user-docs" => "frontend/apps/user-docs",
        _ => anyhow::bail!("Version bumping not supported for app: {}", app),
    };
    
    let package_json = PathBuf::from(package_dir).join("package.json");
    
    if !package_json.exists() {
        anyhow::bail!("package.json not found: {}", package_json.display());
    }
    
    // Read current version
    let content = std::fs::read_to_string(&package_json)
        .context("Failed to read package.json")?;
    
    // Parse version
    let version_line = content.lines()
        .find(|line| line.contains("\"version\""))
        .context("No version field in package.json")?;
    
    let current_version = version_line
        .split(':')
        .nth(1)
        .context("Invalid version format")?
        .trim()
        .trim_matches(|c| c == '"' || c == ',' || c == ' ');
    
    // Parse semver
    let parts: Vec<&str> = current_version.split('.').collect();
    if parts.len() != 3 {
        anyhow::bail!("Invalid version format: {}", current_version);
    }
    
    let major: u32 = parts[0].parse()?;
    let minor: u32 = parts[1].parse()?;
    let patch: u32 = parts[2].parse()?;
    
    // Bump version
    let new_version = match bump_type {
        "patch" => format!("{}.{}.{}", major, minor, patch + 1),
        "minor" => format!("{}.{}.0", major, minor + 1),
        "major" => format!("{}.0.0", major + 1),
        _ => anyhow::bail!("Invalid bump type: {}. Use patch, minor, or major", bump_type),
    };
    
    println!("  {} → {}", current_version, new_version);
    
    if !dry_run {
        // Update package.json
        let new_content = content.replace(
            &format!("\"version\": \"{}\"", current_version),
            &format!("\"version\": \"{}\"", new_version)
        );
        
        std::fs::write(&package_json, new_content)
            .context("Failed to write package.json")?;
        
        println!("  ✅ Updated {}", package_json.display());
    } else {
        println!("  🔍 Dry run - would update {}", package_json.display());
    }
    
    Ok(())
}

pub fn run(app: &str, bump: Option<&str>, dry_run: bool) -> Result<()> {
    // Step 1: Bump version if requested
    if let Some(bump_type) = bump {
        println!("📦 Bumping version ({})...", bump_type);
        bump_version(app, bump_type, dry_run)?;
        println!();
    } else {
        println!("⚠️  Deploying current version (no version bump)");
        println!();
    }

    // Step 2: Run deployment gates (unless dry run)
    if !dry_run {
        gates::check_gates(app)?;
    } else {
        println!("🔍 Dry run - skipping deployment gates");
        println!();
    }

    // Step 3: Deploy
    match app {
        // Cloudflare deployments
        "worker" | "gwc" | "worker-catalog" => worker_catalog::deploy(dry_run),
        "commercial" => commercial::deploy(dry_run),
        "marketplace" => marketplace::deploy(dry_run),
        "docs" | "user-docs" => docs::deploy(dry_run),
        
        // Binary deployments (GitHub Releases)
        "keeper" | "rbee-keeper" => binaries::deploy_keeper(dry_run),
        "queen" | "queen-rbee" => binaries::deploy_queen(dry_run),
        "hive" | "rbee-hive" => binaries::deploy_hive(dry_run),
        "llm-worker" | "llm-worker-rbee" => binaries::deploy_llm_worker(dry_run),
        "sd-worker" | "sd-worker-rbee" => binaries::deploy_sd_worker(dry_run),
        
        _ => anyhow::bail!(
            "Unknown app: {}\n\nCloudflare Apps:\n  - worker (gwc.rbee.dev)\n  - commercial (rbee.dev)\n  - marketplace (marketplace.rbee.dev)\n  - docs (docs.rbee.dev)\n\nRust Binaries (GitHub Releases):\n  - keeper (rbee-keeper)\n  - queen (queen-rbee)\n  - hive (rbee-hive)\n  - llm-worker (llm-worker-rbee)\n  - sd-worker (sd-worker-rbee)",
            app
        ),
    }
}
