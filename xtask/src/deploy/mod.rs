// Deployment commands
// Created by: TEAM-451
// Individual deployment commands for each app and binary
// TEAM-463: Added nextjs_ssg abstraction for Next.js deployments

pub mod admin;
pub mod binaries;
pub mod commercial;
pub mod docs;
pub mod gates;
pub mod marketplace;
pub mod nextjs_ssg; // TEAM-463: Shared Next.js SSG deployment logic
pub mod worker_catalog;

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

fn bump_version(app: &str, bump_type: &str, dry_run: bool) -> Result<()> {
    // Map app name to package directory
    let package_dir = match app {
        "admin" => "bin/78-admin",
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

// TEAM-452: RULE ZERO FIX - Removed duplicate interactive menu
// The release command already has an interactive menu that handles this
// TEAM-463: Added production flag for production deployments
pub fn run(app: &str, bump: Option<&str>, production: bool, dry_run: bool) -> Result<()> {
    // TEAM-452: Bump is optional - if called from release, version already bumped
    if let Some(bump_type) = bump {
        println!("📦 Bumping version ({})...", bump_type);
        bump_version(app, bump_type, dry_run)?;
        println!();
        
        // TEAM-452: Only run gates if we're doing a standalone deploy with version bump
        // If called from release command, gates already ran before version bump
        if !dry_run {
            println!("{}", "🚦 Running deployment gates...".bright_cyan());
            println!();
            gates::check_gates(app)?;
            println!();
            println!("{}", "✅ All gates passed!".bright_green());
            println!();
        } else {
            println!("🔍 Dry run - skipping deployment gates");
            println!();
        }
    }
    // If bump is None, gates were already checked by release command

    // Step 3: Deploy
    match app {
        // Cloudflare deployments
        "admin" => admin::deploy(production, dry_run),
        "worker" | "gwc" | "worker-catalog" => worker_catalog::deploy(production, dry_run),
        "commercial" => commercial::deploy(production, dry_run),
        "marketplace" => marketplace::deploy(production, dry_run),
        "docs" | "user-docs" => docs::deploy(production, dry_run),
        
        // Binary deployments (GitHub Releases)
        "keeper" | "rbee-keeper" => binaries::deploy_keeper(dry_run),
        "queen" | "queen-rbee" => binaries::deploy_queen(dry_run),
        "hive" | "rbee-hive" => binaries::deploy_hive(dry_run),
        "llm-worker" | "llm-worker-rbee" => binaries::deploy_llm_worker(dry_run),
        "sd-worker" | "sd-worker-rbee" => binaries::deploy_sd_worker(dry_run),
        
        _ => anyhow::bail!(
            "Unknown app: {}\n\nCloudflare Apps:\n  - admin (install.rbee.dev)\n  - worker (gwc.rbee.dev)\n  - commercial (rbee.dev)\n  - marketplace (marketplace.rbee.dev)\n  - docs (docs.rbee.dev)\n\nRust Binaries (GitHub Releases):\n  - keeper (rbee-keeper)\n  - queen (queen-rbee)\n  - hive (rbee-hive)\n  - llm-worker (llm-worker-rbee)\n  - sd-worker (sd-worker-rbee)",
            app
        ),
    }
}
