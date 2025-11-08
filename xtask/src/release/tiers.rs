// Tier configuration loading
// Created by: TEAM-451

use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct TierConfig {
    pub name: String,
    pub description: String,
    pub rust: RustConfig,
    pub javascript: JavaScriptConfig,
}

#[derive(Debug, Deserialize)]
pub struct RustConfig {
    #[serde(default)]
    pub crates: Vec<String>,
    #[serde(default)]
    pub shared_crates: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct JavaScriptConfig {
    #[serde(default)]
    pub packages: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum BumpType {
    Patch,
    Minor,
    Major,
}

pub fn load_tier_config(tier: &str) -> Result<TierConfig> {
    let path = PathBuf::from(".version-tiers").join(format!("{}.toml", tier));

    if !path.exists() {
        let available = list_available_tiers()?;
        anyhow::bail!(
            "Tier config not found: {}\n\nAvailable tiers: {}",
            path.display(),
            available.join(", ")
        );
    }

    let content = std::fs::read_to_string(&path)?;
    let config: TierConfig = toml::from_str(&content)?;

    Ok(config)
}

/// List all available tier configurations
/// TEAM-451: RULE ZERO - Tiers are discovered dynamically, not hardcoded
fn list_available_tiers() -> Result<Vec<String>> {
    let tier_dir = PathBuf::from(".version-tiers");
    
    if !tier_dir.exists() {
        return Ok(vec![]);
    }

    let mut tiers = Vec::new();
    
    // Discover all .toml files in .version-tiers/
    for entry in std::fs::read_dir(&tier_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                tiers.push(stem.to_string());
            }
        }
    }
    
    tiers.sort();
    Ok(tiers)
}

/// Discover all worker tiers dynamically from bin/ directory
/// TEAM-451: Workers are infinite - discover them, don't hardcode them
pub fn discover_worker_tiers() -> Result<Vec<String>> {
    let bin_dir = PathBuf::from("bin");
    let mut workers = Vec::new();
    
    if !bin_dir.exists() {
        return Ok(workers);
    }
    
    // Scan bin/ for worker crates
    for entry in std::fs::read_dir(&bin_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if !path.is_dir() {
            continue;
        }
        
        // Check if it's a worker (has "worker" in name and has Cargo.toml)
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.contains("worker") && path.join("Cargo.toml").exists() {
                // Extract worker type from directory name
                // e.g., "30_llm_worker_rbee" -> "llm-worker"
                // e.g., "31_sd_worker_rbee" -> "sd-worker"
                let worker_type = extract_worker_type(name);
                if !workers.contains(&worker_type) {
                    workers.push(worker_type);
                }
            }
        }
    }
    
    workers.sort();
    Ok(workers)
}

/// Extract worker type from directory name
/// TEAM-451: Parse worker type from directory name dynamically
fn extract_worker_type(dir_name: &str) -> String {
    // Remove number prefix (e.g., "30_")
    let without_prefix = dir_name.split('_')
        .skip_while(|s| s.chars().all(|c| c.is_numeric()))
        .collect::<Vec<_>>()
        .join("_");
    
    // Remove "_rbee" suffix and convert to kebab-case
    let without_suffix = without_prefix.replace("_rbee", "");
    without_suffix.replace('_', "-")
}

pub fn get_all_rust_crates(config: &TierConfig) -> Vec<PathBuf> {
    let mut crates = Vec::new();

    // Add main crates
    for crate_path in &config.rust.crates {
        crates.push(PathBuf::from(crate_path));
    }

    // Add shared crates
    for crate_path in &config.rust.shared_crates {
        crates.push(PathBuf::from(crate_path));
    }

    crates
}
