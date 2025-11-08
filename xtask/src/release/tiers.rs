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
fn list_available_tiers() -> Result<Vec<String>> {
    let tier_dir = PathBuf::from(".version-tiers");
    
    if !tier_dir.exists() {
        return Ok(vec![]);
    }

    let mut tiers = Vec::new();
    
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
