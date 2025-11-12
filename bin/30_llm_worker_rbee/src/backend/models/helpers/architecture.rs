// TEAM-482: Architecture detection helpers
//
// Extracted from models/mod.rs for better organization.
// Contains utilities for detecting model architecture from config files.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

/// Load config.json from model path
///
/// TEAM-017: Helper to load and parse config.json
/// TEAM-482: Extracted to helpers module
pub fn load_config_json(model_path: &Path) -> Result<Value> {
    let parent = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or_else(|| Path::new("."))
    };

    let config_path = parent.join("config.json");
    let config_json: Value = serde_json::from_reader(
        std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
    )
    .context("Failed to parse config.json")?;

    Ok(config_json)
}

/// Detect model architecture from config.json
///
/// TEAM-017: Checks `model_type` and architectures fields
/// TEAM-482: Extracted to helpers module
pub fn detect_architecture(config_json: &Value) -> Result<String> {
    // Check "model_type" field
    if let Some(model_type) = config_json.get("model_type").and_then(|v| v.as_str()) {
        return Ok(model_type.to_lowercase());
    }

    // Check "architectures" array
    if let Some(archs) = config_json.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            let arch_lower = arch.to_lowercase();
            // Normalize architecture names
            if arch_lower.contains("llama") {
                return Ok("llama".to_string());
            } else if arch_lower.contains("mistral") {
                return Ok("mistral".to_string());
            } else if arch_lower.contains("phi") {
                return Ok("phi".to_string());
            } else if arch_lower.contains("qwen") {
                return Ok("qwen".to_string());
            } else if arch_lower.contains("gemma") {
                return Ok("gemma".to_string());
            }
            return Ok(arch_lower);
        }
    }

    bail!("Could not detect model architecture from config.json");
}
