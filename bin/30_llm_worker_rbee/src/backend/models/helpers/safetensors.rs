// TEAM-482: Safetensors helper functions
//
// Extracted from models/mod.rs for better organization.
// Contains utilities for working with safetensors files.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};

/// Scan for safetensors files
///
/// TEAM-017: Candle-idiomatic helper to find safetensors files
/// TEAM-482: Extracted to helpers module
pub fn find_safetensors_files(path: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        Ok((parent.to_path_buf(), vec![path.to_path_buf()]))
    } else if path.is_dir() {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(entry_path);
            }
        }
        if files.is_empty() {
            bail!("No safetensors files found at {}", path.display());
        }
        Ok((path.to_path_buf(), files))
    } else {
        bail!("Path must be a .safetensors file or directory");
    }
}

/// Calculate model size in bytes from safetensors or GGUF files
///
/// TEAM-017: Helper to calculate total model size
/// TEAM-036: Added GGUF support
/// TEAM-482: Extracted to helpers module
pub fn calculate_model_size(model_path: &str) -> Result<u64> {
    let path = Path::new(model_path);

    // TEAM-036: Handle GGUF files
    if model_path.ends_with(".gguf") {
        let metadata = std::fs::metadata(path)?;
        return Ok(metadata.len());
    }

    // Handle safetensors files
    let safetensor_files =
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    files.push(entry_path);
                }
            }
            files
        } else {
            bail!("Path must be a .safetensors, .gguf file or directory");
        };

    if safetensor_files.is_empty() {
        bail!("No safetensors files found at {}", path.display());
    }

    let model_size_bytes: u64 =
        safetensor_files.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();

    Ok(model_size_bytes)
}

/// Load and parse config.json for a model
///
/// TEAM-482: Extracted common pattern from all safetensors loaders
/// Generic function that works with any serde-deserializable config type
pub fn load_config<T: serde::de::DeserializeOwned>(
    model_path: &Path,
    model_name: &str,
) -> Result<T> {
    let config_path = model_path.join("config.json");
    serde_json::from_reader(
        std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
    )
    .with_context(|| format!("Failed to parse {} config.json", model_name))
}

/// Create VarBuilder from safetensors files
///
/// TEAM-482: Extracted common pattern from all safetensors loaders
/// TEAM-019: Always uses F32 dtype (Metal F16 causes forward pass failures)
pub fn create_varbuilder<'a>(
    safetensor_files: &[PathBuf],
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    let dtype = DType::F32; // TEAM-019: F32 for all backends
    tracing::debug!(dtype = ?dtype, device = ?device, "Creating VarBuilder");

    unsafe { VarBuilder::from_mmaped_safetensors(safetensor_files, dtype, device) }
        .context("Failed to create VarBuilder from safetensors")
}
