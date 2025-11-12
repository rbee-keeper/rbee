// TEAM-482: Safetensors helper functions
//
// Extracted from models/mod.rs for better organization.
// Contains utilities for working with safetensors files.

use anyhow::{bail, Context, Result};
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
