// TEAM-482: GGUF helper functions
//
// Extracted from models/mod.rs for better organization.
// Contains utilities for working with GGUF (quantized) model files.

use anyhow::{Context, Result};
use std::path::Path;

/// Detect architecture from GGUF metadata
///
/// TEAM-090: Read GGUF file and extract architecture from general.architecture field
/// TEAM-482: Extracted to helpers module
pub fn detect_architecture_from_gguf(gguf_path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(gguf_path)
        .with_context(|| format!("Failed to open GGUF file: {}", gguf_path.display()))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .with_context(|| format!("Failed to read GGUF content from {}", gguf_path.display()))?;

    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .context("Missing general.architecture in GGUF metadata")?;

    Ok(arch)
}
