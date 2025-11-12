// TEAM-482: GGUF helper functions
//
// Extracted from models/mod.rs for better organization.
// Contains utilities for working with GGUF (quantized) model files.

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use std::path::Path;

/// Load GGUF file and parse content
///
/// TEAM-482: Extracted common pattern from all quantized loaders
/// Returns both the file handle (needed for weight loading) and parsed content
pub fn load_gguf_content(path: &Path) -> Result<(std::fs::File, gguf_file::Content)> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open GGUF file at {}", path.display()))?;

    let content = gguf_file::Content::read(&mut file)
        .with_context(|| format!("Failed to read GGUF content from {}", path.display()))?;

    Ok((file, content))
}

/// Extract vocab_size from GGUF metadata
///
/// TEAM-482: Extracted common pattern from all quantized loaders
/// TEAM-089: Supports fallback to tokenizer array if vocab_size not in metadata
///
/// Tries in order:
/// 1. {architecture}.vocab_size (e.g., "llama.vocab_size", "phi.vocab_size")
/// 2. llama.vocab_size (common fallback)
/// 3. tokenizer.ggml.tokens array length (derive from tokenizer)
pub fn extract_vocab_size(content: &gguf_file::Content, architecture: &str) -> Result<usize> {
    // Try architecture-specific key first
    let arch_key = format!("{}.vocab_size", architecture);

    let vocab_size = content
        .metadata
        .get(&arch_key)
        .or_else(|| content.metadata.get("llama.vocab_size")) // Common fallback
        .and_then(|v| v.to_u32().ok())
        .or_else(|| {
            // Derive from tokenizer array as last resort
            content.metadata.get("tokenizer.ggml.tokens").and_then(|v| match v {
                gguf_file::Value::Array(arr) => {
                    tracing::debug!(
                        vocab_size = arr.len(),
                        source = "tokenizer.ggml.tokens",
                        "Derived vocab_size from tokenizer array"
                    );
                    Some(arr.len() as u32)
                }
                _ => None,
            })
        })
        .with_context(|| {
            format!(
                "Cannot determine vocab_size from GGUF metadata (tried {}.vocab_size, \
                 llama.vocab_size, tokenizer.ggml.tokens)",
                architecture
            )
        })?;

    Ok(vocab_size as usize)
}

/// Extract EOS token ID from GGUF metadata
///
/// TEAM-482: Extracted common pattern from all quantized loaders
/// Returns the EOS token ID from metadata, or the provided default if not found
pub fn extract_eos_token_id(content: &gguf_file::Content, default: u32) -> u32 {
    content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(default)
}

/// Detect architecture from GGUF metadata
///
/// TEAM-090: Read GGUF file and extract architecture from general.architecture field
/// TEAM-482: Extracted to helpers module
pub fn detect_architecture_from_gguf(gguf_path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(gguf_path)
        .with_context(|| format!("Failed to open GGUF file: {}", gguf_path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .with_context(|| format!("Failed to read GGUF content from {}", gguf_path.display()))?;

    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| match v {
            gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .context("Missing general.architecture in GGUF metadata")?;

    Ok(arch)
}
