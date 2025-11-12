// TEAM-482: Shared SafeTensors loading utilities
//
// Consolidates unsafe mmap operations with proper safety documentation

use crate::error::{Error, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::Path;

/// Load SafeTensors weights with memory mapping
///
/// # Safety
/// This function uses unsafe memory-mapped file access, which is safe because:
/// 1. Files are from trusted sources (HuggingFace Hub or local validated files)
/// 2. Files are validated by hf-hub before use
/// 3. Candle's mmap implementation handles alignment and bounds checking
/// 4. Files are read-only and immutable after download
///
/// # Arguments
/// * `weights_path` - Path to SafeTensors file
/// * `dtype` - Data type (F16/F32)
/// * `device` - Target device (CPU/CUDA/Metal)
/// * `component_name` - Component name for error messages
///
/// # Returns
/// VarBuilder for loading model weights
///
/// # Performance
/// Memory-mapped loading is ~10x faster than loading into RAM
#[inline]
pub fn load_safetensors(
    weights_path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
    component_name: &str,
) -> Result<VarBuilder<'static>> {
    let weights_path = weights_path.as_ref();

    // SAFETY: See function documentation above
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device).map_err(|e| {
            Error::ModelLoading(format!(
                "Failed to load {} weights from {:?}: {e}",
                component_name, weights_path
            ))
        })?
    };

    Ok(vb)
}

/// Load SafeTensors weights from multiple files
///
/// # Safety
/// Same safety guarantees as `load_safetensors`
///
/// # Arguments
/// * `weights_paths` - Slice of paths to SafeTensors files
/// * `dtype` - Data type (F16/F32)
/// * `device` - Target device (CPU/CUDA/Metal)
/// * `component_name` - Component name for error messages
///
/// # Returns
/// VarBuilder for loading model weights
#[inline]
pub fn load_safetensors_multi(
    weights_paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component_name: &str,
) -> Result<VarBuilder<'static>> {
    let paths: Vec<_> = weights_paths.iter().map(|p| p.as_ref()).collect();

    // SAFETY: See load_safetensors documentation
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&paths, dtype, device).map_err(|e| {
            Error::ModelLoading(format!(
                "Failed to load {} weights from {} files: {e}",
                component_name,
                paths.len()
            ))
        })?
    };

    Ok(vb)
}
