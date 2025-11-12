// TEAM-482: Shared tensor operations for SD and FLUX
//
// Common tensor manipulation functions used across model implementations.

use crate::error::{Error, Result};
use candle_core::DType;
use candle_core::{Device, IndexOp, Tensor};

/// Convert tensor to image data (common pattern)
///
/// Handles the tensor → RGB conversion with proper normalization.
/// Used by both SD and FLUX implementations.
///
/// # Arguments
/// * `tensor` - Input tensor [batch, channels, height, width]
/// * `normalization` - Normalization mode (SD vs FLUX use different ranges)
///
/// # Returns
/// Flattened u8 vector ready for RGB image creation
#[inline]
pub fn tensor_to_rgb_data(
    tensor: &Tensor,
    normalization: TensorNormalization,
) -> Result<(Vec<u8>, usize, usize)> {
    // Normalize based on model type
    let tensor = match normalization {
        TensorNormalization::StableDiffusion => {
            // SD: [-1, 1] → [0, 1] → [0, 255]
            let tensor = ((tensor / 2.)? + 0.5)?;
            (tensor.clamp(0f32, 1.)? * 255.)?
        }
        TensorNormalization::Flux => {
            // FLUX: [-1, 1] → [0, 1] → [0, 255]
            ((tensor.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        }
    };

    let tensor = tensor.to_device(&Device::Cpu)?;
    let tensor = tensor.to_dtype(DType::U8)?;

    // Validate dimensions
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(Error::Generation(format!("Expected 4D tensor, got {}D", dims.len())));
    }

    let (batch, channel, height, width) = (dims[0], dims[1], dims[2], dims[3]);

    if batch != 1 {
        return Err(Error::Generation(format!("Expected batch size 1, got {batch}")));
    }

    if channel != 3 {
        return Err(Error::Generation(format!("Expected 3 channels, got {channel}")));
    }

    // Convert to RGB data: [batch, C, H, W] → [H, W, C] → flat Vec<u8>
    let image_data = tensor.i((0,))?.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;

    Ok((image_data, width, height))
}

/// Normalization mode for different model types
#[derive(Debug, Clone, Copy)]
pub enum TensorNormalization {
    /// Stable Diffusion: [-1, 1] range
    StableDiffusion,
    /// FLUX: [-1, 1] range with different scaling
    Flux,
}

/// Validate tensor batch size
///
/// Common validation used across models
#[inline(always)]
pub fn validate_batch_size(tensor: &Tensor, expected: usize) -> Result<()> {
    let dims = tensor.dims();
    if dims.is_empty() {
        return Err(Error::Generation("Empty tensor".to_string()));
    }

    let batch = dims[0];
    if batch != expected {
        return Err(Error::Generation(format!("Expected batch size {expected}, got {batch}")));
    }

    Ok(())
}

/// Validate tensor channel count
///
/// Common validation for RGB images
#[inline(always)]
pub fn validate_channels(tensor: &Tensor, expected: usize) -> Result<()> {
    let dims = tensor.dims();
    if dims.len() < 2 {
        return Err(Error::Generation("Tensor has insufficient dimensions".to_string()));
    }

    let channels = dims[1];
    if channels != expected {
        return Err(Error::Generation(format!("Expected {expected} channels, got {channels}")));
    }

    Ok(())
}
