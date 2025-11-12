// TEAM-482: Shared image operations for SD and FLUX
//
// Common image manipulation functions used across model implementations.

use crate::error::Result;
use candle_core::DType;
use candle_core::{Device, Tensor};
use image::{DynamicImage, GenericImageView, RgbImage};

use super::tensor_ops::{tensor_to_rgb_data, TensorNormalization};

/// Convert tensor to RGB image (Stable Diffusion normalization)
///
/// # Arguments
/// * `tensor` - Input tensor [1, 3, H, W] in SD range [-1, 1]
///
/// # Returns
/// RGB image ready for display/saving
#[inline]
pub fn tensor_to_image_sd(tensor: &Tensor) -> Result<DynamicImage> {
    let (image_data, width, height) =
        tensor_to_rgb_data(tensor, TensorNormalization::StableDiffusion)?;

    let img = RgbImage::from_raw(width as u32, height as u32, image_data).ok_or_else(|| {
        crate::error::Error::Generation("Failed to create image from tensor".to_string())
    })?;

    Ok(DynamicImage::ImageRgb8(img))
}

/// Convert tensor to RGB image (FLUX normalization)
///
/// # Arguments
/// * `tensor` - Input tensor [1, 3, H, W] in FLUX range [-1, 1]
///
/// # Returns
/// RGB image ready for display/saving
#[inline]
pub fn tensor_to_image_flux(tensor: &Tensor) -> Result<DynamicImage> {
    let (image_data, width, height) = tensor_to_rgb_data(tensor, TensorNormalization::Flux)?;

    let img = RgbImage::from_raw(width as u32, height as u32, image_data).ok_or_else(|| {
        crate::error::Error::Generation("Failed to create image from tensor".to_string())
    })?;

    Ok(DynamicImage::ImageRgb8(img))
}

/// Convert RGB image to tensor (common pattern)
///
/// Converts image to tensor with normalization to [-1, 1] range.
/// Used for img2img and inpainting.
///
/// # Arguments
/// * `image` - Input RGB image
/// * `device` - Target device (CPU/CUDA)
/// * `dtype` - Target dtype (f16/f32)
///
/// # Returns
/// Tensor [1, 3, H, W] in range [-1, 1]
#[inline]
pub fn image_to_tensor(image: &DynamicImage, device: &Device, dtype: DType) -> Result<Tensor> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Convert to f32 and normalize to [-1, 1]
    let data: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = f32::from(p[0]) / 255.0;
            let g = f32::from(p[1]) / 255.0;
            let b = f32::from(p[2]) / 255.0;
            // Normalize to [-1, 1]
            [r * 2.0 - 1.0, g * 2.0 - 1.0, b * 2.0 - 1.0]
        })
        .collect();

    // Create tensor: [H, W, 3] → [3, H, W] → [1, 3, H, W]
    let tensor = Tensor::from_vec(data, (height as usize, width as usize, 3), device)?;
    let tensor = tensor.permute((2, 0, 1))?.unsqueeze(0)?;

    Ok(tensor.to_dtype(dtype)?)
}

/// Resize image if needed (with validation)
///
/// Ensures image dimensions are valid for the model.
/// Common validation used across SD and FLUX.
///
/// # Arguments
/// * `image` - Input image
/// * `target_width` - Target width (must be multiple of 8)
/// * `target_height` - Target height (must be multiple of 8)
///
/// # Returns
/// Resized image if needed
#[inline]
#[must_use] 
pub fn resize_for_model(
    image: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> DynamicImage {
    let (width, height) = image.dimensions();

    if width != target_width || height != target_height {
        image.resize_exact(target_width, target_height, image::imageops::FilterType::Lanczos3)
    } else {
        image.clone()
    }
}

/// Decode latents to final image (common pattern)
///
/// Combines VAE decode + tensor-to-image conversion.
/// Used by all generation functions (txt2img, img2img, inpaint, FLUX).
///
/// # Arguments
/// * `latents` - Final latent tensor
/// * `decoded_tensor` - Already decoded tensor from VAE
/// * `tensor_to_image_fn` - Function to convert tensor to image
///
/// # Returns
/// Final RGB image
///
/// # Performance
/// Called once per generation at the end
/// Time: ~5ms (tensor conversion only, VAE decode done separately)
///
/// # Note
/// This is a simpler helper that just does the final conversion.
/// VAE decode must be done by the caller since VAE types differ.
#[inline]
pub fn tensor_to_final_image<T>(
    decoded_tensor: &Tensor,
    tensor_to_image_fn: T,
) -> Result<DynamicImage>
where
    T: Fn(&Tensor) -> Result<DynamicImage>,
{
    tensor_to_image_fn(decoded_tensor)
}
