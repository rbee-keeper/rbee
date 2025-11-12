// Created by: TEAM-393
// TEAM-393: Image utilities for SD worker

use crate::error::Result;
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, GenericImageView, ImageFormat}; // TEAM-394: Added GenericImageView trait
use std::io::Cursor;

/// Convert image to base64-encoded PNG
pub fn image_to_base64(image: &DynamicImage) -> Result<String> {
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    image.write_to(&mut cursor, ImageFormat::Png)?;
    Ok(STANDARD.encode(&buffer))
}

/// Decode base64 string to image
pub fn base64_to_image(base64: &str) -> Result<DynamicImage> {
    let bytes = STANDARD
        .decode(base64)
        .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid base64: {}", e)))?;
    let image = image::load_from_memory(&bytes)?;
    Ok(image)
}

/// Resize image to target dimensions
pub fn resize_image(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

/// Ensure image dimensions are multiples of 8 (required for SD)
pub fn ensure_multiple_of_8(image: &DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    let new_width = (width / 8) * 8;
    let new_height = (height / 8) * 8;

    if new_width != width || new_height != height {
        resize_image(image, new_width, new_height)
    } else {
        image.clone()
    }
}

/// Process mask image for inpainting
///
/// TEAM-487: Converts mask to proper format for inpainting
/// - White (255) = inpaint this region
/// - Black (0) = keep this region  
/// - Resize to target dimensions
/// - Binary threshold
///
/// # Arguments
/// * `mask` - Input mask (any format, will be converted to grayscale)
/// * `target_width` - Target width (must match image width)
/// * `target_height` - Target height (must match image height)
///
/// # Returns
/// Processed mask (grayscale, resized, binary)
pub fn process_mask(
    mask: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> Result<DynamicImage> {
    // 1. Convert to grayscale
    let gray = mask.to_luma8();

    // 2. Resize to target dimensions
    let resized = image::DynamicImage::ImageLuma8(gray).resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    );

    // 3. Threshold to binary (0 or 255)
    let mut binary = resized.to_luma8();
    for pixel in binary.pixels_mut() {
        pixel[0] = if pixel[0] > 127 { 255 } else { 0 };
    }

    Ok(image::DynamicImage::ImageLuma8(binary))
}

/// Convert mask to latent space tensor
///
/// TEAM-487: Inpainting models need mask in latent space (1/8 resolution)
///
/// # Arguments
/// * `mask` - Processed mask (binary, full resolution)
/// * `device` - Device to create tensor on
/// * `dtype` - Data type
///
/// # Returns
/// Mask tensor in latent space (shape: [1, 1, height/8, width/8])
pub fn mask_to_latent_tensor(
    mask: &DynamicImage,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<candle_core::Tensor> {
    use candle_core::Tensor;

    // 1. Resize mask to latent dimensions (1/8 of original)
    let (width, height) = mask.dimensions();
    let latent_mask = mask.resize_exact(
        width / 8,
        height / 8,
        image::imageops::FilterType::Nearest, // Use nearest for binary mask
    );

    // 2. Convert to grayscale and get pixel data
    let gray = latent_mask.to_luma8();
    let data = gray.into_raw();

    // 3. Convert to f32 and normalize [0.0, 1.0]
    let data: Vec<f32> = data.iter().map(|&x| x as f32 / 255.0).collect();

    // 4. Reshape to (1, 1, height, width)
    let h = (height / 8) as usize;
    let w = (width / 8) as usize;
    let tensor = Tensor::from_vec(data, (h, w), device)?;
    let tensor = tensor.unsqueeze(0)?.unsqueeze(0)?; // Add batch and channel dims

    Ok(tensor.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_base64_roundtrip() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let base64 = image_to_base64(&img).unwrap();
        let decoded = base64_to_image(&base64).unwrap();
        assert_eq!(img.dimensions(), decoded.dimensions());
    }

    #[test]
    fn test_resize() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let resized = resize_image(&img, 64, 64);
        assert_eq!(resized.dimensions(), (64, 64));
    }

    #[test]
    fn test_ensure_multiple_of_8() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(67, 67));
        let fixed = ensure_multiple_of_8(&img);
        let (w, h) = fixed.dimensions();
        assert_eq!(w % 8, 0);
        assert_eq!(h % 8, 0);
        assert_eq!((w, h), (64, 64));
    }

    #[test]
    fn test_process_mask() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(67, 67));
        let mask = process_mask(&img, 64, 64).unwrap();
        let (w, h) = mask.dimensions();
        assert_eq!((w, h), (64, 64));
        assert_eq!(w % 8, 0);
        assert_eq!(h % 8, 0);
    }
}
