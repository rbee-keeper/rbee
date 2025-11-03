// Created by: TEAM-393
// TEAM-393: Image utilities for SD worker

use crate::error::Result;
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, ImageFormat};
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
        .map_err(|e| crate::error::Error::InvalidInput(format\!("Invalid base64: {}", e)))?;
    let image = image::load_from_memory(&bytes)?;
    Ok(image)
}

/// Resize image to target dimensions
pub fn resize_image(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

/// Ensure image dimensions are multiples of 8 (required for SD)
pub fn ensure_multiple_of_8(image: &DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    let new_width = (width / 8) * 8;
    let new_height = (height / 8) * 8;
    
    if new_width \!= width || new_height \!= height {
        resize_image(image, new_width, new_height)
    } else {
        image.clone()
    }
}

/// Process mask image for inpainting
pub fn process_mask(mask: &DynamicImage) -> Result<DynamicImage> {
    // Convert to grayscale
    let gray = mask.to_luma8();
    
    // Ensure dimensions are multiples of 8
    let (width, height) = gray.dimensions();
    let new_width = (width / 8) * 8;
    let new_height = (height / 8) * 8;
    
    let resized = image::imageops::resize(
        &gray,
        new_width,
        new_height,
        image::imageops::FilterType::Lanczos3,
    );
    
    Ok(DynamicImage::ImageLuma8(resized))
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
        assert_eq\!(img.dimensions(), decoded.dimensions());
    }

    #[test]
    fn test_resize() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let resized = resize_image(&img, 64, 64);
        assert_eq\!(resized.dimensions(), (64, 64));
    }

    #[test]
    fn test_ensure_multiple_of_8() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(67, 67));
        let fixed = ensure_multiple_of_8(&img);
        let (w, h) = fixed.dimensions();
        assert_eq\!(w % 8, 0);
        assert_eq\!(h % 8, 0);
        assert_eq\!((w, h), (64, 64));
    }

    #[test]
    fn test_process_mask() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(67, 67));
        let mask = process_mask(&img).unwrap();
        let (w, h) = mask.dimensions();
        assert_eq\!(w % 8, 0);
        assert_eq\!(h % 8, 0);
    }
}
