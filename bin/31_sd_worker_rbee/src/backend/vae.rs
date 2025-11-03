// Created by: TEAM-392
// TEAM-392: VAE decoder for Stable Diffusion

use crate::error::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use image::{DynamicImage, RgbImage};

pub struct VaeDecoder {
    model: AutoEncoderKL,
    scale_factor: f64,
}

impl VaeDecoder {
    pub fn new(model: AutoEncoderKL, scale_factor: f64) -> Self {
        Self {
            model,
            scale_factor,
        }
    }

    pub fn decode(&self, latents: &Tensor) -> Result<DynamicImage> {
        let scaled_latents = (latents / self.scale_factor)?;
        let decoded = self.model.decode(&scaled_latents)?;
        
        let image = tensor_to_image(&decoded)?;
        Ok(image)
    }
}

pub fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    let tensor = ((tensor / 2.)? + 0.5)?;
    let tensor = tensor.to_device(&Device::Cpu)?;
    let tensor = (tensor.clamp(0f32, 1.)? * 255.)?;
    let tensor = tensor.to_dtype(DType::U8)?;
    
    let (channel, height, width) = tensor.dims3()?;
    if channel != 3 {
        return Err(crate::error::Error::Generation(
            format!("Expected 3 channels, got {}", channel)
        ));
    }

    let data = tensor.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let img = RgbImage::from_raw(width as u32, height as u32, data)
        .ok_or_else(|| crate::error::Error::Generation("Failed to create image".to_string()))?;
    
    Ok(DynamicImage::ImageRgb8(img))
}

pub fn image_to_tensor(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    let img = image.to_rgb8();
    let (width, height) = img.dimensions();
    let data = img.into_raw();
    
    let tensor = Tensor::from_vec(data, (height as usize, width as usize, 3), device)?;
    let tensor = tensor.permute((2, 0, 1))?.to_dtype(DType::F32)?;
    let tensor = (tensor.affine(2. / 255., -1.))?.unsqueeze(0)?;
    
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_image_roundtrip() {
        // Test tensor <-> image conversion
    }
}
