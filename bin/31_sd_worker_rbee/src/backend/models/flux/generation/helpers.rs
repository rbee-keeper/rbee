// TEAM-488: Helper functions for FLUX generation
// Shared utilities based on candle-transformers/models/flux/sampling.rs

use crate::error::Result;
use candle_core::{Device, IndexOp, Module, Tensor};
// flux module is imported via parent module
use image::{DynamicImage, RgbImage};
use tokenizers::Tokenizer;

/// Generate T5 text embeddings
pub(super) fn t5_embeddings(
    prompt: &str,
    tokenizer: &Tokenizer,
    model: &mut candle_transformers::models::t5::T5EncoderModel,
    device: &Device,
) -> Result<Tensor> {
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| crate::error::Error::ModelLoading(format!("T5 tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();
    
    // FLUX uses 256 tokens for T5
    tokens.resize(256, 0);
    
    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let embeddings = model.forward(&input_token_ids)?;
    
    Ok(embeddings)
}

/// Generate CLIP text embeddings
pub(super) fn clip_embeddings(
    prompt: &str,
    tokenizer: &Tokenizer,
    model: &candle_transformers::models::clip::text_model::ClipTextTransformer,
    device: &Device,
) -> Result<Tensor> {
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| crate::error::Error::ModelLoading(format!("CLIP tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();
    
    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let embeddings = model.forward(&input_token_ids)?;
    
    Ok(embeddings)
}

/// Convert tensor to image
/// Based on reference/candle/candle-examples/examples/flux/main.rs
pub(super) fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    // Clamp to [-1, 1] and convert to [0, 255]
    let tensor = ((tensor.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(candle_core::DType::U8)?;
    let tensor = tensor.to_device(&Device::Cpu)?;
    
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(crate::error::Error::Generation(format!(
            "Expected 4D tensor, got {}D",
            dims.len()
        )));
    }
    let (batch, channel, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    
    if batch != 1 {
        return Err(crate::error::Error::Generation(format!(
            "Expected batch size 1, got {}",
            batch
        )));
    }
    
    if channel != 3 {
        return Err(crate::error::Error::Generation(format!(
            "Expected 3 channels, got {}",
            channel
        )));
    }
    
    let image_data = tensor.i((0,))?.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    
    let img = RgbImage::from_raw(width as u32, height as u32, image_data)
        .ok_or_else(|| crate::error::Error::Generation("Failed to create image from tensor".to_string()))?;
    
    Ok(DynamicImage::ImageRgb8(img))
}
