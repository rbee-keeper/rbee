// TEAM-488: Helper functions for FLUX generation
// Shared utilities based on candle-transformers/models/flux/sampling.rs
// TEAM-482: Uses shared helpers to avoid duplication with SD

use crate::backend::models::shared::tensor_to_image_flux;
use crate::error::Result;
use candle_core::{Device, Module, Tensor};
// flux module is imported via parent module
use image::DynamicImage;
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
        .map_err(|e| crate::error::Error::ModelLoading(format!("T5 tokenization failed: {e}")))?
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
        .map_err(|e| crate::error::Error::ModelLoading(format!("CLIP tokenization failed: {e}")))?
        .get_ids()
        .to_vec();

    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let embeddings = model.forward(&input_token_ids)?;

    Ok(embeddings)
}

/// Convert tensor to image
///
/// TEAM-482: Delegates to shared helper to avoid duplication
pub(super) fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    tensor_to_image_flux(tensor)
}
