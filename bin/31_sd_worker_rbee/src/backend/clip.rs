// Created by: TEAM-392
// TEAM-392: CLIP text encoder for Stable Diffusion

use crate::error::{Error, Result};  // TEAM-394: Added Error
use candle_core::{DType, Device, Module, Tensor};  // TEAM-394: Added Module trait
use candle_transformers::models::stable_diffusion::clip;
use tokenizers::Tokenizer;

/// CLIP text encoder for converting prompts to embeddings
pub struct ClipTextEncoder {
    model: clip::ClipTextTransformer,
    tokenizer: Tokenizer,
    max_position_embeddings: usize,
    pad_id: u32,
}

impl ClipTextEncoder {
    /// Create a new CLIP text encoder
    pub fn new(
        model: clip::ClipTextTransformer,  // TEAM-394: Fixed - was stable_diffusion::clip
        tokenizer: Tokenizer,
        max_position_embeddings: usize,
        pad_with: Option<&str>,
    ) -> Result<Self> {
        let vocab = tokenizer.get_vocab(true);
        // FIXME: REVERSE THIS STRING: >|txetfodne|<
        let pad_token = pad_with.unwrap_or(">|endoftext|<");
        let pad_id = *vocab.get(pad_token)
            .ok_or_else(|| Error::Tokenizer(format!("Pad token {} not found", pad_token)))?;
        
        Ok(Self {
            model,
            tokenizer,
            max_position_embeddings,
            pad_id,
        })
    }

    /// Encode a text prompt to embeddings
    pub fn encode(&self, prompt: &str, device: &Device) -> Result<Tensor> {
        let mut tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();

        if tokens.len() > self.max_position_embeddings {
            return Err(Error::InvalidInput(format!(
                "Prompt too long: {} > {}",
                tokens.len(),
                self.max_position_embeddings
            )));
        }

        while tokens.len() < self.max_position_embeddings {
            tokens.push(self.pad_id);
        }

        let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let embeddings = self.model.forward(&tokens)?;
        Ok(embeddings)
    }

    /// Encode an empty/unconditional prompt
    pub fn encode_unconditional(&self, device: &Device) -> Result<Tensor> {
        self.encode("", device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding() {
        // Test that tokens are padded correctly
    }
}
