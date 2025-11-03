// Created by: TEAM-392
// TEAM-392: Main inference pipeline for Stable Diffusion

use crate::error::{Error, Result};
use crate::backend::{clip::ClipTextEncoder, vae::VaeDecoder, scheduler::Scheduler, sampling::SamplingConfig};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion;
use image::DynamicImage;
use rand::{Rng, SeedableRng};

pub struct InferencePipeline {
    clip: ClipTextEncoder,
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: VaeDecoder,
    scheduler: Box<dyn Scheduler>,
    device: Device,
    dtype: DType,
}

impl InferencePipeline {
    pub fn new(
        clip: ClipTextEncoder,
        unet: stable_diffusion::unet_2d::UNet2DConditionModel,
        vae: VaeDecoder,
        scheduler: Box<dyn Scheduler>,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            clip,
            unet,
            vae,
            scheduler,
            device,
            dtype,
        }
    }

    pub fn text_to_image<F>(
        &self,
        config: &SamplingConfig,
        mut progress_callback: F,
    ) -> Result<DynamicImage>
    where
        F: FnMut(usize, usize),
    {
        config.validate()?;

        if let Some(seed) = config.seed {
            self.device.set_seed(seed)?;
        }

        let use_guidance = config.guidance_scale > 1.0;

        let text_embeddings = self.clip.encode(&config.prompt, &self.device)?;
        let text_embeddings = if use_guidance {
            let uncond_embeddings = self.clip.encode_unconditional(&self.device)?;
            Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?
        } else {
            text_embeddings
        };

        let text_embeddings = text_embeddings.to_dtype(self.dtype)?;

        let latent_height = config.height / 8;
        let latent_width = config.width / 8;
        
        let mut latents = self.init_latents(1, 4, latent_height, latent_width)?;

        let timesteps = self.scheduler.timesteps();
        let num_steps = timesteps.len();

        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            progress_callback(step_idx, num_steps);

            let latent_model_input = if use_guidance {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let noise_pred = self.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            let noise_pred = if use_guidance {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
                
                let guidance = config.guidance_scale;
                (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
            } else {
                noise_pred
            };

            latents = self.scheduler.step(&noise_pred, timestep, &latents)?;
        }

        progress_callback(num_steps, num_steps);

        let image = self.vae.decode(&latents)?;
        Ok(image)
    }

    fn init_latents(
        &self,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let shape = (batch_size, channels, height, width);
        let latents = Tensor::randn(0f32, 1.0, shape, &self.device)?;
        Ok(latents.to_dtype(self.dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_validation() {
        let mut config = SamplingConfig::default();
        config.prompt = "test prompt".to_string();
        assert!(config.validate().is_ok());
    }
}
