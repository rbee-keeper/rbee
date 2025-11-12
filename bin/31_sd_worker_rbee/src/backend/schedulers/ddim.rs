// TEAM-481: DDIM Scheduler - Denoising Diffusion Implicit Models
//
// DDIM is a deterministic scheduler that provides high-quality results.
// It's the default scheduler for most Stable Diffusion models.
//
// Reference: https://arxiv.org/abs/2010.02502

use super::traits::{Scheduler, SchedulerConfig};
use crate::error::Result;
use candle_core::Tensor;

// TEAM-481: DDIM scheduler constants
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;
const INITIAL_ALPHA_PROD: f64 = 1.0;
const FINAL_ALPHA_CUMPROD: f64 = 1.0;
const DEFAULT_TIMESTEP: usize = 0;

/// Configuration for DDIM scheduler
/// 
/// TEAM-481: This struct holds all configuration for DDIM.
/// Implements Default for easy instantiation.
#[derive(Debug, Clone, Copy)]
pub struct DDIMSchedulerConfig {
    /// Number of training timesteps (usually 1000)
    pub train_timesteps: usize,
    /// Beta schedule start value
    pub beta_start: f64,
    /// Beta schedule end value
    pub beta_end: f64,
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            train_timesteps: 1000,
            beta_start: BETA_START,
            beta_end: BETA_END,
        }
    }
}

impl SchedulerConfig for DDIMSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(DDIMScheduler::new(
            self.train_timesteps,
            inference_steps,
            self.beta_start,
            self.beta_end,
        )))
    }
}

/// DDIM Scheduler implementation
/// 
/// TEAM-481: Deterministic scheduler with high quality results
pub struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    final_alpha_cumprod: f64,
}

impl DDIMScheduler {
    /// Create a new DDIM scheduler
    /// 
    /// # Arguments
    /// * `num_train_timesteps` - Number of training timesteps (usually 1000)
    /// * `num_inference_steps` - Number of inference steps (e.g., 20, 50)
    /// * `beta_start` - Beta schedule start value
    /// * `beta_end` - Beta schedule end value
    pub fn new(
        num_train_timesteps: usize,
        num_inference_steps: usize,
        beta_start: f64,
        beta_end: f64,
    ) -> Self {
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .map(|i| i * step_ratio)
            .rev()
            .collect();

        let betas: Vec<f64> = (0..num_train_timesteps)
            .map(|i| {
                let t = (i as f64) / (num_train_timesteps as f64 - 1.0);
                beta_start + t * (beta_end - beta_start)
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut alpha_prod = INITIAL_ALPHA_PROD;
        for beta in &betas {
            alpha_prod *= INITIAL_ALPHA_PROD - beta;
            alphas_cumprod.push(alpha_prod);
        }

        Self {
            timesteps,
            alphas_cumprod,
            final_alpha_cumprod: FINAL_ALPHA_CUMPROD,
        }
    }
}

impl Scheduler for DDIMScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let alpha_prod_t = self
            .alphas_cumprod
            .get(timestep)
            .copied()
            .unwrap_or(self.final_alpha_cumprod);

        let prev_timestep = if timestep > DEFAULT_TIMESTEP {
            timestep.saturating_sub(self.timesteps.len() / self.alphas_cumprod.len())
        } else {
            DEFAULT_TIMESTEP
        };

        let alpha_prod_t_prev = self
            .alphas_cumprod
            .get(prev_timestep)
            .copied()
            .unwrap_or(self.final_alpha_cumprod);

        let beta_prod_t = INITIAL_ALPHA_PROD - alpha_prod_t;
        let beta_prod_t_prev = INITIAL_ALPHA_PROD - alpha_prod_t_prev;

        let pred_original_sample =
            ((sample - (beta_prod_t.sqrt() * model_output)?)? / alpha_prod_t.sqrt())?;
        let pred_sample_direction = (beta_prod_t_prev.sqrt() * model_output)?;
        let prev_sample = ((alpha_prod_t_prev.sqrt() * pred_original_sample)? + pred_sample_direction)?;

        Ok(prev_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddim_scheduler_creation() {
        let config = DDIMSchedulerConfig::default();
        let scheduler = config.build(20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_ddim_timesteps() {
        let scheduler = DDIMScheduler::new(1000, 20, BETA_START, BETA_END);
        assert_eq!(scheduler.timesteps().len(), 20);
    }
}
