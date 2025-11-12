// TEAM-481: DDPM Scheduler - Denoising Diffusion Probabilistic Models
//
// DDPM is a probabilistic scheduler that adds noise during sampling.
// It's particularly good for inpainting tasks.
//
// Reference: https://arxiv.org/abs/2006.11239

use super::traits::{Scheduler, SchedulerConfig};
use super::types::BetaSchedule;
use crate::error::Result;
use candle_core::Tensor;

// TEAM-481: DDPM scheduler constants
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;
const INIT_NOISE_SIGMA: f64 = 1.0;
const INITIAL_ALPHA_PROD: f64 = 1.0;
const MIN_VARIANCE: f64 = 1e-20;

/// Configuration for DDPM scheduler
/// 
/// TEAM-481: Simplified version focusing on core functionality
#[derive(Debug, Clone, Copy)]
pub struct DDPMSchedulerConfig {
    /// Number of training timesteps (usually 1000)
    pub train_timesteps: usize,
    /// Beta schedule start value
    pub beta_start: f64,
    /// Beta schedule end value
    pub beta_end: f64,
    /// Beta schedule type
    pub beta_schedule: BetaSchedule,
}

impl Default for DDPMSchedulerConfig {
    fn default() -> Self {
        Self {
            train_timesteps: 1000,
            beta_start: BETA_START,
            beta_end: BETA_END,
            beta_schedule: BetaSchedule::ScaledLinear,
        }
    }
}

impl SchedulerConfig for DDPMSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(DDPMScheduler::new(
            self.train_timesteps,
            inference_steps,
            self.beta_start,
            self.beta_end,
            self.beta_schedule,
        )?))
    }
}

/// DDPM Scheduler implementation
/// 
/// TEAM-481: Probabilistic scheduler with noise injection
pub struct DDPMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
}

impl DDPMScheduler {
    /// Create a new DDPM scheduler
    /// 
    /// # Arguments
    /// * `num_train_timesteps` - Number of training timesteps (usually 1000)
    /// * `num_inference_steps` - Number of inference steps (e.g., 20, 50)
    /// * `beta_start` - Beta schedule start value
    /// * `beta_end` - Beta schedule end value
    /// * `beta_schedule` - Beta schedule type
    pub fn new(
        num_train_timesteps: usize,
        num_inference_steps: usize,
        beta_start: f64,
        beta_end: f64,
        beta_schedule: BetaSchedule,
    ) -> Result<Self> {
        // Create beta schedule
        let betas: Vec<f64> = match beta_schedule {
            BetaSchedule::Linear => {
                (0..num_train_timesteps)
                    .map(|i| {
                        let t = (i as f64) / (num_train_timesteps as f64 - 1.0);
                        beta_start + t * (beta_end - beta_start)
                    })
                    .collect()
            }
            BetaSchedule::ScaledLinear => {
                (0..num_train_timesteps)
                    .map(|i| {
                        let t = (i as f64) / (num_train_timesteps as f64 - 1.0);
                        let beta_sqrt = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                        beta_sqrt * beta_sqrt
                    })
                    .collect()
            }
            BetaSchedule::SquaredcosCapV2 => {
                // Simplified cosine schedule
                (0..num_train_timesteps)
                    .map(|i| {
                        let t = (i as f64) / num_train_timesteps as f64;
                        let alpha_bar = f64::cos((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2);
                        (1.0 - alpha_bar).min(0.999)
                    })
                    .collect()
            }
        };

        // Calculate cumulative product of alphas
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut alpha_prod = INITIAL_ALPHA_PROD;
        for beta in &betas {
            alpha_prod *= INITIAL_ALPHA_PROD - beta;
            alphas_cumprod.push(alpha_prod);
        }

        // Create timesteps
        let inference_steps = num_inference_steps.min(num_train_timesteps);
        let step_ratio = num_train_timesteps / inference_steps;
        let timesteps: Vec<usize> = (0..inference_steps)
            .map(|i| i * step_ratio)
            .rev()
            .collect();

        Ok(Self {
            timesteps,
            alphas_cumprod,
            step_ratio,
        })
    }

    /// Get variance for a given timestep
    fn get_variance(&self, timestep: usize) -> f64 {
        let prev_t = timestep as isize - self.step_ratio as isize;
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            INITIAL_ALPHA_PROD
        };
        
        let current_beta_t = INITIAL_ALPHA_PROD - alpha_prod_t / alpha_prod_t_prev;
        let variance = (INITIAL_ALPHA_PROD - alpha_prod_t_prev) / (INITIAL_ALPHA_PROD - alpha_prod_t) * current_beta_t;
        
        variance.max(MIN_VARIANCE)
    }
}

impl Scheduler for DDPMScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        let alpha_prod = self.alphas_cumprod.get(timestep).copied().unwrap_or(INITIAL_ALPHA_PROD);
        let sqrt_alpha_prod = alpha_prod.sqrt();
        let sqrt_one_minus_alpha_prod = (INITIAL_ALPHA_PROD - alpha_prod).sqrt();
        let scaled_original = (original * sqrt_alpha_prod)?;
        let scaled_noise = (noise * sqrt_one_minus_alpha_prod)?;
        Ok((&scaled_original + scaled_noise)?)
    }

    fn init_noise_sigma(&self) -> f64 {
        INIT_NOISE_SIGMA
    }

    fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Result<Tensor> {
        // DDPM doesn't scale the model input
        Ok(sample)
    }

    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let prev_t = timestep as isize - self.step_ratio as isize;

        // 1. Compute alphas, betas
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            INITIAL_ALPHA_PROD
        };
        let beta_prod_t = INITIAL_ALPHA_PROD - alpha_prod_t;
        let beta_prod_t_prev = INITIAL_ALPHA_PROD - alpha_prod_t_prev;
        let current_alpha_t = alpha_prod_t / alpha_prod_t_prev;
        let current_beta_t = INITIAL_ALPHA_PROD - current_alpha_t;

        // 2. Compute predicted original sample (assuming epsilon prediction)
        let pred_original_sample = ((sample - (beta_prod_t.sqrt() * model_output)?)? / alpha_prod_t.sqrt())?;

        // 3. Compute coefficients
        let pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t;
        let current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t;

        // 4. Compute predicted previous sample
        let pred_prev_sample = ((&pred_original_sample * pred_original_sample_coeff)?
            + (sample * current_sample_coeff)?)?;

        // 5. Add noise (for t > 0)
        if timestep > 0 {
            let variance = self.get_variance(timestep);
            let noise = model_output.randn_like(0.0, 1.0)?;
            let variance_sample = (noise * variance.sqrt())?;
            Ok((&pred_prev_sample + variance_sample)?)
        } else {
            Ok(pred_prev_sample)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddpm_scheduler_creation() {
        let config = DDPMSchedulerConfig::default();
        let scheduler = config.build(20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_ddpm_timesteps() {
        let scheduler = DDPMScheduler::new(1000, 20, BETA_START, BETA_END, BetaSchedule::ScaledLinear).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_ddpm_variance() {
        let scheduler = DDPMScheduler::new(1000, 20, BETA_START, BETA_END, BetaSchedule::Linear).unwrap();
        let variance = scheduler.get_variance(10);
        assert!(variance > 0.0);
        assert!(variance >= MIN_VARIANCE);
    }
}
