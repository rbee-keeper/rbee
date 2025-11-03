// Created by: TEAM-392
// TEAM-392: Diffusion schedulers for Stable Diffusion

use crate::error::Result;
use candle_core::Tensor;

pub trait Scheduler: Send + Sync {{
    fn timesteps(&self) -> &[usize];
    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor>;
}}

pub struct DDIMScheduler {{
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    final_alpha_cumprod: f64,
}}

impl DDIMScheduler {{
    pub fn new(num_train_timesteps: usize, num_inference_steps: usize) -> Self {{
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .map(|i| i * step_ratio)
            .rev()
            .collect();

        let betas: Vec<f64> = (0..num_train_timesteps)
            .map(|i| {{
                let beta_start = 0.00085_f64;
                let beta_end = 0.012_f64;
                let t = (i as f64) / (num_train_timesteps as f64 - 1.0);
                beta_start + t * (beta_end - beta_start)
            }})
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut alpha_prod = 1.0;
        for beta in &betas {{
            alpha_prod *= 1.0 - beta;
            alphas_cumprod.push(alpha_prod);
        }}

        Self {{
            timesteps,
            alphas_cumprod,
            final_alpha_cumprod: 1.0,
        }}
    }}
}}

impl Scheduler for DDIMScheduler {{
    fn timesteps(&self) -> &[usize] {{
        &self.timesteps
    }}

    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {{
        let alpha_prod_t = self.alphas_cumprod.get(timestep).copied().unwrap_or(self.final_alpha_cumprod);
        
        let prev_timestep = if timestep > 0 {{
            timestep.saturating_sub(self.timesteps.len() / self.alphas_cumprod.len())
        }} else {{
            0
        }};
        
        let alpha_prod_t_prev = self.alphas_cumprod.get(prev_timestep).copied().unwrap_or(self.final_alpha_cumprod);

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        let pred_original_sample = ((sample - (beta_prod_t.sqrt() * model_output)?)? / alpha_prod_t.sqrt())?;
        let pred_sample_direction = (beta_prod_t_prev.sqrt() * model_output)?;
        let prev_sample = ((alpha_prod_t_prev.sqrt() * pred_original_sample)? + pred_sample_direction)?;

        Ok(prev_sample)
    }}
}}

pub struct EulerScheduler {{
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
}}

impl EulerScheduler {{
    pub fn new(num_train_timesteps: usize, num_inference_steps: usize) -> Self {{
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .map(|i| i * step_ratio)
            .rev()
            .collect();

        let sigmas: Vec<f64> = timesteps.iter()
            .map(|&t| {{
                let t_norm = (t as f64) / (num_train_timesteps as f64);
                ((1.0 - t_norm) / t_norm).sqrt()
            }})
            .collect();

        Self {{ timesteps, sigmas }}
    }}
}}

impl Scheduler for EulerScheduler {{
    fn timesteps(&self) -> &[usize] {{
        &self.timesteps
    }}

    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {{
        let sigma = self.sigmas.get(timestep).copied().unwrap_or(0.0);
        let pred_original_sample = (sample - (sigma * model_output)?)?;
        Ok(pred_original_sample)
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_ddim_timesteps() {{
        let scheduler = DDIMScheduler::new(1000, 20);
        assert_eq\!(scheduler.timesteps().len(), 20);
    }}

    #[test]
    fn test_euler_timesteps() {{
        let scheduler = EulerScheduler::new(1000, 20);
        assert_eq\!(scheduler.timesteps().len(), 20);
    }}
}}
