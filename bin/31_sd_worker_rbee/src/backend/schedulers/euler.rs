// TEAM-481: Euler Scheduler - Simple and fast
//
// Euler scheduler is faster than DDIM and provides good quality.
// It's commonly used with FLUX and other modern models.

use super::noise_schedules::calculate_sigmas;
use super::traits::{Scheduler, SchedulerConfig};
use super::types::NoiseSchedule;
use crate::error::Result;
use candle_core::Tensor;

// TEAM-481: Euler scheduler constants
const INITIAL_ALPHA_PROD: f64 = 1.0;
const DEFAULT_SIGMA: f64 = 0.0;
// TEAM-482: Default sigma range for noise schedules
const DEFAULT_SIGMA_MIN: f64 = 0.0292;
const DEFAULT_SIGMA_MAX: f64 = 14.6146;

/// Configuration for Euler scheduler
///
/// TEAM-481: This struct holds all configuration for Euler.
/// TEAM-482: Now supports different noise schedules (Karras, Exponential, etc.)
/// Implements Default for easy instantiation.
#[derive(Debug, Clone, Copy)]
pub struct EulerSchedulerConfig {
    /// Number of training timesteps (usually 1000)
    pub train_timesteps: usize,
    /// Noise schedule to use
    /// TEAM-482: Choose from Simple, Karras, Exponential, etc.
    pub noise_schedule: NoiseSchedule,
}

impl Default for EulerSchedulerConfig {
    fn default() -> Self {
        Self {
            train_timesteps: 1000,
            noise_schedule: NoiseSchedule::default(), // TEAM-482: Simple by default
        }
    }
}

impl SchedulerConfig for EulerSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(EulerScheduler::new(
            self.train_timesteps,
            inference_steps,
            self.noise_schedule, // TEAM-482: Pass noise schedule
        )))
    }
}

/// Euler Scheduler implementation
///
/// TEAM-481: Fast scheduler with good quality
pub struct EulerScheduler {
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
}

impl EulerScheduler {
    /// Create a new Euler scheduler
    ///
    /// TEAM-482: Now supports different noise schedules
    ///
    /// # Arguments
    /// * `num_train_timesteps` - Number of training timesteps (usually 1000)
    /// * `num_inference_steps` - Number of inference steps (e.g., 20, 50)
    /// * `noise_schedule` - Noise schedule to use (Simple, Karras, Exponential, etc.)
    pub fn new(
        num_train_timesteps: usize,
        num_inference_steps: usize,
        noise_schedule: NoiseSchedule,
    ) -> Self {
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> =
            (0..num_inference_steps).map(|i| i * step_ratio).rev().collect();

        // TEAM-482: Calculate sigmas using the specified noise schedule
        let sigmas = calculate_sigmas(
            noise_schedule,
            num_inference_steps,
            DEFAULT_SIGMA_MIN,
            DEFAULT_SIGMA_MAX,
        );

        Self { timesteps, sigmas }
    }
}

impl Scheduler for EulerScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(timestep).copied().unwrap_or(DEFAULT_SIGMA);
        Ok((original + (noise * sigma))?)
    }

    fn init_noise_sigma(&self) -> f64 {
        self.sigmas.first().copied().unwrap_or(INITIAL_ALPHA_PROD)
    }

    fn scale_model_input(&self, sample: Tensor, _timestep: usize) -> Result<Tensor> {
        // Euler doesn't scale the model input
        Ok(sample)
    }

    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas.get(timestep).copied().unwrap_or(DEFAULT_SIGMA);
        let pred_original_sample = (sample - (sigma * model_output)?)?;
        Ok(pred_original_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_scheduler_creation() {
        let config = EulerSchedulerConfig::default();
        let scheduler = config.build(20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_euler_timesteps() {
        let scheduler = EulerScheduler::new(1000, 20, NoiseSchedule::Simple);
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_euler_with_karras() {
        // TEAM-482: Test Karras schedule
        let scheduler = EulerScheduler::new(1000, 20, NoiseSchedule::Karras);
        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.sigmas.len(), 21); // 20 steps + final 0.0
    }
}
