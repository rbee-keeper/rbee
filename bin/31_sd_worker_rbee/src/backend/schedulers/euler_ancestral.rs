// TEAM-481: Euler Ancestral Discrete Scheduler (Full Implementation)
//
// Ancestral sampling with Euler method steps - better quality than regular Euler.
// Adds stochastic noise for improved sample diversity and prevents mode collapse.
//
// Based on Katherine Crowson's k-diffusion implementation:
// https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L72
//
// Key features:
// - Stochastic ancestral sampling (adds noise at each step)
// - Flexible timestep spacing (Leading, Trailing, Linspace)
// - Multiple beta schedules (Linear, ScaledLinear, SquaredcosCapV2)
// - Sigma-based noise scheduling
// - Linear interpolation for smooth sigma transitions

use super::traits::{Scheduler, SchedulerConfig};
use super::types::{BetaSchedule, PredictionType, TimestepSpacing};
use crate::error::Result;
use candle_core::Tensor;

// TEAM-481: Euler Ancestral scheduler constants
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;
const STEPS_OFFSET: usize = 1;
const INITIAL_ALPHA_PROD: f64 = 1.0;
const DEFAULT_SIGMA: f64 = 1.0;

/// Noise generation strategy for ancestral sampling
#[derive(Debug, Clone, Copy)]
pub enum NoiseStrategy {
    /// Standard Gaussian noise (default)
    Gaussian,
    /// Scaled noise based on sigma ratio
    Scaled,
}

impl Default for NoiseStrategy {
    fn default() -> Self {
        Self::Gaussian
    }
}

/// Configuration for Euler Ancestral Discrete scheduler
/// 
/// TEAM-481: Full-featured configuration with all ancestral sampling options
#[derive(Debug, Clone, Copy)]
pub struct EulerAncestralSchedulerConfig {
    /// Beta schedule start value
    pub beta_start: f64,
    /// Beta schedule end value
    pub beta_end: f64,
    /// Beta schedule type
    pub beta_schedule: BetaSchedule,
    /// Offset for timestep indexes
    pub steps_offset: usize,
    /// Prediction type (epsilon, v_prediction, or sample)
    pub prediction_type: PredictionType,
    /// Number of training timesteps (usually 1000)
    pub train_timesteps: usize,
    /// Timestep spacing strategy
    pub timestep_spacing: TimestepSpacing,
    /// Noise generation strategy
    pub noise_strategy: NoiseStrategy,
    /// Eta parameter for noise scaling (0.0 = deterministic, 1.0 = full ancestral)
    pub eta: f64,
}

impl Default for EulerAncestralSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: BETA_START,
            beta_end: BETA_END,
            beta_schedule: BetaSchedule::ScaledLinear,
            steps_offset: STEPS_OFFSET,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
            timestep_spacing: TimestepSpacing::Leading,
            noise_strategy: NoiseStrategy::Gaussian,
            eta: 1.0, // Full ancestral sampling by default
        }
    }
}

impl SchedulerConfig for EulerAncestralSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(EulerAncestralScheduler::new(
            inference_steps,
            *self,
        )?))
    }
}

/// Euler Ancestral Discrete Scheduler
/// 
/// TEAM-481: Better quality than regular Euler due to stochastic noise
pub struct EulerAncestralScheduler {
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    config: EulerAncestralSchedulerConfig,
}

impl EulerAncestralScheduler {
    /// Create a new Euler Ancestral scheduler
    /// 
    /// # Arguments
    /// * `inference_steps` - Number of inference steps (e.g., 20, 50)
    /// * `config` - Scheduler configuration
    pub fn new(
        inference_steps: usize,
        config: EulerAncestralSchedulerConfig,
    ) -> Result<Self> {
        let step_ratio = config.train_timesteps / inference_steps;
        
        // Create timesteps based on spacing strategy
        let timesteps: Vec<usize> = match config.timestep_spacing {
            TimestepSpacing::Leading => {
                (0..inference_steps)
                    .map(|s| s * step_ratio + config.steps_offset)
                    .rev()
                    .collect()
            }
            TimestepSpacing::Trailing => {
                std::iter::successors(Some(config.train_timesteps), |n| {
                    if *n > step_ratio {
                        Some(n - step_ratio)
                    } else {
                        None
                    }
                })
                .map(|n| n - 1)
                .collect()
            }
            TimestepSpacing::Linspace => {
                // Simplified linspace for timesteps
                (0..inference_steps)
                    .map(|i| {
                        let t = (i as f64) / (inference_steps as f64 - 1.0);
                        ((config.train_timesteps - 1) as f64 * (1.0 - t)) as usize
                    })
                    .collect()
            }
        };

        // Create beta schedule
        let betas: Vec<f64> = match config.beta_schedule {
            BetaSchedule::Linear => {
                (0..config.train_timesteps)
                    .map(|i| {
                        let t = (i as f64) / (config.train_timesteps as f64 - 1.0);
                        config.beta_start + t * (config.beta_end - config.beta_start)
                    })
                    .collect()
            }
            BetaSchedule::ScaledLinear => {
                (0..config.train_timesteps)
                    .map(|i| {
                        let t = (i as f64) / (config.train_timesteps as f64 - 1.0);
                        let beta_sqrt = config.beta_start.sqrt() + t * (config.beta_end.sqrt() - config.beta_start.sqrt());
                        beta_sqrt * beta_sqrt
                    })
                    .collect()
            }
            BetaSchedule::SquaredcosCapV2 => {
                // Simplified cosine schedule
                (0..config.train_timesteps)
                    .map(|i| {
                        let t = (i as f64) / config.train_timesteps as f64;
                        let alpha_bar = f64::cos((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2);
                        (1.0 - alpha_bar).min(0.999)
                    })
                    .collect()
            }
        };

        // Calculate cumulative product of alphas
        let mut alphas_cumprod = Vec::with_capacity(config.train_timesteps);
        let mut alpha_prod = INITIAL_ALPHA_PROD;
        for beta in &betas {
            alpha_prod *= INITIAL_ALPHA_PROD - beta;
            alphas_cumprod.push(alpha_prod);
        }

        // Calculate sigmas from alphas
        let sigmas: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&alpha| ((1.0 - alpha) / alpha).sqrt())
            .collect();

        // Interpolate sigmas for inference timesteps
        let sigmas_xa: Vec<f64> = (0..sigmas.len()).map(|i| i as f64).collect();
        let timesteps_f64: Vec<f64> = timesteps.iter().map(|&t| t as f64).collect();
        
        let mut sigmas_int = interp(&timesteps_f64, &sigmas_xa, &sigmas);
        sigmas_int.push(0.0); // Add final sigma

        // Calculate initial noise sigma (max of all sigmas)
        let init_noise_sigma = sigmas_int
            .iter()
            .fold(0.0_f64, |a, &b| if a > b { a } else { b });

        Ok(Self {
            timesteps,
            sigmas: sigmas_int,
            init_noise_sigma,
            config,
        })
    }
}

impl Scheduler for EulerAncestralScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| candle_core::Error::Msg("timestep out of bounds".to_string()))?;

        let sigma = self.sigmas.get(step_index)
            .ok_or_else(|| candle_core::Error::Msg("step_index out of sigma bounds".to_string()))?;

        Ok((original + (noise * *sigma)?)?)
    }

    fn init_noise_sigma(&self) -> f64 {
        match self.config.timestep_spacing {
            TimestepSpacing::Trailing | TimestepSpacing::Linspace => self.init_noise_sigma,
            TimestepSpacing::Leading => (self.init_noise_sigma.powi(2) + 1.0).sqrt(),
        }
    }

    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> Result<Tensor> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| candle_core::Error::Msg("timestep out of bounds".to_string()))?;

        let sigma = self.sigmas.get(step_index)
            .ok_or_else(|| candle_core::Error::Msg("step_index out of sigma bounds".to_string()))?;

        // Scale by (sigma^2 + 1)^0.5 to match K-LMS algorithm
        Ok((sample / ((sigma.powi(2) + 1.0).sqrt()))?)
    }

    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| candle_core::Error::Msg("timestep out of bounds".to_string()))?;

        let sigma_from = self.sigmas[step_index];
        let sigma_to = self.sigmas[step_index + 1];

        // 1. Compute predicted original sample (x_0) from sigma-scaled predicted noise
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                (sample - (model_output * sigma_from)?)?
            }
            PredictionType::VPrediction => {
                let scale1 = -sigma_from / (sigma_from.powi(2) + 1.0).sqrt();
                let scale2 = 1.0 / (sigma_from.powi(2) + 1.0);
                ((model_output * scale1)? + (sample * scale2)?)?
            }
            PredictionType::Sample => {
                return Err(candle_core::Error::Msg("Sample prediction type not implemented".to_string()).into());
            }
        };

        // Calculate sigma_up and sigma_down for ancestral sampling
        // eta controls the amount of stochastic noise (0.0 = deterministic, 1.0 = full ancestral)
        let sigma_up = self.config.eta * (sigma_to.powi(2) * (sigma_from.powi(2) - sigma_to.powi(2)) / sigma_from.powi(2)).sqrt();
        let sigma_down = (sigma_to.powi(2) - sigma_up.powi(2)).sqrt();

        // 2. Convert to ODE derivative
        let derivative = ((sample - &pred_original_sample)? / sigma_from)?;
        let dt = sigma_down - sigma_from;
        let prev_sample = (sample + (derivative * dt)?)?;

        // 3. Add stochastic noise (ancestral sampling)
        if sigma_up > 0.0 {
            let noise = match self.config.noise_strategy {
                NoiseStrategy::Gaussian => {
                    // Standard Gaussian noise
                    prev_sample.randn_like(0.0, 1.0)?
                }
                NoiseStrategy::Scaled => {
                    // Scaled noise based on sigma ratio
                    let scale = sigma_up / sigma_from;
                    (prev_sample.randn_like(0.0, 1.0)? * scale)?
                }
            };
            Ok((prev_sample + (noise * sigma_up)?)?)
        } else {
            // Deterministic mode (eta = 0.0)
            Ok(prev_sample)
        }
    }
}

/// Linear interpolation helper
/// 
/// TEAM-481: Interpolates y values at x positions given xp and fp arrays
fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let mut interpolator = LinearInterpolator { xp, fp, cache: 0 };
    x.iter().map(|&x_val| interpolator.eval(x_val)).collect()
}

/// Linear interpolator for sorted arrays
struct LinearInterpolator<'x, 'y> {
    xp: &'x [f64],
    fp: &'y [f64],
    cache: usize,
}

impl LinearInterpolator<'_, '_> {
    fn accel_find(&mut self, x: f64) -> usize {
        let xidx = self.cache;
        if x < self.xp[xidx] {
            self.cache = self.xp[0..xidx].partition_point(|o| *o < x);
            self.cache = self.cache.saturating_sub(1);
        } else if xidx + 1 < self.xp.len() && x >= self.xp[xidx + 1] {
            self.cache = self.xp[xidx..self.xp.len()].partition_point(|o| *o < x) + xidx;
            self.cache = self.cache.saturating_sub(1);
        }
        self.cache
    }

    fn eval(&mut self, x: f64) -> f64 {
        if x < self.xp[0] || x > self.xp[self.xp.len() - 1] {
            return f64::NAN;
        }

        let idx = self.accel_find(x);
        
        if idx + 1 >= self.xp.len() {
            return self.fp[idx];
        }

        let x_l = self.xp[idx];
        let x_h = self.xp[idx + 1];
        let y_l = self.fp[idx];
        let y_h = self.fp[idx + 1];
        let dx = x_h - x_l;
        
        if dx > 0.0 {
            y_l + (x - x_l) / dx * (y_h - y_l)
        } else {
            f64::NAN
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_ancestral_scheduler_creation() {
        let config = EulerAncestralSchedulerConfig::default();
        let scheduler = config.build(20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_euler_ancestral_timesteps() {
        let config = EulerAncestralSchedulerConfig::default();
        let scheduler = EulerAncestralScheduler::new(20, config).unwrap();
        assert_eq!(scheduler.timesteps.len(), 20);
        assert!(scheduler.sigmas.len() == 21); // timesteps + 1 for final sigma
    }

    #[test]
    fn test_euler_ancestral_init_noise_sigma() {
        let config = EulerAncestralSchedulerConfig::default();
        let scheduler = EulerAncestralScheduler::new(20, config).unwrap();
        let sigma = scheduler.init_noise_sigma();
        assert!(sigma > 0.0);
    }

    #[test]
    fn test_interp() {
        let x = vec![0.5, 1.5, 2.5];
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![0.0, 10.0, 20.0, 30.0];
        let result = interp(&x, &xp, &fp);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 0.001);
        assert!((result[1] - 15.0).abs() < 0.001);
        assert!((result[2] - 25.0).abs() < 0.001);
    }
}
