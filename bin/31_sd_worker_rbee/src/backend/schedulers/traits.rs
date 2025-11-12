// TEAM-481: Scheduler traits - base interface for all schedulers
//
// This follows Candle's pattern for extensible scheduler architecture.
// Each scheduler implements these traits to provide a consistent interface.

use crate::error::Result;
use candle_core::Tensor;

/// Base trait that all schedulers must implement
/// 
/// TEAM-481: This trait defines the core interface for diffusion schedulers.
/// Any new scheduler just needs to implement these methods.
pub trait Scheduler: Send + Sync {
    /// Get the timesteps for this scheduler
    fn timesteps(&self) -> &[usize];
    
    /// Perform one denoising step
    /// 
    /// # Arguments
    /// * `model_output` - The output from the UNet model
    /// * `timestep` - Current timestep in the diffusion process
    /// * `sample` - Current noisy sample
    /// 
    /// # Returns
    /// The denoised sample for the next step
    /// 
    /// TEAM-481: Takes &self (not &mut self) since schedulers are stateless
    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor>;
}

/// Configuration trait for building schedulers
/// 
/// TEAM-481: This trait allows schedulers to be built from configuration.
/// Each scheduler has a Config struct that implements this trait.
pub trait SchedulerConfig: std::fmt::Debug + Send + Sync {
    /// Build a scheduler with the given number of inference steps
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>>;
}
