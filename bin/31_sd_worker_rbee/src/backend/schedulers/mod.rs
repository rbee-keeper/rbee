// TEAM-481: Schedulers module - modular scheduler architecture
//
// This module provides a clean, extensible architecture for diffusion schedulers.
// Each scheduler is in its own file and implements the Scheduler trait.
//
// ## How to add a new scheduler:
//
// 1. Create a new file: `schedulers/my_scheduler.rs`
// 2. Define a Config struct that implements `SchedulerConfig`
// 3. Define a Scheduler struct that implements `Scheduler`
// 4. Add the module here: `pub mod my_scheduler;`
// 5. Add a variant to `SchedulerType` in `types.rs`
// 6. Add a match arm in `build_scheduler()` below
//
// That's it! Your scheduler is now available to users.

pub mod traits;
pub mod types;

// TEAM-482: Noise schedule calculations
pub mod noise_schedules;

// ✅ SHARED: Sigma schedule implementations (used by all schedulers)
pub mod sigma_schedules;

// TEAM-481: Scheduler implementations
pub mod ddim;
pub mod ddpm;
pub mod dpm_solver_multistep;
pub mod euler;
pub mod euler_ancestral;

// TEAM-489: UniPC scheduler - ✅ FULLY IMPLEMENTED
pub mod uni_pc;

// TEAM-481: Re-exports for convenience
pub use traits::{Scheduler, SchedulerConfig};
pub use types::{BetaSchedule, PredictionType, TimestepSpacing};

// TEAM-482: Re-export new types
pub use types::{NoiseSchedule, SamplerType};

// TEAM-481: Re-export configs for easy access
pub use ddim::DDIMSchedulerConfig;
pub use ddpm::DDPMSchedulerConfig;
pub use dpm_solver_multistep::DPMSolverMultistepSchedulerConfig;
pub use euler::EulerSchedulerConfig;
pub use euler_ancestral::EulerAncestralSchedulerConfig;

// TEAM-489: UniPC config (STUB)
pub use uni_pc::UniPCSchedulerConfig;

use crate::error::Result;

/// Build a scheduler from a `SamplerType` and `NoiseSchedule`
///
/// TEAM-482: Now accepts both sampler and schedule for proper architecture.
/// Euler scheduler uses the noise schedule, others use their beta schedules.
///
/// # Arguments
/// * `sampler` - The sampler to use (Euler, DDIM, etc.)
/// * `schedule` - The noise schedule to use (Simple, Karras, etc.)
/// * `inference_steps` - Number of inference steps
///
/// # Returns
/// A boxed scheduler ready to use
///
/// # Example
/// ```ignore
/// let scheduler = build_scheduler(SamplerType::Euler, NoiseSchedule::Karras, 20)?;
/// ```
pub fn build_scheduler(
    sampler: SamplerType,
    schedule: NoiseSchedule,
    inference_steps: usize,
) -> Result<Box<dyn Scheduler>> {
    match sampler {
        SamplerType::Ddim => {
            let config = DDIMSchedulerConfig::default();
            config.build(inference_steps)
        }
        SamplerType::Euler => {
            // TEAM-482: Euler uses noise schedule!
            let config = EulerSchedulerConfig { train_timesteps: 1000, noise_schedule: schedule };
            config.build(inference_steps)
        }
        SamplerType::Ddpm => {
            let config = DDPMSchedulerConfig::default();
            config.build(inference_steps)
        }
        SamplerType::EulerAncestral => {
            let config = EulerAncestralSchedulerConfig::default();
            config.build(inference_steps)
        }
        SamplerType::DpmSolverMultistep => {
            let config = DPMSolverMultistepSchedulerConfig::default();
            config.build(inference_steps)
        }
        // TEAM-489: UniPC sampler (STUB - will panic until implemented)
        SamplerType::UniPc => {
            let config = UniPCSchedulerConfig::default();
            config.build(inference_steps)
        } // TEAM-482: Add new samplers here!
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ddim_scheduler() {
        // TEAM-482: Updated to use new signature
        let scheduler = build_scheduler(SamplerType::Ddim, NoiseSchedule::Simple, 20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_build_euler_scheduler() {
        // TEAM-482: Updated to use new signature
        let scheduler = build_scheduler(SamplerType::Euler, NoiseSchedule::Simple, 20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_build_euler_with_karras() {
        // TEAM-482: Test Karras schedule with Euler
        let scheduler = build_scheduler(SamplerType::Euler, NoiseSchedule::Karras, 20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_sampler_type_from_str() {
        use std::str::FromStr;

        // TEAM-482: Updated to use SamplerType
        assert!(matches!(SamplerType::from_str("ddim"), Ok(SamplerType::Ddim)));
        assert!(matches!(SamplerType::from_str("euler"), Ok(SamplerType::Euler)));
        assert!(matches!(SamplerType::from_str("DDIM"), Ok(SamplerType::Ddim)));
        assert!(SamplerType::from_str("unknown").is_err());
    }

    #[test]
    fn test_noise_schedule_from_str() {
        use std::str::FromStr;

        // TEAM-482: Test NoiseSchedule parsing
        assert!(matches!(NoiseSchedule::from_str("simple"), Ok(NoiseSchedule::Simple)));
        assert!(matches!(NoiseSchedule::from_str("karras"), Ok(NoiseSchedule::Karras)));
        assert!(matches!(NoiseSchedule::from_str("exponential"), Ok(NoiseSchedule::Exponential)));
        assert!(NoiseSchedule::from_str("unknown").is_err());
    }
}
