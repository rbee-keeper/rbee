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

// TEAM-481: Scheduler implementations
pub mod ddim;
pub mod ddpm;
pub mod euler;

// TEAM-481: Re-exports for convenience
pub use traits::{Scheduler, SchedulerConfig};
pub use types::{BetaSchedule, PredictionType, SchedulerType, TimestepSpacing};

// TEAM-481: Re-export configs for easy access
pub use ddim::DDIMSchedulerConfig;
pub use ddpm::DDPMSchedulerConfig;
pub use euler::EulerSchedulerConfig;

use crate::error::Result;

/// Build a scheduler from a SchedulerType
/// 
/// TEAM-481: This is the main entry point for creating schedulers.
/// It uses the default configuration for each scheduler type.
/// 
/// # Arguments
/// * `scheduler_type` - The type of scheduler to build
/// * `inference_steps` - Number of inference steps
/// 
/// # Returns
/// A boxed scheduler ready to use
/// 
/// # Example
/// ```ignore
/// let scheduler = build_scheduler(SchedulerType::Ddim, 20)?;
/// ```
pub fn build_scheduler(
    scheduler_type: SchedulerType,
    inference_steps: usize,
) -> Result<Box<dyn Scheduler>> {
    match scheduler_type {
        SchedulerType::Ddim => {
            let config = DDIMSchedulerConfig::default();
            config.build(inference_steps)
        }
        SchedulerType::Euler => {
            let config = EulerSchedulerConfig::default();
            config.build(inference_steps)
        }
        SchedulerType::Ddpm => {
            let config = DDPMSchedulerConfig::default();
            config.build(inference_steps)
        }
        // TEAM-481: Add new schedulers here!
        // SchedulerType::EulerAncestral => {
        //     let config = EulerAncestralSchedulerConfig::default();
        //     config.build(inference_steps)
        // }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ddim_scheduler() {
        let scheduler = build_scheduler(SchedulerType::Ddim, 20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_build_euler_scheduler() {
        let scheduler = build_scheduler(SchedulerType::Euler, 20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }

    #[test]
    fn test_scheduler_type_from_str() {
        use std::str::FromStr;
        
        assert!(matches!(SchedulerType::from_str("ddim"), Ok(SchedulerType::Ddim)));
        assert!(matches!(SchedulerType::from_str("euler"), Ok(SchedulerType::Euler)));
        assert!(matches!(SchedulerType::from_str("DDIM"), Ok(SchedulerType::Ddim)));
        assert!(SchedulerType::from_str("unknown").is_err());
    }
}
