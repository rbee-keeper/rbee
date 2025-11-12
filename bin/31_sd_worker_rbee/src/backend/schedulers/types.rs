// TEAM-481: Scheduler types - shared enums and types
//
// These types are used across multiple schedulers for configuration.

use serde::{Deserialize, Serialize};

/// How beta ranges from minimum to maximum during training
/// 
/// TEAM-481: Different beta schedules affect the noise schedule
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum BetaSchedule {
    /// Linear interpolation
    Linear,
    /// Linear interpolation of the square root of beta (default)
    #[default]
    ScaledLinear,
    /// Glide cosine schedule
    SquaredcosCapV2,
}

/// Prediction type for the scheduler
/// 
/// TEAM-481: Determines what the model is predicting
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PredictionType {
    /// Predicting the noise (epsilon)
    Epsilon,
    /// Predicting the velocity
    VPrediction,
    /// Directly predicting the sample
    Sample,
}

/// Time step spacing for the diffusion process
/// 
/// TEAM-481: Controls how timesteps are distributed
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum TimestepSpacing {
    /// Leading spacing (default)
    #[default]
    Leading,
    /// Linear spacing
    Linspace,
    /// Trailing spacing
    Trailing,
}

/// Scheduler type enum for user selection
/// 
/// TEAM-481: This enum makes it easy to add new schedulers.
/// Just add a new variant here and implement the scheduler!
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerType {
    /// DDIM scheduler (default, good quality)
    Ddim,
    /// Euler scheduler (faster)
    Euler,
    /// DDPM scheduler (probabilistic, good for inpainting)
    Ddpm,
    // TEAM-481: Add new schedulers here!
    // EulerAncestral,
    // UniPc,
    // DpmPlusPlus2M,
}

impl Default for SchedulerType {
    fn default() -> Self {
        Self::Ddim
    }
}

impl std::fmt::Display for SchedulerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ddim => write!(f, "ddim"),
            Self::Euler => write!(f, "euler"),
            Self::Ddpm => write!(f, "ddpm"),
        }
    }
}

impl std::str::FromStr for SchedulerType {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ddim" => Ok(Self::Ddim),
            "euler" => Ok(Self::Euler),
            "ddpm" => Ok(Self::Ddpm),
            _ => Err(format!("Unknown scheduler type: {}", s)),
        }
    }
}
