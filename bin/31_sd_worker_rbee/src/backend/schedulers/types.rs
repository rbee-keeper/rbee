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

/// Sampler type enum - the sampling algorithm
///
/// TEAM-482: Separated from noise schedules for proper architecture.
/// Samplers define HOW we sample, schedules define the noise curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerType {
    /// DDIM sampler (good quality)
    Ddim,
    /// Euler sampler (fast and stable)
    Euler,
    /// DDPM sampler (probabilistic, good for inpainting)
    Ddpm,
    /// Euler Ancestral sampler (better quality than Euler, stochastic)
    EulerAncestral,
    /// DPM-Solver++ Multistep (fast, high-quality, popular in ComfyUI/A1111)
    DpmSolverMultistep,
    // TEAM-482: Add new samplers here!
}

impl Default for SamplerType {
    fn default() -> Self {
        Self::Euler // TEAM-482: Euler is fast and stable
    }
}

impl std::fmt::Display for SamplerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ddim => write!(f, "ddim"),
            Self::Euler => write!(f, "euler"),
            Self::Ddpm => write!(f, "ddpm"),
            Self::EulerAncestral => write!(f, "euler_ancestral"),
            Self::DpmSolverMultistep => write!(f, "dpm_solver_multistep"),
        }
    }
}

impl std::str::FromStr for SamplerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ddim" => Ok(Self::Ddim),
            "euler" => Ok(Self::Euler),
            "ddpm" => Ok(Self::Ddpm),
            "euler_ancestral" => Ok(Self::EulerAncestral),
            "dpm_solver_multistep" | "dpm++" | "dpmpp" => Ok(Self::DpmSolverMultistep),
            _ => Err(format!("Unknown sampler type: {s}")),
        }
    }
}

/// Noise schedule type - defines the noise curve
///
/// TEAM-482: Noise schedules control HOW noise is distributed across timesteps.
/// Different schedules can dramatically affect image quality.
/// Karras is very popular for high-quality results!
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseSchedule {
    /// Simple linear schedule (most compatible)
    Simple,
    /// Karras schedule (very popular! high quality)
    Karras,
    /// Exponential schedule
    Exponential,
    /// SGM uniform schedule
    SgmUniform,
    /// DDIM uniform schedule
    DdimUniform,
}

impl Default for NoiseSchedule {
    fn default() -> Self {
        Self::Simple // TEAM-482: Simple is most compatible
    }
}

impl std::fmt::Display for NoiseSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Karras => write!(f, "karras"),
            Self::Exponential => write!(f, "exponential"),
            Self::SgmUniform => write!(f, "sgm_uniform"),
            Self::DdimUniform => write!(f, "ddim_uniform"),
        }
    }
}

impl std::str::FromStr for NoiseSchedule {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simple" => Ok(Self::Simple),
            "karras" => Ok(Self::Karras),
            "exponential" => Ok(Self::Exponential),
            "sgm_uniform" => Ok(Self::SgmUniform),
            "ddim_uniform" => Ok(Self::DdimUniform),
            _ => Err(format!("Unknown noise schedule: {s}")),
        }
    }
}

// TEAM-482: SchedulerType removed - use SamplerType instead
// RULE ZERO: Breaking changes > backwards compatibility
// The compiler will find all call sites. Fix them.
