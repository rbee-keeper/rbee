// Created by: TEAM-392
// TEAM-392: Sampling configuration for Stable Diffusion
// TEAM-487: Added LoRA support
// TEAM-481: Added constants for validation limits
// TEAM-481: Added scheduler selection

use crate::backend::lora::LoRAConfig;
use crate::backend::schedulers::{NoiseSchedule, SamplerType};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

// TEAM-481: Validation constants - single source of truth
const MIN_STEPS: usize = 1;
const MAX_STEPS: usize = 150;
const MIN_GUIDANCE: f64 = 0.0;
const MAX_GUIDANCE: f64 = 20.0;
const DIMENSION_MULTIPLE: usize = 8;
const MIN_DIMENSION: usize = 256;
const MAX_DIMENSION: usize = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub width: usize,
    pub height: usize,

    /// Sampler to use (algorithm)
    /// TEAM-482: Choose from Euler, EulerAncestral, DpmSolverMultistep, Ddim, Ddpm
    /// Defaults to Euler if not specified
    #[serde(default)]
    pub sampler: SamplerType,

    /// Noise schedule to use
    /// TEAM-482: Choose from Simple, Karras, Exponential, etc.
    /// Karras is popular for high-quality results
    /// Defaults to Simple if not specified
    #[serde(default)]
    pub schedule: NoiseSchedule,

    /// LoRAs to apply (optional)
    /// TEAM-487: Allows stacking multiple LoRAs for customization
    #[serde(default)]
    pub loras: Vec<LoRAConfig>,
}

impl SamplingConfig {
    /// Validate sampling configuration
    ///
    /// TEAM-481: Now uses constants for validation limits
    /// TEAM-481: #[must_use] ensures validation result is checked
    #[must_use = "validation result must be checked"]
    pub fn validate(&self) -> Result<()> {
        if self.prompt.is_empty() {
            return Err(Error::InvalidInput("Prompt cannot be empty".to_string()));
        }

        if self.steps < MIN_STEPS || self.steps > MAX_STEPS {
            return Err(Error::InvalidInput(format!(
                "Steps must be between {} and {}, got {}",
                MIN_STEPS, MAX_STEPS, self.steps
            )));
        }

        if self.guidance_scale < MIN_GUIDANCE || self.guidance_scale > MAX_GUIDANCE {
            return Err(Error::InvalidInput(format!(
                "Guidance scale must be between {} and {}, got {}",
                MIN_GUIDANCE, MAX_GUIDANCE, self.guidance_scale
            )));
        }

        if self.width % DIMENSION_MULTIPLE != 0 || self.height % DIMENSION_MULTIPLE != 0 {
            return Err(Error::InvalidInput(format!(
                "Width and height must be multiples of {}, got {}x{}",
                DIMENSION_MULTIPLE, self.width, self.height
            )));
        }

        if self.width < MIN_DIMENSION || self.width > MAX_DIMENSION {
            return Err(Error::InvalidInput(format!(
                "Width must be between {} and {}, got {}",
                MIN_DIMENSION, MAX_DIMENSION, self.width
            )));
        }

        if self.height < MIN_DIMENSION || self.height > MAX_DIMENSION {
            return Err(Error::InvalidInput(format!(
                "Height must be between {} and {}, got {}",
                MIN_DIMENSION, MAX_DIMENSION, self.height
            )));
        }

        // TEAM-487: Validate LoRA configurations
        for lora in &self.loras {
            lora.validate()?;
        }

        Ok(())
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
            width: 512,
            height: 512,
            sampler: SamplerType::default(), // TEAM-482: Defaults to Euler
            schedule: NoiseSchedule::default(), // TEAM-482: Defaults to Simple
            loras: vec![],                   // TEAM-487: No LoRAs by default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let mut config = SamplingConfig::default();
        config.prompt = "test".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut config = SamplingConfig::default();
        config.prompt = "test".to_string();
        config.width = 513; // Not multiple of 8
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_steps() {
        let mut config = SamplingConfig::default();
        config.prompt = "test".to_string();
        config.steps = 0;
        assert!(config.validate().is_err());
    }
}
