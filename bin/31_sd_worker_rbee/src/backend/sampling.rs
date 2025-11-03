// Created by: TEAM-392
// TEAM-392: Sampling configuration for Stable Diffusion

use crate::error::{{Error, Result}};
use serde::{{Deserialize, Serialize}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {{
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub width: usize,
    pub height: usize,
}}

impl SamplingConfig {{
    pub fn validate(&self) -> Result<()> {{
        if self.prompt.is_empty() {{
            return Err(Error::InvalidInput("Prompt cannot be empty".to_string()));
        }}

        if self.steps == 0 || self.steps > 150 {{
            return Err(Error::InvalidInput(format\!(
                "Steps must be between 1 and 150, got {{}}",
                self.steps
            )));
        }}

        if self.guidance_scale < 0.0 || self.guidance_scale > 20.0 {{
            return Err(Error::InvalidInput(format\!(
                "Guidance scale must be between 0 and 20, got {{}}",
                self.guidance_scale
            )));
        }}

        if self.width % 8 \!= 0 || self.height % 8 \!= 0 {{
            return Err(Error::InvalidInput(format\!(
                "Width and height must be multiples of 8, got {{}}x{{}}",
                self.width, self.height
            )));
        }}

        if self.width < 256 || self.width > 2048 {{
            return Err(Error::InvalidInput(format\!(
                "Width must be between 256 and 2048, got {{}}",
                self.width
            )));
        }}

        if self.height < 256 || self.height > 2048 {{
            return Err(Error::InvalidInput(format\!(
                "Height must be between 256 and 2048, got {{}}",
                self.height
            )));
        }}

        Ok(())
    }}
}}

impl Default for SamplingConfig {{
    fn default() -> Self {{
        Self {{
            prompt: String::new(),
            negative_prompt: None,
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
            width: 512,
            height: 512,
        }}
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_default_config_valid() {{
        let mut config = SamplingConfig::default();
        config.prompt = "test".to_string();
        assert\!(config.validate().is_ok());
    }}

    #[test]
    fn test_invalid_dimensions() {{
        let mut config = SamplingConfig::default();
        config.prompt = "test".to_string();
        config.width = 513; // Not multiple of 8
        assert\!(config.validate().is_err());
    }}

    #[test]
    fn test_invalid_steps() {{
        let mut config = SamplingConfig::default();
        config.prompt = "test".to_string();
        config.steps = 0;
        assert\!(config.validate().is_err());
    }}
}}
