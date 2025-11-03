// TEAM-390: SD model configuration structures
//
// Configuration for different SD model architectures.

use super::SDVersion;

/// Configuration for Stable Diffusion models
#[derive(Debug, Clone)]
pub struct SDConfig {
    pub version: SDVersion,
    pub height: usize,
    pub width: usize,
    pub guidance_scale: f64,
    pub n_steps: usize,
    pub use_f16: bool,
    pub use_flash_attn: bool,
    pub sliced_attention_size: Option<usize>,
}

impl SDConfig {
    /// Create default configuration for a model version
    pub fn new(version: SDVersion) -> Self {
        let (width, height) = version.default_size();
        Self {
            version,
            height,
            width,
            guidance_scale: version.default_guidance_scale(),
            n_steps: version.default_steps(),
            use_f16: false,
            use_flash_attn: false,
            sliced_attention_size: None,
        }
    }

    /// Set image dimensions (must be multiples of 8)
    pub fn with_size(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set guidance scale
    pub fn with_guidance_scale(mut self, scale: f64) -> Self {
        self.guidance_scale = scale;
        self
    }

    /// Set number of inference steps
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.n_steps = steps;
        self
    }

    /// Enable FP16 precision
    pub fn with_f16(mut self, use_f16: bool) -> Self {
        self.use_f16 = use_f16;
        self
    }

    /// Enable flash attention (CUDA only)
    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }

    /// Set sliced attention size
    pub fn with_sliced_attention(mut self, size: Option<usize>) -> Self {
        self.sliced_attention_size = size;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        // Check dimensions are multiples of 8
        if self.width % 8 != 0 || self.height % 8 != 0 {
            return Err(crate::error::Error::InvalidInput(format!(
                "Image dimensions must be multiples of 8, got {}x{}",
                self.width, self.height
            )));
        }

        // Check reasonable step count
        if self.n_steps == 0 || self.n_steps > 150 {
            return Err(crate::error::Error::InvalidInput(format!(
                "Number of steps must be between 1 and 150, got {}",
                self.n_steps
            )));
        }

        // Check guidance scale
        if self.guidance_scale < 0.0 || self.guidance_scale > 20.0 {
            return Err(crate::error::Error::InvalidInput(format!(
                "Guidance scale must be between 0.0 and 20.0, got {}",
                self.guidance_scale
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SDConfig::new(SDVersion::V1_5);
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert_eq!(config.n_steps, 20);
    }

    #[test]
    fn test_validation() {
        let config = SDConfig::new(SDVersion::V1_5);
        assert!(config.validate().is_ok());

        // Invalid dimensions
        let bad_config = config.clone().with_size(513, 512);
        assert!(bad_config.validate().is_err());

        // Invalid steps
        let bad_config = config.clone().with_steps(0);
        assert!(bad_config.validate().is_err());

        // Invalid guidance
        let bad_config = config.clone().with_guidance_scale(25.0);
        assert!(bad_config.validate().is_err());
    }
}
