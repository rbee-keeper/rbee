// TEAM-488: FLUX configuration
// Model-specific settings for FLUX.1-dev and FLUX.1-schnell

use super::super::SDVersion;

/// FLUX model configuration
#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub version: SDVersion,
    pub default_steps: usize,
    pub default_guidance: f64,
    pub use_time_shift: bool,
}

impl FluxConfig {
    /// Get configuration for FLUX.1-dev
    pub fn dev() -> Self {
        Self {
            version: SDVersion::FluxDev,
            default_steps: 50,
            default_guidance: 3.5,
            use_time_shift: true,
        }
    }

    /// Get configuration for FLUX.1-schnell
    pub fn schnell() -> Self {
        Self {
            version: SDVersion::FluxSchnell,
            default_steps: 4,
            default_guidance: 0.0, // Schnell doesn't use guidance
            use_time_shift: false,
        }
    }
}
