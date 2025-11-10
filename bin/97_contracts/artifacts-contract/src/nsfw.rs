// TEAM-464: NSFW filtering levels for Civitai content
// Based on Civitai's 5-level system

use serde::{Deserialize, Serialize};

#[cfg(feature = "specta")]
use specta::Type;

#[cfg(target_arch = "wasm32")]
use tsify::Tsify;

/// NSFW content level (Civitai's 5-level system)
/// 
/// Levels:
/// - None: Safe for work, no mature content
/// - Soft: Suggestive but not explicit
/// - Mature: Partial nudity, mature themes
/// - X: Explicit nudity
/// - XXX: Pornographic content
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
#[serde(rename_all = "PascalCase")]
pub enum NsfwLevel {
    /// Safe for work - no mature content
    None = 1,
    /// Suggestive content
    Soft = 2,
    /// Mature themes, partial nudity
    Mature = 4,
    /// Explicit nudity
    X = 8,
    /// Pornographic content
    #[serde(rename = "XXX")]
    Xxx = 16,
}

impl NsfwLevel {
    /// Get all levels up to and including this one
    /// 
    /// Example: `NsfwLevel::Mature.allowed_levels()` returns `[None, Soft, Mature]`
    pub fn allowed_levels(&self) -> Vec<NsfwLevel> {
        match self {
            NsfwLevel::None => vec![NsfwLevel::None],
            NsfwLevel::Soft => vec![NsfwLevel::None, NsfwLevel::Soft],
            NsfwLevel::Mature => vec![NsfwLevel::None, NsfwLevel::Soft, NsfwLevel::Mature],
            NsfwLevel::X => vec![NsfwLevel::None, NsfwLevel::Soft, NsfwLevel::Mature, NsfwLevel::X],
            NsfwLevel::Xxx => vec![NsfwLevel::None, NsfwLevel::Soft, NsfwLevel::Mature, NsfwLevel::X, NsfwLevel::Xxx],
        }
    }

    /// Get the numeric value for Civitai API
    pub fn as_number(&self) -> u8 {
        *self as u8
    }

    /// Get display label
    pub fn label(&self) -> &'static str {
        match self {
            NsfwLevel::None => "PG (Safe for work)",
            NsfwLevel::Soft => "PG-13 (Suggestive)",
            NsfwLevel::Mature => "R (Mature)",
            NsfwLevel::X => "X (Explicit)",
            NsfwLevel::Xxx => "XXX (Pornographic)",
        }
    }
}

impl Default for NsfwLevel {
    fn default() -> Self {
        NsfwLevel::None
    }
}

/// NSFW filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(target_arch = "wasm32", tsify(into_wasm_abi, from_wasm_abi))]
pub struct NsfwFilter {
    /// Maximum NSFW level to show
    pub max_level: NsfwLevel,
    /// Whether to blur mature content
    pub blur_mature: bool,
}

impl Default for NsfwFilter {
    fn default() -> Self {
        Self {
            max_level: NsfwLevel::None,
            blur_mature: true,
        }
    }
}
