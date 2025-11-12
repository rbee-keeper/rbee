// TEAM-482: Shared preview generation logic
//
// Consolidates preview generation across SD and FLUX to eliminate duplication

use crate::error::Result;
use image::DynamicImage;

/// Preview generation frequency (every N steps)
const PREVIEW_FREQUENCY: usize = 5;

/// Check if preview should be generated at this step
///
/// # Arguments
/// * `step_idx` - Current step index (0-based)
/// * `num_steps` - Total number of steps
///
/// # Returns
/// `true` if preview should be generated
#[inline(always)]
pub fn should_generate_preview(step_idx: usize, num_steps: usize) -> bool {
    step_idx.is_multiple_of(PREVIEW_FREQUENCY) || step_idx == num_steps - 1
}

/// Handle preview generation with error handling
///
/// Wraps preview generation with proper error handling and progress callback.
/// Used by txt2img, img2img, inpaint, and FLUX generation.
///
/// # Arguments
/// * `step_idx` - Current step index (0-based)
/// * `num_steps` - Total number of steps
/// * `preview_fn` - Function that generates the preview (VAE decode + conversion)
/// * `progress_callback` - Callback for progress updates
///
/// # Performance
/// Called 4-6 times per generation (every 5 steps + final)
/// Each call: ~85ms (80ms VAE decode + 5ms conversion)
///
/// # Note
/// The caller must provide the preview_fn that does VAE decode,
/// since VAE types differ between SD and FLUX.
#[inline]
pub fn handle_preview<F, P>(
    step_idx: usize,
    num_steps: usize,
    preview_fn: P,
    mut progress_callback: F,
) where
    F: FnMut(usize, usize, Option<DynamicImage>),
    P: FnOnce() -> Result<DynamicImage>,
{
    if should_generate_preview(step_idx, num_steps) {
        match preview_fn() {
            Ok(preview) => progress_callback(step_idx + 1, num_steps, Some(preview)),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to generate preview image");
                progress_callback(step_idx + 1, num_steps, None);
            }
        }
    } else {
        progress_callback(step_idx + 1, num_steps, None);
    }
}
