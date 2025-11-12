// TEAM-488: ImageModel trait - unified interface for all model types
// TEAM-482: Adopted sealed trait pattern from LLM Worker for API stability
//
// This trait provides a clean abstraction for different image generation models:
// - Stable Diffusion (1.5, 2.1, XL, Turbo)
// - FLUX (dev, schnell)
// - Future models (SD3, Kandinsky, etc.)
//
// Benefits:
// - No conditionals or match statements in generation engine
// - No unsafe casts needed
// - Easy to add new model types
// - Each model is self-contained
// - Sealed trait prevents external implementations (API stability)

use crate::error::Result;
use image::DynamicImage;

/// TEAM-482: Sealed trait pattern - prevents external implementations
///
/// This ensures only our internal model types can implement ImageModel.
/// External crates cannot add their own implementations, which:
/// - Maintains API stability
/// - Prevents breaking changes
/// - Follows Rust API Guidelines
mod sealed {
    pub trait Sealed {}

    // TEAM-482: Only these types can implement ImageModel
    impl Sealed for crate::backend::models::stable_diffusion::StableDiffusionModel {}
    impl Sealed for crate::backend::models::flux::FluxModel {}
    // Add new models here
}

/// Capabilities of an image generation model
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Supports image-to-image transformation
    pub img2img: bool,

    /// Supports inpainting with masks
    pub inpainting: bool,

    /// Supports LoRA weight adaptation
    pub lora: bool,

    /// Supports ControlNet conditioning
    pub controlnet: bool,

    /// Default image size (width, height)
    pub default_size: (usize, usize),

    /// List of supported image sizes
    pub supported_sizes: Vec<(usize, usize)>,

    /// Recommended step count
    pub default_steps: usize,

    /// Supports guidance scale
    pub supports_guidance: bool,
}

/// Unified interface for all image generation models
///
/// TEAM-482: Now sealed - only internal types can implement this trait.
/// This prevents external crates from adding implementations and maintains API stability.
///
/// Each model type implements this trait to provide:
/// - Capability reporting (what operations are supported)
/// - Unified generation interface (same function signature for all models)
/// - Self-contained implementation (all model logic inside the impl)
///
/// The generation engine doesn't need to know which model type it's using.
/// It just calls `generate()` and the model handles everything internally.
pub trait ImageModel: sealed::Sealed + Send + Sync {
    /// Model type identifier (e.g., "stable-diffusion", "flux")
    /// TEAM-482: Returns &'static str for zero-cost abstraction
    /// Note: Implementations should add #[inline] for performance
    fn model_type(&self) -> &'static str;

    /// Model variant (e.g., "v1-5", "flux-dev")
    /// TEAM-482: Returns &'static str for zero-cost abstraction
    /// Note: Implementations should add #[inline] for performance
    fn model_variant(&self) -> &'static str;

    /// Get model capabilities
    /// TEAM-482: Returns reference for zero-cost access
    /// Note: Implementations should add #[inline] for performance
    fn capabilities(&self) -> &ModelCapabilities;

    /// Check if model supports image-to-image
    /// TEAM-482: Default implementation with inline hint for zero-cost
    #[inline]
    fn supports_img2img(&self) -> bool {
        self.capabilities().img2img
    }

    /// Check if model supports inpainting
    /// TEAM-482: Default implementation with inline hint for zero-cost
    #[inline]
    fn supports_inpainting(&self) -> bool {
        self.capabilities().inpainting
    }

    /// Check if model supports LoRA
    /// TEAM-482: Default implementation with inline hint for zero-cost
    #[inline]
    fn supports_lora(&self) -> bool {
        self.capabilities().lora
    }

    /// Check if model supports ControlNet
    /// TEAM-482: Default implementation with inline hint for zero-cost
    #[inline]
    fn supports_controlnet(&self) -> bool {
        self.capabilities().controlnet
    }

    /// Generate image with unified interface
    ///
    /// TEAM-482: NOT inlined - this is already expensive (2-50s runtime).
    /// Inlining would increase binary size with no performance benefit.
    ///
    /// This is the main generation function that all models implement.
    /// The model decides internally how to handle the request based on:
    /// - Whether input_image is present (img2img vs txt2img)
    /// - Whether mask is present (inpainting)
    /// - Model-specific capabilities
    ///
    /// # Arguments
    /// * `request` - Generation request with all parameters
    /// * `progress_callback` - Boxed closure called periodically with (step, total, optional_preview)
    ///
    /// # Returns
    /// Generated image on success
    ///
    /// # Object Safety
    /// TEAM-481: Changed from generic `F` to `Box<dyn FnMut>` for object safety.
    /// This allows using `Box<dyn ImageModel>` instead of enum wrapper.
    /// Performance impact: ~100ns heap allocation vs 2-50s generation = negligible.
    fn generate(
        &mut self,
        request: &GenerationRequest,
        progress_callback: Box<dyn FnMut(usize, usize, Option<DynamicImage>) + Send>,
    ) -> Result<DynamicImage>;
}

/// Unified generation request for all model types
///
/// This replaces the patchwork of different config types.
/// Models extract what they need and ignore the rest.
#[derive(Clone)]
pub struct GenerationRequest {
    /// Request ID for tracking
    pub request_id: String,

    /// Text prompt
    pub prompt: String,

    /// Negative prompt (optional)
    pub negative_prompt: Option<String>,

    /// Image width in pixels
    pub width: usize,

    /// Image height in pixels
    pub height: usize,

    /// Number of denoising steps
    pub steps: usize,

    /// Guidance scale (classifier-free guidance strength)
    pub guidance_scale: f64,

    /// Random seed (optional)
    pub seed: Option<u64>,

    /// Input image for img2img (optional)
    pub input_image: Option<DynamicImage>,

    /// Mask for inpainting (optional)
    pub mask: Option<DynamicImage>,

    /// Strength for img2img (0.0 to 1.0)
    /// Higher = more deviation from input
    pub strength: f64,
}

impl GenerationRequest {
    /// Check if this is an inpainting request
    pub fn is_inpainting(&self) -> bool {
        self.input_image.is_some() && self.mask.is_some()
    }

    /// Check if this is an img2img request
    pub fn is_img2img(&self) -> bool {
        self.input_image.is_some() && self.mask.is_none()
    }

    /// Check if this is a txt2img request
    pub fn is_txt2img(&self) -> bool {
        self.input_image.is_none()
    }

    /// Get operation type as string
    pub fn operation_type(&self) -> &str {
        if self.is_inpainting() {
            "inpaint"
        } else if self.is_img2img() {
            "img2img"
        } else {
            "txt2img"
        }
    }
}
