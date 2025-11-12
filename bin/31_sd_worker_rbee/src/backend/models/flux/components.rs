// TEAM-488: FLUX model components
// Direct Candle types - RULE ZERO compliant

use candle_core::{Device, DType};
use candle_transformers::models::{clip, flux, t5};
use tokenizers::Tokenizer;

use super::super::SDVersion;

/// Wrapper to make FLUX model Send+Sync (safe because we control threading)
/// TEAM-488: FLUX models are used sequentially by the generation engine
/// The generation queue ensures only one generation happens at a time
pub(super) struct SendFluxModel(pub Box<dyn flux::WithForward>);

// SAFETY: We guarantee single-threaded access via the generation queue
// The generation engine processes requests sequentially, never concurrently
unsafe impl Send for SendFluxModel {}
unsafe impl Sync for SendFluxModel {}

/// FLUX model components loaded into memory
/// TEAM-488: Direct Candle types, NO wrappers (RULE ZERO)
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub dtype: DType,
    
    // Text encoders
    pub t5_tokenizer: Tokenizer,
    pub t5_model: t5::T5EncoderModel,
    pub clip_tokenizer: Tokenizer,
    pub clip_model: clip::text_model::ClipTextTransformer,
    
    // FLUX transformer (trait object for full/quantized)
    // TEAM-488: Wrapped in SendFluxModel to enable spawn_blocking
    pub(super) flux_model: SendFluxModel,
    
    // VAE
    pub vae: flux::autoencoder::AutoEncoder,
}

impl ModelComponents {
    /// Get reference to FLUX model for generation
    pub fn flux_model(&self) -> &dyn flux::WithForward {
        &*self.flux_model.0
    }
    
    /// Get mutable reference to FLUX model for generation
    pub fn flux_model_mut(&mut self) -> &mut dyn flux::WithForward {
        &mut *self.flux_model.0
    }
}
