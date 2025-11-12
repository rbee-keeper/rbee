// TEAM-488: FLUX model loading
//
// Loads FLUX.1-dev and FLUX.1-schnell models from HuggingFace Hub.
// Based on: reference/candle/candle-examples/examples/flux/main.rs

use crate::error::{Error, Result};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, flux, t5};
use std::path::Path;
use tokenizers::Tokenizer;

use super::components::{ModelComponents, SendFluxModel};
use super::super::SDVersion;

/// Load FLUX model from directory
///
/// # Arguments
/// * `model_path` - Path to model directory (from HuggingFace cache)
/// * `version` - FLUX variant (FluxDev or FluxSchnell)
/// * `device` - Device to load on (CPU/CUDA)
/// * `use_f16` - Use F16 precision (recommended for GPU)
/// * `quantized` - Use quantized GGUF model (memory efficient)
///
/// # Returns
/// Loaded FLUX components ready for generation
///
/// # Errors
/// Returns error if model files are missing or loading fails
pub fn load_model(
    model_path: &str,
    version: SDVersion,
    device: &Device,
    use_f16: bool,
    quantized: bool,
) -> Result<ModelComponents> {
    if !version.is_flux() {
        return Err(Error::InvalidInput(format!(
            "Expected FLUX model, got {:?}",
            version
        )));
    }

    let dtype = if use_f16 { DType::F16 } else { DType::F32 };
    let model_path = Path::new(model_path);
    
    tracing::info!("Loading FLUX {:?} from {:?}", version, model_path);
    tracing::info!("Using dtype: {:?}, quantized: {}", dtype, quantized);
    
    // 1. Load T5-XXL text encoder
    tracing::info!("Loading T5-XXL tokenizer and model...");
    let t5_tokenizer = {
        let tokenizer_path = model_path.join("tokenizer_2/tokenizer.json");
        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::ModelLoading(format!("Failed to load T5 tokenizer: {}", e)))?
    };
    
    let t5_model = {
        // T5-v1.1-XXL config
        // TEAM-488: Manually create config since v1_1_xxl() doesn't exist in this Candle version
        let mut config = t5::Config::musicgen_small();
        // Override with XXL dimensions
        config.vocab_size = 32128;
        config.d_model = 4096;
        config.d_kv = 64;
        config.d_ff = 10240;
        config.num_layers = 24;
        config.num_heads = 64;
        
        let weights_path = model_path.join("text_encoder_2/model.safetensors");
        // SAFETY: Memory-mapped file access is safe for read-only model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .map_err(|e| Error::ModelLoading(format!("Failed to load T5 weights: {}", e)))?
        };
        
        t5::T5EncoderModel::load(vb, &config)
            .map_err(|e| Error::ModelLoading(format!("Failed to build T5 model: {}", e)))?
    };
    
    // 2. Load CLIP text encoder
    tracing::info!("Loading CLIP tokenizer and model...");
    let clip_tokenizer = {
        let tokenizer_path = model_path.join("tokenizer/tokenizer.json");
        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::ModelLoading(format!("Failed to load CLIP tokenizer: {}", e)))?
    };
    
    let clip_model = {
        // CLIP ViT-L/14 config (from openai/clip-vit-large-patch14)
        let config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            projection_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            intermediate_size: 3072,
            embed_dim: 768,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        };
        
        let weights_path = model_path.join("text_encoder/model.safetensors");
        // SAFETY: Memory-mapped file access is safe for read-only model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .map_err(|e| Error::ModelLoading(format!("Failed to load CLIP weights: {}", e)))?
        };
        
        clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config)
            .map_err(|e| Error::ModelLoading(format!("Failed to build CLIP model: {}", e)))?
    };
    
    // 3. Load FLUX transformer
    tracing::info!("Loading FLUX transformer (quantized: {})...", quantized);
    let flux_model: Box<dyn flux::WithForward> = if quantized {
        // Load quantized GGUF model (memory efficient)
        let gguf_filename = match version {
            SDVersion::FluxSchnell => "flux1-schnell.gguf",
            SDVersion::FluxDev => "flux1-dev.gguf",
            _ => unreachable!(),
        };
        let gguf_path = model_path.join(gguf_filename);
        
        if !gguf_path.exists() {
            return Err(Error::ModelLoading(format!(
                "Quantized GGUF file not found: {}. Use non-quantized mode or download GGUF.",
                gguf_path.display()
            )));
        }
        
        let config = match version {
            SDVersion::FluxDev => flux::model::Config::dev(),
            SDVersion::FluxSchnell => flux::model::Config::schnell(),
            _ => unreachable!(),
        };
        
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            gguf_path,
            device,
        )
        .map_err(|e| Error::ModelLoading(format!("Failed to load GGUF: {}", e)))?;
        
        let model = flux::quantized_model::Flux::new(&config, vb)
            .map_err(|e| Error::ModelLoading(format!("Failed to build quantized FLUX: {}", e)))?;
        
        Box::new(model)
    } else {
        // Load full precision model
        let config = match version {
            SDVersion::FluxDev => flux::model::Config::dev(),
            SDVersion::FluxSchnell => flux::model::Config::schnell(),
            _ => unreachable!(),
        };
        
        let weights_filename = match version {
            SDVersion::FluxDev => "flux1-dev.safetensors",
            SDVersion::FluxSchnell => "flux1-schnell.safetensors",
            _ => unreachable!(),
        };
        let weights_path = model_path.join(weights_filename);
        
        // SAFETY: Memory-mapped file access is safe for read-only model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .map_err(|e| Error::ModelLoading(format!("Failed to load FLUX weights: {}", e)))?
        };
        
        let model = flux::model::Flux::new(&config, vb)
            .map_err(|e| Error::ModelLoading(format!("Failed to build FLUX model: {}", e)))?;
        
        Box::new(model)
    };
    
    // 4. Load VAE (autoencoder)
    tracing::info!("Loading FLUX VAE...");
    let vae = {
        let config = match version {
            SDVersion::FluxDev => flux::autoencoder::Config::dev(),
            SDVersion::FluxSchnell => flux::autoencoder::Config::schnell(),
            _ => unreachable!(),
        };
        
        let weights_path = model_path.join("ae.safetensors");
        // SAFETY: Memory-mapped file access is safe for read-only model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .map_err(|e| Error::ModelLoading(format!("Failed to load VAE weights: {}", e)))?
        };
        
        flux::autoencoder::AutoEncoder::new(&config, vb)
            .map_err(|e| Error::ModelLoading(format!("Failed to build VAE: {}", e)))?
    };
    
    tracing::info!("FLUX model loaded successfully");
    
    Ok(ModelComponents {
        version,
        device: device.clone(),
        dtype,
        t5_tokenizer,
        t5_model,
        clip_tokenizer,
        clip_model,
        flux_model: SendFluxModel(flux_model),
        vae,
    })
}
