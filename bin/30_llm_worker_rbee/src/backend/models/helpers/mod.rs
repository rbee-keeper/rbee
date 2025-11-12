// TEAM-482: Helper functions module
//
// Organized helper functions for model loading and detection.
// Extracted from models/mod.rs for better code organization.

pub mod architecture;
pub mod gguf;
pub mod safetensors;

// Re-export commonly used functions for convenience
pub use architecture::{detect_architecture, load_config_json};
pub use gguf::{detect_architecture_from_gguf, extract_eos_token_id, extract_vocab_size, load_gguf_content};
pub use safetensors::{calculate_model_size, create_varbuilder, find_safetensors_files, load_config};
