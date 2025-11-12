// TEAM-488: FLUX model module
//
// Self-contained FLUX implementation mirroring stable_diffusion structure
// Based on: reference/candle/candle-transformers/src/models/flux/

mod components;
mod config;
mod loader;
pub mod generation;

pub use components::ModelComponents;
pub use config::FluxConfig;
pub use loader::load_model;

// Re-export generation functions
pub use generation::txt2img;
