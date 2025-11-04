// TEAM-402: Marketplace SDK main entry point

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use wasm_bindgen::prelude::*;

// Modules
mod types;

// Re-export types
pub use types::*;

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    web_sys::console::log_1(&"🛒 [Marketplace SDK] WASM initialized!".into());
}
