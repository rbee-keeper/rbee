// TEAM-391: SD Worker WASM SDK - thin wrapper around job-client
// Pattern: Same as llm-worker-sdk, hive-sdk, and queen-sdk

#![warn(missing_docs)]

//! sd-worker SDK - Rust SDK that compiles to WASM
//!
//! This crate provides JavaScript/TypeScript bindings to the SD worker system
//! by wrapping existing Rust crates (job-client, job-server, operations-contract)
//! and compiling to WASM.
//!
//! # Architecture
//!
//! ```text
//! job-client + job-server (existing) → sd-worker-sdk (thin wrapper) → WASM → JavaScript
//! ```
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { SDWorkerClient } from '@rbee/sd-worker-sdk';
//!
//! await init();
//! const client = new SDWorkerClient('http://localhost:8600', 'sd-worker-1');
//! await client.generateImage({ prompt: 'a photo of a cat', steps: 20 });
//! ```

use wasm_bindgen::prelude::*;

// TEAM-391: Modules (same structure as llm-worker-sdk)
mod client;
mod conversions;

// TEAM-391: Re-export main client
pub use client::SDWorkerClient;

/// Initialize the WASM module
///
/// TEAM-391: This is called automatically when the WASM module is loaded
#[wasm_bindgen(start)]
pub fn init() {
    // TEAM-391: Log to console so we know WASM loaded
    web_sys::console::log_1(&"🎨 [SD Worker SDK] WASM module initialized successfully!".into());
}
