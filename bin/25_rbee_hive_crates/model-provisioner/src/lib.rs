//! Model Provisioner - Downloads models from HuggingFace
//!
//! Uses the official `hf-hub` Rust crate (same library used by Candle)
//! to download GGUF models from HuggingFace Hub.
//!
//! # Architecture
//!
//! ```text
//! ModelProvisioner
//!     ↓
//! HuggingFaceVendor (implements VendorSource)
//!     ↓
//! hf-hub crate (official HuggingFace Rust client)
//!     ↓
//! Model provisioning for rbee-hive
//!
//! Downloads models from various sources (HuggingFace, GitHub, local builds).
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod download_tracker;
mod huggingface;
mod provisioner;

pub use download_tracker::{DownloadProgress, DownloadTracker};
pub use huggingface::HuggingFaceVendor;
pub use provisioner::ModelProvisioner;
