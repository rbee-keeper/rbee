//! # Error Types
//!
//! TEAM-390: Error types for SD worker with comprehensive error handling.
//!
//! This module provides a unified error type for all SD worker operations,
//! including model loading, inference, image processing, and HTTP handling.

use thiserror::Error;

/// Result type alias for SD worker operations
///
/// Uses [`Error`] as the error type for all fallible operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for SD worker operations
///
/// TEAM-390: Covers all error cases in the SD worker:
/// - Candle framework errors (model operations, tensor ops)
/// - I/O errors (file operations, network)
/// - Image processing errors (encoding/decoding)
/// - Model loading errors (`HuggingFace` Hub, safetensors)
/// - Generation errors (sampling, scheduling)
/// - HTTP errors (server, client)
#[derive(Error, Debug)]
pub enum Error {
    /// Candle framework error (tensor operations, model inference)
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// I/O error (file operations, network)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Image processing error (encoding/decoding, format conversion)
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    /// Tokenizer error (text encoding, vocabulary issues)
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Model loading error (`HuggingFace` Hub, safetensors, config parsing)
    #[error("Model loading error: {0}")]
    ModelLoading(String),

    /// Generation error (sampling, scheduling, inference pipeline)
    #[error("Generation error: {0}")]
    Generation(String),

    /// Invalid input error (validation failures, out-of-range values)
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Device error (CUDA/Metal initialization, memory allocation)
    #[error("Device error: {0}")]
    Device(String),

    /// HTTP error (server errors, client errors, network issues)
    #[error("HTTP error: {0}")]
    Http(String),

    /// Other error (catch-all for unexpected errors)
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}
