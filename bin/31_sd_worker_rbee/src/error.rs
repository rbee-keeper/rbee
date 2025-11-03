// TEAM-XXX: Error types for SD worker

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}
