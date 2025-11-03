// TEAM-XXX: Stable Diffusion worker library
//
// This crate provides a Candle-based Stable Diffusion inference worker
// with support for CPU, CUDA, and Metal backends.

#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

pub mod backend;
pub mod device;
pub mod error;
pub mod http;
pub mod job_router;
pub mod narration;

pub use error::{Error, Result};
