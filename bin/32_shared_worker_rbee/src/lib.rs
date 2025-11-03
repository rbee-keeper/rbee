// TEAM-XXX: Shared worker utilities
//
// This crate provides common infrastructure for all worker types (LLM, SD, etc.)

#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

pub mod device;
pub mod heartbeat;

pub use device::{init_cpu_device, verify_device};

#[cfg(feature = "cuda")]
pub use device::init_cuda_device;

#[cfg(feature = "metal")]
pub use device::init_metal_device;

pub use heartbeat::{send_heartbeat_to_queen, start_heartbeat_task};
