//! Operation handlers for rbee-hive
//!
//! TEAM-388: Split job_router.rs into focused modules
//!
//! This module contains handlers for different operation types:
//! - hive: Hive management operations (check, status, etc.)
//! - worker: Worker catalog and process operations
//! - model: Model catalog and provisioning operations

pub mod hive;
pub mod model;
pub mod worker;

// Re-export handlers for convenience
pub use hive::handle_hive_operation;
pub use model::handle_model_operation;
pub use worker::handle_worker_operation;
