// TEAM-487: Job handlers for SD worker operations
// Each operation gets its own module for better organization

mod types;

pub mod image_generation;
pub mod image_transform;
pub mod image_inpaint;

// Re-export shared types
pub use types::JobResponse;
