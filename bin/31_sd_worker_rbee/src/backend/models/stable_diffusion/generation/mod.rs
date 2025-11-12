// TEAM-488: Stable Diffusion generation functions
// Split into logical modules for readability

mod helpers;
mod img2img;
mod inpaint;
mod txt2img;

pub use img2img::img2img;
pub use inpaint::inpaint;
pub use txt2img::txt2img;
// Helper functions are used internally by generation modules
// pub use helpers::{encode_image_to_latents, prepare_inpainting_latents};
