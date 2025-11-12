// TEAM-482: Traits module - interface definitions separated from implementations
//
// Following SD Worker's pattern of separating traits into their own module.
// This provides clear separation between interface and implementation.

mod model_trait;

pub use model_trait::{arch, EosTokens, ModelCapabilities, ModelTrait};
