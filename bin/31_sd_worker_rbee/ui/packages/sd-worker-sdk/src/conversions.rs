// TEAM-391: Type conversions between Rust and JavaScript
// Pattern: Same as llm-worker-sdk conversions

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

/// Helper functions for converting between Rust and JavaScript types
///
/// These are used internally by the SDK to bridge the gap between
/// Rust structs and JavaScript objects.

/// Convert a JavaScript object to a Rust type
pub fn from_js<T>(value: JsValue) -> Result<T, JsValue>
where
    T: for<'de> Deserialize<'de>,
{
    serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsValue::from_str(&format!("Deserialization error: {}", e)))
}

/// Convert a Rust type to a JavaScript object
pub fn to_js<T>(value: &T) -> Result<JsValue, JsValue>
where
    T: Serialize,
{
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
