// TEAM-460: WASM bindings for HuggingFace API
//! WASM bindings for HuggingFace client - compiles to JavaScript/TypeScript

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use tsify::Tsify;

/// WASM-compatible HuggingFace model response
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct HuggingFaceModel {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model author
    pub author: Option<String>,
    /// Download count
    pub downloads: f64,
    /// Like count
    pub likes: f64,
    /// Model tags
    pub tags: Vec<String>,
    /// Last modified timestamp
    #[serde(rename = "lastModified")]
    pub last_modified: Option<String>,
}

/// List HuggingFace models (WASM binding)
///
/// # Arguments
/// * `limit` - Maximum number of models to return
/// * `sort` - Sort order
///
/// # Returns
/// Promise that resolves to array of HuggingFace models
#[wasm_bindgen]
pub async fn list_huggingface_models(
    limit: Option<i32>,
    sort: Option<String>,
) -> Result<JsValue, JsValue> {
    let url = build_huggingface_url(limit, sort.as_deref());

    let response = fetch_json(&url)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to fetch HuggingFace models: {:?}", e)))?;

    Ok(response)
}

/// Get a specific HuggingFace model by ID (WASM binding)
///
/// # Arguments
/// * `model_id` - HuggingFace model ID
///
/// # Returns
/// Promise that resolves to HuggingFace model
#[wasm_bindgen]
pub async fn get_huggingface_model(model_id: String) -> Result<JsValue, JsValue> {
    let url = format!("https://huggingface.co/api/models/{}", model_id);

    fetch_json(&url)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to fetch HuggingFace model: {:?}", e)))
}

/// Get compatible HuggingFace models for rbee (WASM binding)
///
/// Returns top 100 LLM models
///
/// # Returns
/// Promise that resolves to array of compatible HuggingFace models
#[wasm_bindgen]
pub async fn get_compatible_huggingface_models() -> Result<JsValue, JsValue> {
    list_huggingface_models(
        Some(100),
        Some("downloads".to_string()),
    )
    .await
}

// Helper function to build HuggingFace API URL
fn build_huggingface_url(
    limit: Option<i32>,
    sort: Option<&str>,
) -> String {
    let mut url = "https://huggingface.co/api/models".to_string();
    let mut params = Vec::new();

    if let Some(limit) = limit {
        params.push(format!("limit={}", limit));
    }
    if let Some(sort) = sort {
        params.push(format!("sort={}", sort));
    }
    params.push("filter=text-generation".to_string());

    if !params.is_empty() {
        url.push('?');
        url.push_str(&params.join("&"));
    }

    url
}

// Helper function to fetch JSON from URL using browser fetch API
async fn fetch_json(url: &str) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)?;
    request.headers().set("Content-Type", "application/json")?;

    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP error: {}",
            resp.status()
        )));
    }

    let json = JsFuture::from(resp.json()?).await?;
    Ok(json)
}
