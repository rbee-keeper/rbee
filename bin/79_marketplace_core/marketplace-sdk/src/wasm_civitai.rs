// TEAM-460: WASM bindings for Civitai API
//! WASM bindings for Civitai client - compiles to JavaScript/TypeScript

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use tsify::Tsify;

/// WASM-compatible Civitai model response
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CivitaiModel {
    /// Model ID
    pub id: i64,
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model type (Checkpoint, LORA, etc.)
    #[serde(rename = "type")]
    pub model_type: String,
    /// Whether model is NSFW
    pub nsfw: bool,
    /// Commercial use permission
    #[serde(rename = "allowCommercialUse")]
    pub allow_commercial_use: String,
    /// Model statistics
    pub stats: CivitaiStats,
    /// Model creator
    pub creator: CivitaiCreator,
    /// Model tags
    pub tags: Vec<String>,
    /// Model versions
    #[serde(rename = "modelVersions")]
    pub model_versions: Vec<CivitaiModelVersion>,
}

/// WASM-compatible Civitai statistics
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CivitaiStats {
    /// Download count
    #[serde(rename = "downloadCount")]
    pub download_count: i64,
    /// Favorite count
    #[serde(rename = "favoriteCount")]
    pub favorite_count: i64,
    /// Comment count
    #[serde(rename = "commentCount")]
    pub comment_count: i64,
    /// Rating count
    #[serde(rename = "ratingCount")]
    pub rating_count: i64,
    /// Average rating
    pub rating: f64,
}

/// WASM-compatible Civitai creator
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CivitaiCreator {
    /// Creator username
    pub username: String,
    /// Creator avatar image URL
    pub image: Option<String>,
}

/// WASM-compatible Civitai model version
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CivitaiModelVersion {
    /// Version ID
    pub id: i64,
    /// Parent model ID
    #[serde(rename = "modelId")]
    pub model_id: i64,
    /// Version name
    pub name: String,
    /// Base model (SDXL, SD 1.5, etc.)
    #[serde(rename = "baseModel")]
    pub base_model: String,
    /// Trained trigger words
    #[serde(rename = "trainedWords")]
    pub trained_words: Vec<String>,
    /// Model files
    pub files: Vec<CivitaiFile>,
    /// Preview images
    pub images: Vec<CivitaiImage>,
}

/// WASM-compatible Civitai file
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CivitaiFile {
    /// File name
    pub name: String,
    /// File ID
    pub id: i64,
    /// File size in KB
    #[serde(rename = "sizeKB")]
    pub size_kb: f64,
    /// Download URL
    #[serde(rename = "downloadUrl")]
    pub download_url: String,
    /// Whether this is the primary file
    pub primary: bool,
}

/// WASM-compatible Civitai image
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CivitaiImage {
    /// Image URL
    pub url: String,
    /// Whether image is NSFW
    pub nsfw: bool,
    /// Image width
    pub width: i32,
    /// Image height
    pub height: i32,
}

/// List Civitai models (WASM binding)
///
/// # Arguments
/// * `limit` - Maximum number of models to return
/// * `types` - Filter by model types (e.g., "Checkpoint,LORA")
/// * `sort` - Sort order ("Highest Rated", "Most Downloaded", "Newest")
/// * `nsfw` - Include NSFW models
///
/// # Returns
/// Promise that resolves to array of Civitai models
#[wasm_bindgen]
pub async fn list_civitai_models(
    limit: Option<i32>,
    types: Option<String>,
    sort: Option<String>,
    nsfw: Option<bool>,
) -> Result<JsValue, JsValue> {
    let url = build_civitai_url(limit, types.as_deref(), sort.as_deref(), nsfw);

    let response = fetch_json(&url)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to fetch Civitai models: {:?}", e)))?;

    // Extract items array from response
    let items = js_sys::Reflect::get(&response, &JsValue::from_str("items"))
        .map_err(|_| JsValue::from_str("Response missing 'items' field"))?;

    Ok(items)
}

/// Get a specific Civitai model by ID (WASM binding)
///
/// # Arguments
/// * `model_id` - Civitai model ID
///
/// # Returns
/// Promise that resolves to Civitai model
#[wasm_bindgen]
pub async fn get_civitai_model(model_id: i64) -> Result<JsValue, JsValue> {
    let url = format!("https://civitai.com/api/v1/models/{}", model_id);

    fetch_json(&url)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to fetch Civitai model: {:?}", e)))
}

/// Get compatible Civitai models for rbee (WASM binding)
///
/// Returns top 100 safe, commercial-use Stable Diffusion models
///
/// # Returns
/// Promise that resolves to array of compatible Civitai models
#[wasm_bindgen]
pub async fn get_compatible_civitai_models() -> Result<JsValue, JsValue> {
    list_civitai_models(
        Some(100),
        Some("Checkpoint,LORA".to_string()),
        Some("Most Downloaded".to_string()),
        Some(false),
    )
    .await
}

// Helper function to build Civitai API URL
fn build_civitai_url(
    limit: Option<i32>,
    types: Option<&str>,
    sort: Option<&str>,
    nsfw: Option<bool>,
) -> String {
    let mut url = "https://civitai.com/api/v1/models".to_string();
    let mut params = Vec::new();

    if let Some(limit) = limit {
        params.push(format!("limit={}", limit));
    }
    if let Some(types) = types {
        params.push(format!("types={}", types));
    }
    if let Some(sort) = sort {
        params.push(format!("sort={}", sort));
    }
    if let Some(nsfw) = nsfw {
        params.push(format!("nsfw={}", nsfw));
    }
    params.push("allowCommercialUse=Sell".to_string());
    params.push("primaryFileOnly=true".to_string());

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
