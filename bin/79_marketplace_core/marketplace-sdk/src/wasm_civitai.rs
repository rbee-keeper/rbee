// TEAM-460: WASM bindings for Civitai API
// TEAM-429: Updated to use CivitaiFilters from artifacts-contract
//! WASM bindings for Civitai client - compiles to JavaScript/TypeScript

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};
use artifacts_contract::CivitaiFilters;

/// List Civitai models (WASM binding)
///
/// # Arguments
/// * `filters` - Filter configuration (CivitaiFilters from artifacts-contract)
///
/// # Returns
/// Promise that resolves to array of Civitai models
///
/// # Example (TypeScript)
/// ```typescript
/// import { list_civitai_models, CivitaiFilters } from './marketplace_sdk'
/// 
/// const filters: CivitaiFilters = {
///   timePeriod: 'Month',
///   modelType: 'Checkpoint',
///   baseModel: 'SDXL 1.0',
///   sort: 'Most Downloaded',
///   nsfw: { maxLevel: 'None', blurMature: true },
///   page: null,
///   limit: 100,
/// }
/// const models = await list_civitai_models(filters)
/// ```
#[wasm_bindgen]
pub async fn list_civitai_models(filters: CivitaiFilters) -> Result<JsValue, JsValue> {
    let url = build_civitai_url(&filters);
    
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
    let filters = CivitaiFilters::default();
    list_civitai_models(filters).await
}

// Helper function to build Civitai API URL from filters
fn build_civitai_url(filters: &CivitaiFilters) -> String {
    let mut url = "https://civitai.com/api/v1/models".to_string();
    let mut params = Vec::new();

    // Limit and page
    params.push(format!("limit={}", filters.limit));
    if let Some(page) = filters.page {
        params.push(format!("page={}", page));
    }

    // Model types
    if filters.model_type != artifacts_contract::CivitaiModelType::All {
        params.push(format!("types={}", filters.model_type.as_str()));
    } else {
        // Default: Checkpoint and LORA
        params.push("types=Checkpoint".to_string());
        params.push("types=LORA".to_string());
    }

    // Sort
    params.push(format!("sort={}", filters.sort.as_str()));

    // Time period
    if filters.time_period != artifacts_contract::TimePeriod::AllTime {
        params.push(format!("period={}", filters.time_period.as_str()));
    }

    // Base model
    if filters.base_model != artifacts_contract::BaseModel::All {
        params.push(format!("baseModel={}", filters.base_model.as_str()));
    }

    // NSFW filtering
    let nsfw_levels = filters.nsfw.max_level.allowed_levels();
    for level in nsfw_levels {
        params.push(format!("nsfwLevel={}", level.as_number()));
    }

    if !params.is_empty() {
        url.push('?');
        url.push_str(&params.join("&"));
    }

    url
}

// Helper function to fetch JSON from URL using browser fetch API
async fn fetch_json(url: &str) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

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
