// TEAM-412: Protocol handler for rbee:// URLs
// Handles one-click model installation from marketplace

use tauri::{AppHandle, Emitter};
use serde::{Deserialize, Serialize};

/// Protocol action types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProtocolAction {
    /// Install a model: rbee://install/model?id=meta-llama/Llama-3.2-1B
    Install,
    /// Open marketplace: rbee://marketplace
    Marketplace,
    /// Open model details: rbee://model?id=meta-llama/Llama-3.2-1B
    Model,
}

/// Parsed protocol URL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolUrl {
    pub action: ProtocolAction,
    pub model_id: Option<String>,
    pub worker_type: Option<String>,
}

/// Parse rbee:// protocol URL
///
/// Examples:
/// - rbee://install/model?id=meta-llama/Llama-3.2-1B&worker=cpu
/// - rbee://marketplace
/// - rbee://model?id=meta-llama/Llama-3.2-1B
pub fn parse_protocol_url(url: &str) -> Result<ProtocolUrl, String> {
    // Remove rbee:// prefix
    let url = url.strip_prefix("rbee://")
        .ok_or_else(|| "Invalid protocol URL: missing rbee:// prefix".to_string())?;
    
    // Split into path and query
    let (path, query) = if let Some(pos) = url.find('?') {
        (&url[..pos], Some(&url[pos + 1..]))
    } else {
        (url, None)
    };
    
    // Parse action from path
    let action = match path {
        "marketplace" => ProtocolAction::Marketplace,
        "model" => ProtocolAction::Model,
        path if path.starts_with("install/model") => ProtocolAction::Install,
        _ => return Err(format!("Unknown protocol action: {}", path)),
    };
    
    // Parse query parameters
    let mut model_id = None;
    let mut worker_type = None;
    
    if let Some(query) = query {
        for param in query.split('&') {
            if let Some((key, value)) = param.split_once('=') {
                match key {
                    // TEAM-412: No URL decoding needed - Tauri handles it
                    "id" => model_id = Some(value.to_string()),
                    "worker" => worker_type = Some(value.to_string()),
                    _ => {} // Ignore unknown parameters
                }
            }
        }
    }
    
    Ok(ProtocolUrl {
        action,
        model_id,
        worker_type,
    })
}

/// Handle protocol URL
pub async fn handle_protocol_url(app: &AppHandle, url: &str) -> Result<(), String> {
    let parsed = parse_protocol_url(url)?;
    
    match parsed.action {
        ProtocolAction::Marketplace => {
            // TEAM-412: Navigate to marketplace page (Tauri v2 uses emit() not emit_all())
            app.emit("navigate", "/marketplace/llm-models")
                .map_err(|e| format!("Failed to navigate: {}", e))?;
        }
        
        ProtocolAction::Model => {
            if let Some(model_id) = parsed.model_id {
                // TEAM-412: Navigate to model details page
                app.emit("navigate", format!("/marketplace/llm-models/{}", model_id))
                    .map_err(|e| format!("Failed to navigate: {}", e))?;
            } else {
                return Err("Missing model ID".to_string());
            }
        }
        
        ProtocolAction::Install => {
            if let Some(model_id) = parsed.model_id {
                // TEAM-412: Show install dialog
                app.emit("install-model", serde_json::json!({
                    "modelId": model_id,
                    "workerType": parsed.worker_type,
                }))
                .map_err(|e| format!("Failed to trigger install: {}", e))?;
            } else {
                return Err("Missing model ID".to_string());
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_marketplace_url() {
        let url = "rbee://marketplace";
        let parsed = parse_protocol_url(url).unwrap();
        assert!(matches!(parsed.action, ProtocolAction::Marketplace));
        assert!(parsed.model_id.is_none());
    }
    
    #[test]
    fn test_parse_model_url() {
        let url = "rbee://model?id=meta-llama/Llama-3.2-1B";
        let parsed = parse_protocol_url(url).unwrap();
        assert!(matches!(parsed.action, ProtocolAction::Model));
        assert_eq!(parsed.model_id, Some("meta-llama/Llama-3.2-1B".to_string()));
    }
    
    #[test]
    fn test_parse_install_url() {
        let url = "rbee://install/model?id=meta-llama/Llama-3.2-1B&worker=cpu";
        let parsed = parse_protocol_url(url).unwrap();
        assert!(matches!(parsed.action, ProtocolAction::Install));
        assert_eq!(parsed.model_id, Some("meta-llama/Llama-3.2-1B".to_string()));
        assert_eq!(parsed.worker_type, Some("cpu".to_string()));
    }
    
    #[test]
    fn test_parse_invalid_url() {
        let url = "rbee://unknown";
        assert!(parse_protocol_url(url).is_err());
    }
}
