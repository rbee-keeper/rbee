//! Environment variable configuration helpers
//!
//! TEAM-XXX: Centralized environment variable loading
//!
//! Provides type-safe access to environment variables with sensible defaults.
//! All hardcoded ports and URLs should use this module instead.

use std::env;

/// Get environment variable or return default value
fn get_env_or_default(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Get environment variable as u16 port number
fn get_port_or_default(key: &str, default: u16) -> u16 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// ============================================================
// Backend Service Configuration
// ============================================================

/// Get Queen-rbee port (default: 7833)
pub fn queen_port() -> u16 {
    get_port_or_default("QUEEN_PORT", 7833)
}

/// Get Queen-rbee URL (default: http://localhost:7833)
pub fn queen_url() -> String {
    get_env_or_default("QUEEN_URL", "http://localhost:7833")
}

/// Get Rbee-hive port (default: 7835)
pub fn hive_port() -> u16 {
    get_port_or_default("HIVE_PORT", 7835)
}

/// Get Rbee-hive URL (default: http://localhost:7835)
pub fn hive_url() -> String {
    get_env_or_default("HIVE_URL", "http://localhost:7835")
}

/// Get LLM worker port (default: 8080)
pub fn llm_worker_port() -> u16 {
    get_port_or_default("LLM_WORKER_PORT", 8080)
}

/// Get LLM worker URL (default: http://localhost:8080)
pub fn llm_worker_url() -> String {
    get_env_or_default("LLM_WORKER_URL", "http://localhost:8080")
}

/// Get SD worker port (default: 8081)
pub fn sd_worker_port() -> u16 {
    get_port_or_default("SD_WORKER_PORT", 8081)
}

/// Get SD worker URL (default: http://localhost:8081)
pub fn sd_worker_url() -> String {
    get_env_or_default("SD_WORKER_URL", "http://localhost:8081")
}

/// Get ComfyUI worker port (default: 8188)
pub fn comfy_worker_port() -> u16 {
    get_port_or_default("COMFY_WORKER_PORT", 8188)
}

/// Get ComfyUI worker URL (default: http://localhost:8188)
pub fn comfy_worker_url() -> String {
    get_env_or_default("COMFY_WORKER_URL", "http://localhost:8188")
}

/// Get vLLM worker port (default: 8000)
pub fn vllm_worker_port() -> u16 {
    get_port_or_default("VLLM_WORKER_PORT", 8000)
}

/// Get vLLM worker URL (default: http://localhost:8000)
pub fn vllm_worker_url() -> String {
    get_env_or_default("VLLM_WORKER_URL", "http://localhost:8000")
}

/// Get worker catalog port (default: 8787)
pub fn worker_catalog_port() -> u16 {
    get_port_or_default("WORKER_CATALOG_PORT", 8787)
}

/// Get worker catalog URL (default: http://localhost:8787)
pub fn worker_catalog_url() -> String {
    get_env_or_default("WORKER_CATALOG_URL", "http://localhost:8787")
}

// ============================================================
// Frontend Development Configuration
// ============================================================

/// Get Keeper dev port (default: 5173)
pub fn keeper_dev_port() -> u16 {
    get_port_or_default("KEEPER_DEV_PORT", 5173)
}

/// Get Queen UI dev port (default: 7834)
pub fn queen_ui_dev_port() -> u16 {
    get_port_or_default("QUEEN_UI_DEV_PORT", 7834)
}

/// Get Hive UI dev port (default: 7836)
pub fn hive_ui_dev_port() -> u16 {
    get_port_or_default("HIVE_UI_DEV_PORT", 7836)
}

/// Get LLM Worker UI dev port (default: 7837)
pub fn llm_worker_ui_dev_port() -> u16 {
    get_port_or_default("LLM_WORKER_UI_DEV_PORT", 7837)
}

// ============================================================
// External Service Configuration
// ============================================================

/// Get Hugging Face base URL (default: https://huggingface.co)
pub fn huggingface_base_url() -> String {
    get_env_or_default("HUGGINGFACE_BASE_URL", "https://huggingface.co")
}

/// Get GitHub releases base URL (default: https://github.com)
pub fn github_releases_base_url() -> String {
    get_env_or_default("GITHUB_RELEASES_BASE_URL", "https://github.com")
}

// ============================================================
// Hive Configuration
// ============================================================

/// Get Hive ID (default: localhost)
pub fn hive_id() -> String {
    get_env_or_default("HIVE_ID", "localhost")
}

/// Get Queen URL for hive heartbeat (default: http://localhost:7833)
pub fn hive_queen_url() -> String {
    get_env_or_default("HIVE_QUEEN_URL", "http://localhost:7833")
}

// ============================================================
// Development Settings
// ============================================================

/// Check if development mode is enabled (default: true)
pub fn is_dev_mode() -> bool {
    env::var("DEV_MODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true)
}

/// Check if HTTPS should be used (default: false)
pub fn use_https() -> bool {
    env::var("USE_HTTPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_ports() {
        // Test that defaults work when env vars are not set
        assert_eq!(queen_port(), 7833);
        assert_eq!(hive_port(), 7835);
        assert_eq!(llm_worker_port(), 8080);
        assert_eq!(worker_catalog_port(), 8787);
    }

    #[test]
    fn test_default_urls() {
        assert_eq!(queen_url(), "http://localhost:7833");
        assert_eq!(hive_url(), "http://localhost:7835");
        assert_eq!(worker_catalog_url(), "http://localhost:8787");
    }

    #[test]
    fn test_external_urls() {
        assert_eq!(huggingface_base_url(), "https://huggingface.co");
        assert_eq!(github_releases_base_url(), "https://github.com");
    }

    #[test]
    fn test_dev_mode_defaults() {
        // Without env var set, should default to true
        assert!(is_dev_mode());
        assert!(!use_https());
    }
}
