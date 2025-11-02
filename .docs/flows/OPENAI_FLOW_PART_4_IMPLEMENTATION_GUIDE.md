# OpenAI Flow Part 4: Complete Implementation Guide

**Complete step-by-step implementation guide for OpenAI adapter**  
**Date:** November 2, 2025  
**Status:** ⚠️ IMPLEMENTATION GUIDE

---

## Overview

This document provides a complete, step-by-step guide to implementing the OpenAI adapter, from initial setup to production deployment.

---

## Phase 1: Basic Translation (M2)

### Step 1: Create Translation Module

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/translation.rs` (NEW)

```rust
//! Request/response translation between OpenAI and rbee formats

use crate::error::OpenAIError;
use crate::types::{ChatCompletionRequest, ChatMessage};
use operations_contract::{Operation, InferRequest};

/// Extract prompt from OpenAI messages
pub fn extract_prompt(messages: Vec<ChatMessage>) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Map OpenAI model names to rbee model IDs
pub fn map_model_name(openai_model: &str) -> Result<String, OpenAIError> {
    match openai_model {
        "gpt-3.5-turbo" | "gpt-3.5-turbo-16k" => Ok("tinyllama".to_string()),
        "gpt-4" | "gpt-4-32k" => Ok("llama-7b".to_string()),
        _ => Err(OpenAIError::ModelNotFound(
            format!("Model '{}' not found", openai_model)
        ))
    }
}

/// Convert OpenAI request to rbee Operation
pub fn to_rbee_operation(
    request: ChatCompletionRequest,
    hive_id: String,
) -> Result<Operation, OpenAIError> {
    let prompt = extract_prompt(request.messages);
    let model = map_model_name(&request.model)?;
    
    Ok(Operation::Infer(InferRequest {
        hive_id,
        model,
        prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: None,
        device: None,
        worker_id: None,
        stream: request.stream,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_prompt() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }
        ];
        assert_eq!(extract_prompt(messages), "user: Hello");
    }
    
    #[test]
    fn test_map_model_name() {
        assert_eq!(map_model_name("gpt-3.5-turbo").unwrap(), "tinyllama");
        assert!(map_model_name("unknown").is_err());
    }
}
```

---

### Step 2: Create Adapter State

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/lib.rs`

**Add to existing file:**
```rust
use std::collections::HashMap;
use std::sync::Arc;

/// State for OpenAI adapter
#[derive(Clone)]
pub struct AdapterState {
    pub queen_url: String,
    pub default_hive_id: String,
    pub model_mappings: Arc<HashMap<String, String>>,
}

impl AdapterState {
    pub fn new(queen_url: String, default_hive_id: String) -> Self {
        let mut mappings = HashMap::new();
        mappings.insert("gpt-3.5-turbo".to_string(), "tinyllama".to_string());
        mappings.insert("gpt-4".to_string(), "llama-7b".to_string());
        
        Self {
            queen_url,
            default_hive_id,
            model_mappings: Arc::new(mappings),
        }
    }
}
```

---

### Step 3: Update Chat Completions Handler

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

**Replace stub with:**
```rust
use crate::translation::{to_rbee_operation};
use crate::types::*;
use crate::error::OpenAIError;
use crate::AdapterState;
use axum::{extract::State, response::IntoResponse, Json};
use jobs_contract::JobResponse;

pub async fn chat_completions(
    State(state): State<AdapterState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, OpenAIError> {
    // Validate request
    if request.messages.is_empty() {
        return Err(OpenAIError::InvalidRequest(
            "Messages array cannot be empty".to_string()
        ));
    }
    
    // Convert to rbee operation
    let operation = to_rbee_operation(request.clone(), state.default_hive_id.clone())?;
    
    // Submit to queen
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/jobs", state.queen_url))
        .json(&operation)
        .send()
        .await
        .map_err(|e| OpenAIError::Internal(format!("Failed to submit job: {}", e)))?;
    
    let job_response: JobResponse = response
        .json()
        .await
        .map_err(|e| OpenAIError::Internal(format!("Failed to parse response: {}", e)))?;
    
    // For now, return job info (streaming implementation in Phase 2)
    Ok(Json(serde_json::json!({
        "job_id": job_response.job_id,
        "sse_url": job_response.sse_url,
        "message": "Job submitted. Streaming implementation coming in Phase 2."
    })))
}
```

---

### Step 4: Update Router

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/router.rs`

**Replace with:**
```rust
use crate::{handlers, AdapterState};
use axum::{routing::{get, post}, Router};

pub fn create_openai_router(state: AdapterState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/:model", get(handlers::get_model))
        .with_state(state)
}
```

---

### Step 5: Mount in Queen

**File:** `bin/10_queen_rbee/src/main.rs`

**Add after other router setup:**
```rust
// Create OpenAI adapter (if feature enabled)
#[cfg(feature = "openai-adapter")]
{
    let openai_state = rbee_openai_adapter::AdapterState::new(
        format!("http://localhost:{}", args.port),
        "localhost".to_string(),
    );
    let openai_router = rbee_openai_adapter::create_openai_router(openai_state);
    app = app.nest("/openai", openai_router);
}
```

---

### Step 6: Add Feature Flag

**File:** `bin/10_queen_rbee/Cargo.toml`

**Add feature:**
```toml
[features]
default = []
openai-adapter = ["rbee-openai-adapter"]

[dependencies]
rbee-openai-adapter = { path = "../15_queen_rbee_crates/rbee-openai-adapter", optional = true }
```

---

### Step 7: Test Phase 1

**Manual Test:**
```bash
# Start queen with OpenAI adapter
cargo run --bin queen-rbee --features openai-adapter -- --port 7833

# Test translation
curl -X POST http://localhost:7833/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

**Expected Response:**
```json
{
  "job_id": "job_abc123",
  "sse_url": "/v1/jobs/job_abc123/stream",
  "message": "Job submitted. Streaming implementation coming in Phase 2."
}
```

---

## Phase 2: Streaming Responses (M2)

### Step 1: Implement Non-Streaming Response

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/streaming.rs` (NEW)

```rust
//! Response streaming and translation

use crate::error::OpenAIError;
use crate::types::*;
use crate::AdapterState;
use axum::response::sse::{Event, Sse};
use futures::StreamExt;
use jobs_contract::JobResponse;
use observability_narration_core::NarrationEvent;

pub async fn handle_non_streaming(
    job_response: JobResponse,
    request: ChatCompletionRequest,
    state: AdapterState,
) -> Result<ChatCompletionResponse, OpenAIError> {
    let sse_url = format!("{}{}", state.queen_url, job_response.sse_url);
    let client = reqwest::Client::new();
    let response = client.get(&sse_url).send().await
        .map_err(|e| OpenAIError::Internal(e.to_string()))?;
    
    let mut collected_tokens = Vec::new();
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| OpenAIError::Internal(e.to_string()))?;
        let text = String::from_utf8_lossy(&chunk);
        
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..];
                if data == "[DONE]" {
                    break;
                }
                
                if let Ok(event) = serde_json::from_str::<NarrationEvent>(data) {
                    if event.action == "infer_token" {
                        collected_tokens.push(event.formatted);
                    }
                }
            }
        }
    }
    
    let content = collected_tokens.join("");
    Ok(build_completion_response(
        &job_response.job_id,
        &request.model,
        content,
        collected_tokens.len() as u32,
    ))
}

fn build_completion_response(
    job_id: &str,
    model: &str,
    content: String,
    completion_tokens: u32,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{}", job_id),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens,
            total_tokens: completion_tokens,
        },
    }
}
```

---

### Step 2: Implement Streaming Response

**Add to `streaming.rs`:**
```rust
pub async fn handle_streaming(
    job_response: JobResponse,
    request: ChatCompletionRequest,
    state: AdapterState,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let sse_url = format!("{}{}", state.queen_url, job_response.sse_url);
    let job_id = job_response.job_id.clone();
    let model = request.model.clone();
    
    let stream = async_stream::stream! {
        let client = reqwest::Client::new();
        let response = match client.get(&sse_url).send().await {
            Ok(r) => r,
            Err(_) => return,
        };
        
        let mut stream = response.bytes_stream();
        let mut first_chunk = true;
        
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => return,
            };
            
            let text = String::from_utf8_lossy(&chunk);
            
            for line in text.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    
                    if data == "[DONE]" {
                        let final_chunk = build_completion_chunk(
                            &job_id,
                            &model,
                            None,
                            Some("stop".to_string()),
                        );
                        yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }
                    
                    if let Ok(event) = serde_json::from_str::<NarrationEvent>(data) {
                        if event.action == "infer_token" {
                            let mut chunk = build_completion_chunk(
                                &job_id,
                                &model,
                                Some(event.formatted),
                                None,
                            );
                            
                            if first_chunk {
                                chunk.choices[0].delta.role = Some("assistant".to_string());
                                first_chunk = false;
                            }
                            
                            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                        }
                    }
                }
            }
        }
    };
    
    Sse::new(stream)
}

fn build_completion_chunk(
    job_id: &str,
    model: &str,
    content: Option<String>,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: format!("chatcmpl-{}", job_id),
        object: "chat.completion.chunk".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: None,
                content,
            },
            finish_reason,
        }],
    }
}
```

---

### Step 3: Update Handler to Use Streaming

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

**Update `chat_completions`:**
```rust
use crate::streaming::{handle_non_streaming, handle_streaming};

pub async fn chat_completions(
    State(state): State<AdapterState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, OpenAIError> {
    // Validate
    if request.messages.is_empty() {
        return Err(OpenAIError::InvalidRequest(
            "Messages array cannot be empty".to_string()
        ));
    }
    
    // Convert and submit
    let operation = to_rbee_operation(request.clone(), state.default_hive_id.clone())?;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/jobs", state.queen_url))
        .json(&operation)
        .send()
        .await
        .map_err(|e| OpenAIError::Internal(format!("Failed to submit job: {}", e)))?;
    
    let job_response: JobResponse = response
        .json()
        .await
        .map_err(|e| OpenAIError::Internal(format!("Failed to parse response: {}", e)))?;
    
    // Handle streaming vs non-streaming
    if request.stream {
        Ok(handle_streaming(job_response, request, state).await.into_response())
    } else {
        let response = handle_non_streaming(job_response, request, state).await?;
        Ok(Json(response).into_response())
    }
}
```

---

### Step 4: Test Phase 2

**Test Non-Streaming:**
```bash
curl -X POST http://localhost:7833/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

**Test Streaming:**
```bash
curl -X POST http://localhost:7833/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

---

## Phase 3: Full Compatibility (M3)

### Step 1: Implement Models Endpoints

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

```rust
pub async fn list_models(
    State(state): State<AdapterState>,
) -> Result<Json<ModelListResponse>, OpenAIError> {
    let models: Vec<ModelInfo> = state.model_mappings
        .keys()
        .map(|model_id| ModelInfo {
            id: model_id.clone(),
            object: "model".to_string(),
            created: 1677610602,
            owned_by: "rbee".to_string(),
        })
        .collect();
    
    Ok(Json(ModelListResponse {
        object: "list".to_string(),
        data: models,
    }))
}

pub async fn get_model(
    Path(model_id): Path<String>,
    State(state): State<AdapterState>,
) -> Result<Json<ModelInfo>, OpenAIError> {
    if !state.model_mappings.contains_key(&model_id) {
        return Err(OpenAIError::ModelNotFound(
            format!("Model '{}' not found", model_id)
        ));
    }
    
    Ok(Json(ModelInfo {
        id: model_id,
        object: "model".to_string(),
        created: 1677610602,
        owned_by: "rbee".to_string(),
    }))
}
```

---

### Step 2: Add Token Counting

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/tokens.rs` (NEW)

```rust
//! Token counting utilities

/// Count tokens (approximate)
pub fn count_tokens(text: &str) -> u32 {
    let words = text.split_whitespace().count();
    ((words as f32) * 1.3) as u32
}

/// Count message tokens
pub fn count_message_tokens(messages: &[crate::types::ChatMessage]) -> u32 {
    messages
        .iter()
        .map(|msg| count_tokens(&msg.role) + count_tokens(&msg.content) + 4)
        .sum()
}
```

---

### Step 3: Update Usage Statistics

**Update `build_completion_response` in `streaming.rs`:**
```rust
use crate::tokens::{count_tokens, count_message_tokens};

fn build_completion_response(
    job_id: &str,
    model: &str,
    content: String,
    messages: &[ChatMessage],
) -> ChatCompletionResponse {
    let prompt_tokens = count_message_tokens(messages);
    let completion_tokens = count_tokens(&content);
    
    ChatCompletionResponse {
        // ... other fields ...
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}
```

---

## Testing Strategy

### Unit Tests

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/tests/translation_tests.rs` (NEW)

```rust
use rbee_openai_adapter::translation::*;
use rbee_openai_adapter::types::*;

#[test]
fn test_extract_prompt() {
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }
    ];
    assert_eq!(extract_prompt(messages), "user: Hello");
}

#[test]
fn test_model_mapping() {
    assert_eq!(map_model_name("gpt-3.5-turbo").unwrap(), "tinyllama");
    assert!(map_model_name("unknown").is_err());
}
```

---

### Integration Tests

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/tests/integration_tests.rs` (NEW)

```rust
#[tokio::test]
async fn test_openai_chat_completions() {
    // Start test queen
    // Mount OpenAI adapter
    // Send request
    // Verify response format
}
```

---

## Deployment Checklist

- [ ] Phase 1 complete (translation)
- [ ] Phase 2 complete (streaming)
- [ ] Phase 3 complete (full compatibility)
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Documentation updated
- [ ] Feature flag tested
- [ ] Performance benchmarked
- [ ] Error handling verified
- [ ] Logging/narration added

---

**Status:** ✅ COMPLETE IMPLEMENTATION GUIDE  
**Estimated Effort:** 2-3 weeks (M2-M3)  
**Priority:** Medium (M2 milestone)
