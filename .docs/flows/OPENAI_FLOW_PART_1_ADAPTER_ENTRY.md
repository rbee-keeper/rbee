# OpenAI Flow Part 1: Adapter Entry Point

**Flow:** OpenAI Client → rbee OpenAI Adapter → Request Translation  
**Date:** November 2, 2025  
**Status:** ⚠️ STUB IMPLEMENTATION (Design Phase)

---

## Overview

This document traces the flow from when an external OpenAI-compatible client sends a request to rbee's OpenAI adapter, through the translation layer that converts OpenAI format to rbee's internal `Operation` types.

**Example Request:**
```bash
curl -X POST http://localhost:7833/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "stream": true
  }'
```

---

## ⚠️ Current Status: STUB IMPLEMENTATION

**The OpenAI adapter is currently a stub crate in design phase.**

**What Exists:**
- ✅ Module structure (`rbee-openai-adapter` crate)
- ✅ Router definitions (endpoints mapped)
- ✅ Type definitions (OpenAI request/response types)
- ✅ Handler stubs (return `NOT_IMPLEMENTED`)

**What's Missing:**
- ❌ Request translation logic (OpenAI → rbee)
- ❌ Response translation logic (rbee → OpenAI)
- ❌ Streaming SSE implementation
- ❌ Error code mapping
- ❌ Model name mapping

---

## Step 1: Crate Structure

### File: `bin/15_queen_rbee_crates/rbee-openai-adapter/src/lib.rs`

**Crate Purpose:**
```rust
//! OpenAI-compatible API adapter for rbee
//!
//! This crate provides an OpenAI-compatible HTTP API that translates OpenAI API calls
//! to rbee's internal Operation types. This allows existing applications built for
//! OpenAI to work with rbee without modification.
//!
//! # Architecture
//!
//! ```text
//! External App → /openai/v1/chat/completions → OpenAI Adapter → rbee Operations → queen-rbee
//! ```
```

**Module Structure:**
```rust
pub mod error;      // Error types and mapping
pub mod handlers;   // HTTP handlers
pub mod router;     // Axum router
pub mod types;      // OpenAI request/response types

pub use router::create_openai_router;
```

**Location:** Lines 1-43  
**Purpose:** OpenAI-compatible API adapter for rbee

---

## Step 2: Router Configuration

### File: `bin/15_queen_rbee_crates/rbee-openai-adapter/src/router.rs`

**Router Creation:**
```rust
/// Create OpenAI-compatible router
///
/// Returns a router that can be mounted at `/openai` prefix in queen-rbee.
pub fn create_openai_router() -> Router {
    Router::new()
        // Chat completions
        .route("/v1/chat/completions", post(handlers::chat_completions))
        
        // Models
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/:model", get(handlers::get_model))
}
```

**Location:** Lines 22-30  
**Function:** `create_openai_router()`  
**Purpose:** Create Axum router with OpenAI-compatible endpoints

**Endpoints:**
- `POST /v1/chat/completions` — Chat completions (streaming/non-streaming)
- `GET /v1/models` — List available models
- `GET /v1/models/{model}` — Get model details

---

## Step 3: Mounting in Queen

### File: `bin/10_queen_rbee/src/main.rs` (Planned)

**Router Mounting (Not Yet Implemented):**
```rust
// PLANNED: Mount OpenAI adapter at /openai prefix
let openai_router = rbee_openai_adapter::create_openai_router();

let app = Router::new()
    .nest("/openai", openai_router)  // OpenAI-compatible API
    .nest("/v1", rbee_router)        // Native rbee API
    .layer(CorsLayer::permissive());
```

**Full URLs:**
- OpenAI: `http://localhost:7833/openai/v1/chat/completions`
- Native: `http://localhost:7833/v1/jobs`

---

## Step 4: Chat Completions Handler (STUB)

### File: `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

**Handler Function:**
```rust
/// Handle POST /openai/v1/chat/completions
///
/// Translates OpenAI chat completion request to rbee Infer operation.
///
/// # Implementation TODO
///
/// 1. Extract prompt from messages array
/// 2. Map model name to rbee model ID
/// 3. Create rbee Operation::Infer
/// 4. If streaming: return SSE stream
/// 5. If non-streaming: wait for completion and return full response
pub async fn chat_completions(
    State(_state): State<()>, // TODO: Add proper state type
    Json(_request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}
```

**Location:** Lines 20-26  
**Function:** `chat_completions()`  
**Status:** ⚠️ STUB (returns `501 NOT_IMPLEMENTED`)

---

## Step 5: Request Types (OpenAI Format)

### File: `bin/15_queen_rbee_crates/rbee-openai-adapter/src/types.rs`

**ChatCompletionRequest:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID (e.g., "gpt-3.5-turbo")
    pub model: String,
    
    /// Array of messages
    pub messages: Vec<ChatMessage>,
    
    /// Optional: Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Optional: Temperature (0.0-2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Optional: Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Optional: Stream responses
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    
    /// Message content
    pub content: String,
}
```

**Purpose:** OpenAI-compatible request format

---

## Step 6: Planned Translation Logic

### Conceptual Flow (Not Yet Implemented)

**Translation Steps:**

#### 6a. Extract Prompt from Messages
```rust
// PLANNED: Concatenate messages into single prompt
fn extract_prompt(messages: Vec<ChatMessage>) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}
```

**Example:**
```
Input: [
  {"role": "system", "content": "You are helpful"},
  {"role": "user", "content": "Hello"}
]

Output: "system: You are helpful\nuser: Hello"
```

---

#### 6b. Map Model Name
```rust
// PLANNED: Map OpenAI model names to rbee model IDs
fn map_model_name(openai_model: &str) -> Result<String, Error> {
    match openai_model {
        "gpt-3.5-turbo" => Ok("tinyllama".to_string()),
        "gpt-4" => Ok("llama-7b".to_string()),
        _ => Err(Error::UnknownModel(openai_model.to_string()))
    }
}
```

---

#### 6c. Create rbee Operation
```rust
// PLANNED: Convert OpenAI request to rbee Operation::Infer
fn to_rbee_operation(
    request: ChatCompletionRequest,
    hive_id: String,
) -> Result<Operation, Error> {
    let prompt = extract_prompt(request.messages);
    let model = map_model_name(&request.model)?;
    
    Ok(Operation::Infer(InferRequest {
        hive_id,
        model,
        prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: None,  // OpenAI doesn't have top_k
        device: None,
        worker_id: None,
        stream: request.stream,
    }))
}
```

---

#### 6d. Submit to Queen Job System
```rust
// PLANNED: Submit operation to queen's job system
async fn submit_openai_job(
    operation: Operation,
    queen_url: &str,
) -> Result<JobResponse, Error> {
    let client = reqwest::Client::new();
    
    let response = client
        .post(format!("{}/v1/jobs", queen_url))
        .json(&operation)
        .send()
        .await?;
    
    let job_response: JobResponse = response.json().await?;
    Ok(job_response)
}
```

---

#### 6e. Stream or Wait
```rust
// PLANNED: Handle streaming vs non-streaming
async fn handle_response(
    job_response: JobResponse,
    stream: bool,
) -> Result<impl IntoResponse, Error> {
    if stream {
        // Return SSE stream in OpenAI format
        stream_openai_response(job_response).await
    } else {
        // Wait for completion and return full response
        wait_for_completion(job_response).await
    }
}
```

---

## Step 7: Response Types (OpenAI Format)

### File: `bin/15_queen_rbee_crates/rbee-openai-adapter/src/types.rs`

**ChatCompletionResponse (Non-Streaming):**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,  // "chat.completion"
    pub created: u64,    // Unix timestamp
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,  // "stop", "length", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
```

**ChatCompletionChunk (Streaming):**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,  // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoiceDelta {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}
```

---

## Step 8: Error Mapping

### File: `bin/15_queen_rbee_crates/rbee-openai-adapter/src/error.rs`

**Error Types (Planned):**
```rust
#[derive(Debug, thiserror::Error)]
pub enum OpenAIError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Unknown model: {0}")]
    UnknownModel(String),
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for OpenAIError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            OpenAIError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", msg)
            }
            OpenAIError::UnknownModel(model) => {
                (StatusCode::NOT_FOUND, "model_not_found", format!("Model {} not found", model))
            }
            OpenAIError::RateLimitExceeded => {
                (StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded", "Rate limit exceeded".to_string())
            }
            OpenAIError::Internal(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg)
            }
        };
        
        let body = json!({
            "error": {
                "type": error_type,
                "message": message,
            }
        });
        
        (status, Json(body)).into_response()
    }
}
```

---

## Data Flow Summary (Planned)

```
External OpenAI Client
    ↓
POST /openai/v1/chat/completions
    ↓ OpenAI format JSON
    ↓
chat_completions() [handlers.rs:20]
    ↓ extract ChatCompletionRequest
    ↓
extract_prompt() [PLANNED]
    ↓ messages → single prompt
    ↓
map_model_name() [PLANNED]
    ↓ "gpt-3.5-turbo" → "tinyllama"
    ↓
to_rbee_operation() [PLANNED]
    ↓ create Operation::Infer
    ↓
submit_openai_job() [PLANNED]
    ↓ POST /v1/jobs
    ↓
Queen Job System (Part 2)
    ↓ (continued in next part)
```

---

## Implementation Checklist

### Phase 1: Basic Translation
- [ ] Implement `extract_prompt()` function
- [ ] Implement `map_model_name()` function
- [ ] Implement `to_rbee_operation()` function
- [ ] Add state type with queen URL
- [ ] Submit operation to queen job system

### Phase 2: Non-Streaming
- [ ] Wait for job completion
- [ ] Collect all tokens
- [ ] Format as `ChatCompletionResponse`
- [ ] Return JSON response

### Phase 3: Streaming
- [ ] Connect to SSE stream
- [ ] Convert rbee SSE events to OpenAI chunks
- [ ] Format as `ChatCompletionChunk`
- [ ] Stream to client

### Phase 4: Error Handling
- [ ] Map rbee errors to OpenAI error format
- [ ] Handle timeout errors
- [ ] Handle model not found
- [ ] Handle worker unavailable

### Phase 5: Models Endpoint
- [ ] Implement `list_models()` handler
- [ ] Query model catalog
- [ ] Format as OpenAI `ModelListResponse`

---

## Key Files Referenced

| File | Purpose | Status |
|------|---------|--------|
| `bin/15_queen_rbee_crates/rbee-openai-adapter/src/lib.rs` | Crate root | ✅ Stub |
| `bin/15_queen_rbee_crates/rbee-openai-adapter/src/router.rs` | Router config | ✅ Stub |
| `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs` | HTTP handlers | ⚠️ Stub |
| `bin/15_queen_rbee_crates/rbee-openai-adapter/src/types.rs` | Request/response types | ✅ Stub |
| `bin/15_queen_rbee_crates/rbee-openai-adapter/src/error.rs` | Error mapping | ⚠️ Stub |

---

## Configuration

**Mounting Point:**
- Planned: `/openai` prefix in queen-rbee
- Full URL: `http://localhost:7833/openai/v1/chat/completions`

**Model Mapping:**
- Needs configuration file or database
- Maps OpenAI model names to rbee model IDs

---

**Next:** [OPENAI_FLOW_PART_2_QUEEN_ROUTING.md](./OPENAI_FLOW_PART_2_QUEEN_ROUTING.md) — Queen receives operation, routes to scheduler
