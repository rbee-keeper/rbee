# OpenAI Flow Part 2: Request Translation & Job Submission

**Flow:** OpenAI Request → rbee Operation → Queen Job System  
**Date:** November 2, 2025  
**Status:** ⚠️ PLANNED (Implementation Required)

---

## Overview

This document details the translation layer that converts OpenAI API requests to rbee's internal `Operation` types and submits them to the queen job system.

**Example Translation:**
```
OpenAI Request → rbee Operation → POST /v1/jobs → Job Execution
```

---

## Step 1: Extract Prompt from Messages

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/translation.rs` (NEW)

**Function:**
```rust
/// Extract prompt from OpenAI messages array
///
/// Concatenates all messages into a single prompt string.
/// Format: "{role}: {content}\n"
///
/// # Example
/// ```
/// Input: [
///   {"role": "system", "content": "You are helpful"},
///   {"role": "user", "content": "Hello"}
/// ]
///
/// Output: "system: You are helpful\nuser: Hello"
/// ```
pub fn extract_prompt(messages: Vec<ChatMessage>) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}
```

**Test Cases:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_prompt_single_message() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }
        ];
        
        let prompt = extract_prompt(messages);
        assert_eq!(prompt, "user: Hello");
    }
    
    #[test]
    fn test_extract_prompt_multiple_messages() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        
        let prompt = extract_prompt(messages);
        assert_eq!(prompt, "system: You are helpful\nuser: Hello");
    }
}
```

---

## Step 2: Map Model Names

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/translation.rs`

**Function:**
```rust
/// Map OpenAI model names to rbee model IDs
///
/// This mapping can be configured via:
/// 1. Configuration file (preferred)
/// 2. Environment variables
/// 3. Hardcoded defaults (fallback)
pub fn map_model_name(openai_model: &str) -> Result<String, OpenAIError> {
    // TODO: Load from config file
    // For now, use hardcoded mapping
    
    match openai_model {
        // GPT-3.5 family
        "gpt-3.5-turbo" => Ok("tinyllama".to_string()),
        "gpt-3.5-turbo-16k" => Ok("tinyllama".to_string()),
        
        // GPT-4 family
        "gpt-4" => Ok("llama-7b".to_string()),
        "gpt-4-32k" => Ok("llama-13b".to_string()),
        
        // Unknown model
        _ => Err(OpenAIError::ModelNotFound(
            format!("Model '{}' not found. Available models: gpt-3.5-turbo, gpt-4", openai_model)
        ))
    }
}
```

**Configuration File (Planned):**

**File:** `~/.config/rbee/openai-models.toml`

```toml
# OpenAI model name mappings
[models]
"gpt-3.5-turbo" = "tinyllama"
"gpt-3.5-turbo-16k" = "tinyllama"
"gpt-4" = "llama-7b"
"gpt-4-32k" = "llama-13b"
"gpt-4-turbo" = "llama-70b"
```

**Test Cases:**
```rust
#[test]
fn test_map_model_name_gpt35() {
    let result = map_model_name("gpt-3.5-turbo");
    assert_eq!(result.unwrap(), "tinyllama");
}

#[test]
fn test_map_model_name_unknown() {
    let result = map_model_name("unknown-model");
    assert!(result.is_err());
}
```

---

## Step 3: Convert to rbee Operation

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/translation.rs`

**Function:**
```rust
/// Convert OpenAI request to rbee Operation::Infer
///
/// # Arguments
/// * `request` - OpenAI chat completion request
/// * `hive_id` - Target hive ID (from config or default)
///
/// # Returns
/// * `Operation::Infer` - Ready to submit to queen
pub fn to_rbee_operation(
    request: ChatCompletionRequest,
    hive_id: String,
) -> Result<Operation, OpenAIError> {
    // Step 3a: Extract prompt
    let prompt = extract_prompt(request.messages);
    
    // Step 3b: Map model name
    let model = map_model_name(&request.model)?;
    
    // Step 3c: Create InferRequest
    let infer_request = InferRequest {
        hive_id,
        model,
        prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: None,  // OpenAI doesn't have top_k
        device: None,  // Let scheduler decide
        worker_id: None,  // Let scheduler decide
        stream: request.stream,
    };
    
    // Step 3d: Wrap in Operation enum
    Ok(Operation::Infer(infer_request))
}
```

**Test Cases:**
```rust
#[test]
fn test_to_rbee_operation() {
    let request = ChatCompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }
        ],
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(100),
        stream: true,
        seed: None,
        stop: None,
    };
    
    let operation = to_rbee_operation(request, "localhost".to_string()).unwrap();
    
    match operation {
        Operation::Infer(req) => {
            assert_eq!(req.model, "tinyllama");
            assert_eq!(req.prompt, "user: Hello");
            assert_eq!(req.temperature, Some(0.7));
            assert_eq!(req.max_tokens, Some(100));
            assert_eq!(req.stream, true);
        }
        _ => panic!("Expected Operation::Infer"),
    }
}
```

---

## Step 4: Submit to Queen Job System

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

**Updated Handler:**
```rust
pub async fn chat_completions(
    State(state): State<AdapterState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, OpenAIError> {
    // Step 4a: Validate request
    if request.messages.is_empty() {
        return Err(OpenAIError::InvalidRequest(
            "Messages array cannot be empty".to_string()
        ));
    }
    
    // Step 4b: Convert to rbee operation
    let hive_id = state.default_hive_id.clone();
    let operation = to_rbee_operation(request.clone(), hive_id)?;
    
    // Step 4c: Submit to queen
    let queen_url = &state.queen_url;
    let client = reqwest::Client::new();
    
    let response = client
        .post(format!("{}/v1/jobs", queen_url))
        .json(&operation)
        .send()
        .await
        .map_err(|e| OpenAIError::Internal(format!("Failed to submit job: {}", e)))?;
    
    // Step 4d: Parse job response
    let job_response: JobResponse = response
        .json()
        .await
        .map_err(|e| OpenAIError::Internal(format!("Failed to parse response: {}", e)))?;
    
    // Step 4e: Handle streaming vs non-streaming
    if request.stream {
        // Return SSE stream (Part 3)
        handle_streaming_response(job_response, request, state).await
    } else {
        // Wait for completion and return full response (Part 3)
        handle_non_streaming_response(job_response, request, state).await
    }
}
```

---

## Step 5: Adapter State

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/lib.rs`

**State Definition:**
```rust
/// State for OpenAI adapter
#[derive(Clone)]
pub struct AdapterState {
    /// Queen URL (e.g., "http://localhost:7833")
    pub queen_url: String,
    
    /// Default hive ID for operations
    pub default_hive_id: String,
    
    /// Model name mappings
    pub model_mappings: Arc<HashMap<String, String>>,
}

impl AdapterState {
    /// Create new adapter state
    pub fn new(queen_url: String, default_hive_id: String) -> Self {
        Self {
            queen_url,
            default_hive_id,
            model_mappings: Arc::new(default_model_mappings()),
        }
    }
    
    /// Load model mappings from config file
    pub fn with_model_mappings(mut self, path: &Path) -> Result<Self> {
        let mappings = load_model_mappings(path)?;
        self.model_mappings = Arc::new(mappings);
        Ok(self)
    }
}

/// Default model mappings
fn default_model_mappings() -> HashMap<String, String> {
    let mut mappings = HashMap::new();
    mappings.insert("gpt-3.5-turbo".to_string(), "tinyllama".to_string());
    mappings.insert("gpt-4".to_string(), "llama-7b".to_string());
    mappings
}
```

---

## Step 6: Router Integration

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/router.rs`

**Updated Router:**
```rust
pub fn create_openai_router(state: AdapterState) -> Router {
    Router::new()
        // Chat completions
        .route("/v1/chat/completions", post(handlers::chat_completions))
        
        // Models
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/:model", get(handlers::get_model))
        
        // Add state
        .with_state(state)
}
```

---

## Step 7: Mount in Queen

### Planned Implementation

**File:** `bin/10_queen_rbee/src/main.rs`

**Mount OpenAI Router:**
```rust
// Create OpenAI adapter state
let openai_state = rbee_openai_adapter::AdapterState::new(
    format!("http://localhost:{}", args.port),
    "localhost".to_string(),
);

// Create OpenAI router
let openai_router = rbee_openai_adapter::create_openai_router(openai_state);

// Mount routers
let app = Router::new()
    .nest("/openai", openai_router)  // OpenAI-compatible API
    .nest("/v1", rbee_router)        // Native rbee API
    .layer(CorsLayer::permissive());
```

---

## Data Flow Summary

```
OpenAI Client
    ↓
POST /openai/v1/chat/completions
    ↓ ChatCompletionRequest
    ↓
chat_completions() [handlers.rs]
    ↓ validate request
    ↓
extract_prompt() [translation.rs]
    ↓ messages → single prompt
    ↓
map_model_name() [translation.rs]
    ↓ "gpt-3.5-turbo" → "tinyllama"
    ↓
to_rbee_operation() [translation.rs]
    ↓ create Operation::Infer
    ↓
POST /v1/jobs (internal)
    ↓ submit to queen
    ↓
JobResponse { job_id, sse_url }
    ↓
if streaming:
    handle_streaming_response() (Part 3)
else:
    handle_non_streaming_response() (Part 3)
```

---

## Error Handling

### Translation Errors

**Invalid Request:**
```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Messages array cannot be empty"
  }
}
```

**Model Not Found:**
```json
{
  "error": {
    "type": "model_not_found",
    "message": "Model 'unknown-model' not found. Available models: gpt-3.5-turbo, gpt-4"
  }
}
```

**Internal Error:**
```json
{
  "error": {
    "type": "internal_error",
    "message": "Failed to submit job: Connection refused"
  }
}
```

---

## Configuration

### Model Mappings File

**Location:** `~/.config/rbee/openai-models.toml`

**Format:**
```toml
[models]
"gpt-3.5-turbo" = "tinyllama"
"gpt-4" = "llama-7b"

[defaults]
hive_id = "localhost"
temperature = 0.7
max_tokens = 2048
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_prompt() {
        // Test single message
        // Test multiple messages
        // Test empty messages
    }
    
    #[test]
    fn test_map_model_name() {
        // Test known models
        // Test unknown models
        // Test case sensitivity
    }
    
    #[test]
    fn test_to_rbee_operation() {
        // Test full conversion
        // Test parameter mapping
        // Test error cases
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_openai_to_rbee_roundtrip() {
    // Start queen
    // Mount OpenAI adapter
    // Send OpenAI request
    // Verify rbee operation created
    // Verify job executed
}
```

---

## Implementation Checklist

**Phase 1 (M2):**
- [ ] Create `translation.rs` module
- [ ] Implement `extract_prompt()`
- [ ] Implement `map_model_name()`
- [ ] Implement `to_rbee_operation()`
- [ ] Create `AdapterState` struct
- [ ] Update `chat_completions()` handler
- [ ] Add unit tests
- [ ] Add integration tests

**Phase 2 (M2):**
- [ ] Load model mappings from config file
- [ ] Add configuration validation
- [ ] Add error handling
- [ ] Add logging/narration

---

**Next:** [OPENAI_FLOW_PART_3_RESPONSE_STREAMING.md](./OPENAI_FLOW_PART_3_RESPONSE_STREAMING.md) — Response translation and streaming
