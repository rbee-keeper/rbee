# OpenAI Flow Part 3: Response Translation & Streaming

**Flow:** rbee SSE Events → OpenAI Format → Client  
**Date:** November 2, 2025  
**Status:** ⚠️ PLANNED (Implementation Required)

---

## Overview

This document details the response translation layer that converts rbee's SSE events to OpenAI-compatible response formats, supporting both streaming and non-streaming modes.

---

## Step 1: Non-Streaming Response

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

**Function:**
```rust
/// Handle non-streaming response
///
/// Waits for job completion, collects all tokens, and returns full response
async fn handle_non_streaming_response(
    job_response: JobResponse,
    request: ChatCompletionRequest,
    state: AdapterState,
) -> Result<impl IntoResponse, OpenAIError> {
    // Step 1a: Connect to SSE stream
    let sse_url = format!("{}{}", state.queen_url, job_response.sse_url);
    let client = reqwest::Client::new();
    let response = client.get(&sse_url).send().await
        .map_err(|e| OpenAIError::Internal(format!("Failed to connect to SSE: {}", e)))?;
    
    // Step 1b: Collect all tokens
    let mut collected_tokens = Vec::new();
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| OpenAIError::Internal(e.to_string()))?;
        let text = String::from_utf8_lossy(&chunk);
        
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..];
                
                // Check for [DONE] marker
                if data == "[DONE]" {
                    break;
                }
                
                // Parse narration event
                if let Ok(event) = serde_json::from_str::<NarrationEvent>(data) {
                    // Extract token from event
                    if event.action == "infer_token" {
                        collected_tokens.push(event.formatted);
                    }
                }
            }
        }
    }
    
    // Step 1c: Build OpenAI response
    let completion_text = collected_tokens.join("");
    let response = build_chat_completion_response(
        &job_response.job_id,
        &request.model,
        completion_text,
        collected_tokens.len() as u32,
    );
    
    Ok(Json(response))
}

/// Build OpenAI chat completion response
fn build_chat_completion_response(
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
            prompt_tokens: 0,  // TODO: Calculate from prompt
            completion_tokens,
            total_tokens: completion_tokens,
        },
    }
}
```

**Example Response:**
```json
{
  "id": "chatcmpl-job_abc123",
  "object": "chat.completion",
  "created": 1730563200,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

---

## Step 2: Streaming Response

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

**Function:**
```rust
/// Handle streaming response
///
/// Streams tokens as OpenAI chunks
async fn handle_streaming_response(
    job_response: JobResponse,
    request: ChatCompletionRequest,
    state: AdapterState,
) -> Result<impl IntoResponse, OpenAIError> {
    // Step 2a: Connect to SSE stream
    let sse_url = format!("{}{}", state.queen_url, job_response.sse_url);
    let job_id = job_response.job_id.clone();
    let model = request.model.clone();
    
    // Step 2b: Create streaming response
    let stream = async_stream::stream! {
        let client = reqwest::Client::new();
        let response = match client.get(&sse_url).send().await {
            Ok(r) => r,
            Err(e) => {
                yield Err(OpenAIError::Internal(e.to_string()));
                return;
            }
        };
        
        let mut stream = response.bytes_stream();
        let mut first_chunk = true;
        
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(OpenAIError::Internal(e.to_string()));
                    return;
                }
            };
            
            let text = String::from_utf8_lossy(&chunk);
            
            for line in text.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    
                    // Check for [DONE] marker
                    if data == "[DONE]" {
                        // Send final chunk with finish_reason
                        let final_chunk = build_chat_completion_chunk(
                            &job_id,
                            &model,
                            None,
                            Some("stop".to_string()),
                        );
                        yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
                        
                        // Send [DONE] marker
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }
                    
                    // Parse narration event
                    if let Ok(event) = serde_json::from_str::<NarrationEvent>(data) {
                        // Extract token from event
                        if event.action == "infer_token" {
                            let chunk = build_chat_completion_chunk(
                                &job_id,
                                &model,
                                Some(event.formatted),
                                None,
                            );
                            
                            // Add role in first chunk
                            let mut chunk = chunk;
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
    
    Ok(Sse::new(stream))
}

/// Build OpenAI chat completion chunk
fn build_chat_completion_chunk(
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
                role: None,  // Only in first chunk
                content,
            },
            finish_reason,
        }],
    }
}
```

**Example Streaming Response:**
```
data: {"id":"chatcmpl-job_abc123","object":"chat.completion.chunk","created":1730563200,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-job_abc123","object":"chat.completion.chunk","created":1730563200,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-job_abc123","object":"chat.completion.chunk","created":1730563200,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Step 3: Event Translation

### rbee Event → OpenAI Chunk

**rbee SSE Event:**
```json
{
  "action": "infer_token",
  "actor": "llm-worker-rbee",
  "formatted": "Hello",
  "job_id": "job_abc123",
  "timestamp": "2025-11-02T17:00:00Z"
}
```

**OpenAI Chunk:**
```json
{
  "id": "chatcmpl-job_abc123",
  "object": "chat.completion.chunk",
  "created": 1730563200,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "delta": {
      "content": "Hello"
    },
    "finish_reason": null
  }]
}
```

---

## Step 4: Token Counting

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/tokens.rs` (NEW)

**Function:**
```rust
/// Count tokens in text (approximate)
///
/// Uses simple word-based estimation.
/// For accurate counting, use tiktoken library.
pub fn count_tokens(text: &str) -> u32 {
    // Simple approximation: ~1.3 tokens per word
    let words = text.split_whitespace().count();
    ((words as f32) * 1.3) as u32
}

/// Count tokens in messages
pub fn count_message_tokens(messages: &[ChatMessage]) -> u32 {
    messages
        .iter()
        .map(|msg| {
            // Role + content + formatting overhead
            count_tokens(&msg.role) + count_tokens(&msg.content) + 4
        })
        .sum()
}
```

**Better Alternative (Future):**
```rust
// Use tiktoken for accurate counting
use tiktoken_rs::cl100k_base;

pub fn count_tokens_accurate(text: &str) -> u32 {
    let bpe = cl100k_base().unwrap();
    bpe.encode_with_special_tokens(text).len() as u32
}
```

---

## Step 5: Error Translation

### rbee Error → OpenAI Error

**rbee Error Event:**
```json
{
  "action": "execute_error",
  "formatted": "❌ Job job_abc123 failed: Model not found",
  "job_id": "job_abc123"
}
```

**OpenAI Error Response:**
```json
{
  "error": {
    "type": "internal_error",
    "message": "Model not found",
    "code": "model_not_found"
  }
}
```

**Error Mapping:**
```rust
fn translate_rbee_error(event: &NarrationEvent) -> OpenAIError {
    if event.formatted.contains("Model not found") {
        OpenAIError::ModelNotFound(event.formatted.clone())
    } else if event.formatted.contains("Worker unavailable") {
        OpenAIError::Internal("Service temporarily unavailable".to_string())
    } else {
        OpenAIError::Internal(event.formatted.clone())
    }
}
```

---

## Step 6: Models Endpoint

### Planned Implementation

**File:** `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`

**List Models:**
```rust
pub async fn list_models(
    State(state): State<AdapterState>,
) -> Result<Json<ModelListResponse>, OpenAIError> {
    // Get available models from mappings
    let models: Vec<ModelInfo> = state.model_mappings
        .keys()
        .map(|model_id| ModelInfo {
            id: model_id.clone(),
            object: "model".to_string(),
            created: 1677610602,  // Static timestamp
            owned_by: "rbee".to_string(),
        })
        .collect();
    
    Ok(Json(ModelListResponse {
        object: "list".to_string(),
        data: models,
    }))
}
```

**Example Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1677610602,
      "owned_by": "rbee"
    },
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1677610602,
      "owned_by": "rbee"
    }
  ]
}
```

**Get Model:**
```rust
pub async fn get_model(
    Path(model_id): Path<String>,
    State(state): State<AdapterState>,
) -> Result<Json<ModelInfo>, OpenAIError> {
    // Check if model exists
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

## Complete Flow Example

**OpenAI Client Request:**
```bash
curl -X POST http://localhost:7833/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

**rbee Internal Flow:**
```
1. OpenAI adapter receives request
2. Translates to Operation::Infer
3. Submits to queen (POST /v1/jobs)
4. Queen routes to scheduler
5. Scheduler selects worker
6. Worker generates tokens
7. Tokens streamed via SSE
8. OpenAI adapter translates events
9. Client receives OpenAI chunks
```

**Client Receives:**
```
data: {"id":"chatcmpl-job_abc123","object":"chat.completion.chunk",...,"choices":[{"delta":{"role":"assistant","content":"Hello"}}]}

data: {"id":"chatcmpl-job_abc123","object":"chat.completion.chunk",...,"choices":[{"delta":{"content":"!"}}]}

data: {"id":"chatcmpl-job_abc123","object":"chat.completion.chunk",...,"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Implementation Checklist

**Phase 2 (M2):**
- [ ] Implement `handle_non_streaming_response()`
- [ ] Implement `handle_streaming_response()`
- [ ] Implement `build_chat_completion_response()`
- [ ] Implement `build_chat_completion_chunk()`
- [ ] Implement event translation
- [ ] Implement error translation
- [ ] Implement `list_models()` handler
- [ ] Implement `get_model()` handler
- [ ] Add token counting (approximate)
- [ ] Add unit tests
- [ ] Add integration tests

**Phase 3 (M3):**
- [ ] Add accurate token counting (tiktoken)
- [ ] Add usage statistics
- [ ] Add function calling support (if needed)
- [ ] Add stop sequences support
- [ ] Add seed support (deterministic generation)

---

**Next:** [OPENAI_FLOW_PART_4_IMPLEMENTATION_GUIDE.md](./OPENAI_FLOW_PART_4_IMPLEMENTATION_GUIDE.md) — Complete implementation guide
