# Correct Implementation Plan for SD Worker

**Date:** 2025-11-03  
**Status:** 🎯 READY TO IMPLEMENT  
**Prerequisite:** Read `OPERATIONS_CONTRACT_ANALYSIS.md` first

---

## Executive Summary

SD Worker must integrate with `operations-contract` (like LLM worker does), not create custom endpoints.

**Key Insight:** Both LLM and SD workers accept `Operation` enum variants and route via `job_router.rs`. The difference is which operations they handle.

---

## Phase 1: Update operations-contract

**Location:** `bin/97_contracts/operations-contract/`

### 1.1 Add New Operation Variants

**File:** `src/lib.rs`

```rust
pub enum Operation {
    // ... existing operations
    
    /// Generate image from text prompt (Stable Diffusion)
    ImageGeneration(ImageGenerationRequest),
    
    /// Transform image (img2img)
    ImageTransform(ImageTransformRequest),
    
    /// Inpaint image with mask
    ImageInpaint(ImageInpaintRequest),
}
```

### 1.2 Add Request Types

**File:** `src/requests.rs`

```rust
/// Request to generate image from text prompt
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageGenerationRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
    
    /// Model to use (e.g., "stable-diffusion-v1-5")
    pub model: String,
    
    /// Text prompt
    pub prompt: String,
    
    /// Negative prompt (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    
    /// Number of inference steps
    #[serde(default = "default_steps")]
    pub steps: usize,
    
    /// Guidance scale
    #[serde(default = "default_guidance")]
    pub guidance_scale: f64,
    
    /// Image width
    #[serde(default = "default_width")]
    pub width: usize,
    
    /// Image height
    #[serde(default = "default_height")]
    pub height: usize,
    
    /// Random seed (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    
    /// Specific worker ID (optional, for direct routing)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
}

fn default_steps() -> usize { 20 }
fn default_guidance() -> f64 { 7.5 }
fn default_width() -> usize { 512 }
fn default_height() -> usize { 512 }

/// Request to transform image (img2img)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageTransformRequest {
    pub hive_id: String,
    pub model: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// Base64-encoded input image
    pub init_image: String,
    /// Strength of transformation (0.0-1.0)
    #[serde(default = "default_strength")]
    pub strength: f64,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub worker_id: Option<String>,
}

fn default_strength() -> f64 { 0.8 }

/// Request to inpaint image
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageInpaintRequest {
    pub hive_id: String,
    pub model: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// Base64-encoded input image
    pub init_image: String,
    /// Base64-encoded mask (white = inpaint, black = keep)
    pub mask_image: String,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub worker_id: Option<String>,
}
```

### 1.3 Update Operation Implementation

**File:** `src/operation_impl.rs`

```rust
impl Operation {
    pub fn target_server(&self) -> TargetServer {
        match self {
            Operation::Infer(_) => TargetServer::Queen,
            Operation::ImageGeneration(_) => TargetServer::Queen,  // NEW
            Operation::ImageTransform(_) => TargetServer::Queen,   // NEW
            Operation::ImageInpaint(_) => TargetServer::Queen,     // NEW
            // ... rest
        }
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Operation::ImageGeneration(_) => "image_generation",  // NEW
            Operation::ImageTransform(_) => "image_transform",    // NEW
            Operation::ImageInpaint(_) => "image_inpaint",        // NEW
            // ... rest
        }
    }
}
```

---

## Phase 2: Create SD Worker Job Router

**Location:** `bin/31_sd_worker_rbee/src/`

### 2.1 Create job_router.rs

**File:** `src/job_router.rs` (NEW FILE)

```rust
// Created by: TEAM-396
// SD Worker job router - mirrors LLM worker pattern

use anyhow::Result;
use job_server::JobRegistry;
use observability_narration_core::sse_sink;
use operations_contract::Operation;
use std::sync::Arc;

use crate::backend::generation_engine::GenerationEngine;
use crate::backend::request_queue::{GenerationRequest, GenerationResponse};

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<GenerationResponse>>,
    pub engine: Arc<GenerationEngine>,
}

/// Response from job creation
#[derive(Debug, serde::Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create a new job and store its payload
///
/// Mirrors LLM worker pattern (bin/30_llm_worker_rbee/src/job_router.rs)
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    // Parse operation from JSON
    let operation: Operation = serde_json::from_value(payload)?;
    
    // Route to appropriate handler
    match operation {
        Operation::ImageGeneration(req) => execute_image_generation(state, req).await,
        Operation::ImageTransform(req) => execute_image_transform(state, req).await,
        Operation::ImageInpaint(req) => execute_inpaint(state, req).await,
        _ => Err(anyhow::anyhow!("Unsupported operation for SD worker: {:?}", operation)),
    }
}

/// Execute image generation operation
async fn execute_image_generation(
    state: JobState,
    req: operations_contract::ImageGenerationRequest,
) -> Result<JobResponse> {
    // Create job in registry
    let job_id = state.registry.create_job();
    
    // Create SSE channel for narration
    sse_sink::create_job_channel(job_id.clone(), 1000);
    
    // Create response channel
    let (response_tx, response_rx) = tokio::sync::mpsc::channel(10);
    state.registry.set_token_receiver(&job_id, response_rx);
    
    // Convert to SamplingConfig
    let config = crate::backend::sampling::SamplingConfig {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        steps: req.steps,
        guidance_scale: req.guidance_scale,
        width: req.width,
        height: req.height,
        seed: req.seed,
        ..Default::default()
    };
    
    // Create generation request
    let generation_request = GenerationRequest {
        job_id: job_id.clone(),
        config,
    };
    
    // Submit to engine
    state.engine.submit(generation_request, response_tx).await?;
    
    Ok(JobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/v1/jobs/{}/stream", job_id),
    })
}

/// Execute image transform operation (TODO: TEAM-397)
async fn execute_image_transform(
    _state: JobState,
    _req: operations_contract::ImageTransformRequest,
) -> Result<JobResponse> {
    Err(anyhow::anyhow!("ImageTransform not yet implemented (TEAM-397)"))
}

/// Execute inpaint operation (TODO: TEAM-398)
async fn execute_inpaint(
    _state: JobState,
    _req: operations_contract::ImageInpaintRequest,
) -> Result<JobResponse> {
    Err(anyhow::anyhow!("ImageInpaint not yet implemented (TEAM-398)"))
}
```

### 2.2 Update HTTP Layer

**File:** `src/http/jobs.rs` (NEW FILE - mirrors LLM worker)

```rust
// Created by: TEAM-396
// HTTP wrapper for job creation

use crate::job_router::{JobResponse, JobState};
use axum::{extract::State, Json};

pub async fn handle_create_job(
    State(state): State<JobState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<JobResponse>, (axum::http::StatusCode, String)> {
    crate::job_router::create_job(state, payload)
        .await
        .map(Json)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
```

**File:** `src/http/stream.rs` (NEW FILE - mirrors LLM worker)

```rust
// Created by: TEAM-396
// SSE streaming for job results

use crate::job_router::JobState;
use axum::{
    extract::{Path, State},
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use std::convert::Infallible;

pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<JobState>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (axum::http::StatusCode, String)> {
    // Take receiver from registry
    let mut response_rx = state.registry.take_token_receiver(&job_id)
        .ok_or_else(|| (axum::http::StatusCode::NOT_FOUND, "Job not found".to_string()))?;
    
    // Stream events
    let stream = async_stream::stream! {
        while let Some(response) = response_rx.recv().await {
            match response {
                crate::backend::request_queue::GenerationResponse::Progress { step, total } => {
                    let json = serde_json::json!({
                        "type": "progress",
                        "step": step,
                        "total": total,
                    });
                    yield Ok(Event::default().event("progress").data(json.to_string()));
                }
                crate::backend::request_queue::GenerationResponse::Complete { image } => {
                    let base64 = crate::backend::image_utils::image_to_base64(&image).unwrap();
                    let json = serde_json::json!({
                        "type": "complete",
                        "image": base64,
                        "format": "png",
                    });
                    yield Ok(Event::default().event("complete").data(json.to_string()));
                }
                crate::backend::request_queue::GenerationResponse::Error { message } => {
                    let json = serde_json::json!({
                        "type": "error",
                        "message": message,
                    });
                    yield Ok(Event::default().event("error").data(json.to_string()));
                }
            }
        }
    };
    
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
```

### 2.3 Wire Up Routes

**File:** `src/http/routes.rs`

```rust
use crate::http::{health, jobs, ready, stream};
use crate::job_router::JobState;
use axum::{routing::{get, post}, Router};

pub fn create_router(state: JobState) -> Router {
    Router::new()
        // Health and readiness
        .route("/health", get(health::health_check))
        .route("/ready", get(ready::readiness_check))
        // Job endpoints (same as LLM worker!)
        .route("/v1/jobs", post(jobs::handle_create_job))
        .route("/v1/jobs/:job_id/stream", get(stream::handle_stream_job))
        .with_state(state)
        .layer(/* middleware */)
}
```

---

## Phase 3: Update Queen for Routing

**Location:** `bin/10_queen_rbee/src/job_router.rs`

### 3.1 Add Image Operation Routing

```rust
async fn route_operation(...) -> Result<()> {
    match operation {
        Operation::Infer(req) => {
            // Find LLM worker with model loaded
            // Forward to worker via HTTP
        }
        
        Operation::ImageGeneration(req) => {  // NEW
            // Find SD worker with model loaded
            // Forward to worker via HTTP
            
            // Simplified pseudocode:
            let workers = state.hive_registry.list_online_workers();
            let sd_worker = workers.iter()
                .find(|w| w.model.as_deref() == Some(&req.model))
                .ok_or_else(|| anyhow!("No SD worker with model {}", req.model))?;
            
            // Forward operation to worker
            let worker_url = format!("http://localhost:{}/v1/jobs", sd_worker.port);
            // ... HTTP POST to worker
        }
        
        // ... rest of operations
    }
}
```

---

## Phase 4: Testing

### 4.1 Unit Tests

Test the job router logic:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_image_generation_routing() {
        let state = JobState { /* mock state */ };
        let req = operations_contract::ImageGenerationRequest {
            hive_id: "localhost".to_string(),
            model: "sd-1.5".to_string(),
            prompt: "test".to_string(),
            ..Default::default()
        };
        
        let result = execute_image_generation(state, req).await;
        assert!(result.is_ok());
    }
}
```

### 4.2 Integration Tests

Test end-to-end flow:

```bash
# Start SD worker
./target/debug/sd-worker-cuda --model sd-1.5 --port 8081

# Submit via Queen
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "image_generation",
    "hive_id": "localhost",
    "model": "sd-1.5",
    "prompt": "a beautiful sunset over mountains",
    "steps": 20,
    "width": 512,
    "height": 512
  }'

# Get job_id from response, then stream:
curl http://localhost:7833/v1/jobs/{job_id}/stream
```

---

## Phase 5: CLI Integration

**Location:** `bin/00_rbee_keeper/src/main.rs`

Add image generation commands:

```rust
enum Commands {
    // ... existing commands
    
    Image(ImageCommands),
}

enum ImageCommands {
    Generate {
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "20")]
        steps: usize,
        // ... other options
    },
    Transform { /* img2img options */ },
    Inpaint { /* inpainting options */ },
}
```

---

## Comparison: Wrong vs Right

### ❌ TEAM-395 Approach (WRONG)

```
Client → SD Worker POST /v1/jobs (custom endpoint)
         ↓
     Custom CreateJobRequest (not in operations-contract)
         ↓
     Ad-hoc job handler
         ↓
     Can't route via Queen
```

**Problems:**
- Not in operations-contract
- Different from LLM worker
- Client needs SD-specific code
- Can't leverage Queen's scheduling

### ✅ Correct Approach

```
Client → Queen POST /v1/jobs
         ↓
     Operation::ImageGeneration (from operations-contract)
         ↓
     Queen routes to SD Worker
         ↓
     SD Worker job_router handles it
         ↓
     Same endpoints as LLM worker!
```

**Benefits:**
- Unified contract
- Client uses same code
- Queen can schedule
- Consistent patterns

---

## Summary

1. **operations-contract**: Single source of truth for all operations
2. **job_router.rs**: Each worker implements its own routing logic
3. **Same HTTP endpoints**: All workers use POST /v1/jobs and GET /v1/jobs/{id}/stream
4. **Different operations**: LLM handles `Infer`, SD handles `ImageGeneration`

**This is the correct architecture. TEAM-396 should implement this.**
