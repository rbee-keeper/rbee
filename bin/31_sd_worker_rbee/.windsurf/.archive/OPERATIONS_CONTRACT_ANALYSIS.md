# Operations Contract Analysis

**Date:** 2025-11-03  
**Analyst:** Architecture Review  
**Status:** 🚨 CRITICAL FINDING - Wrong Implementation Approach

---

## 🚨 Critical Discovery

**TEAM-395 implemented worker-specific endpoints WITHOUT considering the operations-contract architecture.**

This is a fundamental architectural violation that would have created:
1. **Fragmentation** - Each worker type with different APIs
2. **Client Complexity** - Clients need different code for LLM vs SD workers
3. **Contract Violation** - Bypasses the unified Operation enum system
4. **Maintenance Nightmare** - Changes need updates in multiple places

---

## ✅ Correct Architecture

### Current System (LLM Worker)

```
rbee-keeper
    ↓ (sends Operation::Infer)
operations-contract
    ↓ (routes to worker)
llm-worker-rbee
    ↓ (job_router.rs handles Operation::Infer)
inference execution
```

**Key Files:**
- `bin/97_contracts/operations-contract/src/lib.rs` - Operation enum (12 operations)
- `bin/30_llm_worker_rbee/src/job_router.rs` - Routes Operation::Infer to inference
- `bin/10_queen_rbee/src/job_router.rs` - Routes operations to Queen or forwards to Hive

### How It Works

1. **Client** (rbee-keeper) creates `Operation::Infer { prompt, model, ... }`
2. **Queen** receives via `POST /v1/jobs`, routes based on operation type
3. **Worker** (LLM or SD) receives operation, job_router dispatches to handler
4. **SSE Streaming** via `GET /v1/jobs/{job_id}/stream`

---

## 🔍 Current Operations

### Queen Operations (2)
- `Operation::Status` - Query hive and worker registries
- `Operation::Infer` - Schedule inference and route to worker (LLM ONLY)

### Hive Operations (8)
- WorkerSpawn, WorkerProcessList, WorkerProcessGet, WorkerProcessDelete
- ModelDownload, ModelList, ModelGet, ModelDelete

### Diagnostic (2)
- QueenCheck, HiveCheck

**Total:** 12 operations

---

## ❓ Key Questions

### Question 1: How does Operation::Infer work?

**Answer:** It's LLM-specific!

```rust
// operations-contract/src/requests.rs
pub struct InferRequest {
    pub hive_id: String,
    pub model: String,
    pub prompt: String,          // ← TEXT ONLY
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    // ... LLM sampling params
}
```

This is **NOT suitable for image generation** because:
- `prompt` is text input (no image data)
- Sampling params are LLM-specific (temperature, top_k, top_p)
- No image dimensions, guidance_scale, etc.

### Question 2: Do we need Operation::ImageGeneration?

**YES!** We need a NEW operation variant.

**Two Options:**

#### Option A: Separate Operations (RECOMMENDED)
```rust
// In operations-contract/src/lib.rs
pub enum Operation {
    // Existing
    Infer(InferRequest),           // LLM inference
    
    // NEW - Image operations
    ImageGeneration(ImageGenerationRequest),  // Text-to-image
    ImageTransform(ImageTransformRequest),    // Image-to-image
    ImageInpaint(ImageInpaintRequest),        // Inpainting
    
    // ... existing operations
}
```

#### Option B: Unified Inference (NOT RECOMMENDED)
```rust
pub enum Operation {
    Infer(UnifiedInferRequest),  // Works for both LLM and SD
}

pub enum InferRequest {
    TextGeneration { prompt, max_tokens, ... },
    ImageGeneration { prompt, width, height, ... },
}
```

**Why Option A is better:**
- Clear separation of concerns
- Type safety (can't mix LLM and SD params)
- Easier to extend (add ImageTransform, ImageInpaint separately)
- Follows existing pattern (separate operations for distinct tasks)

### Question 3: Do both workers use the same endpoints?

**YES! That's the whole point of operations-contract.**

**Current Setup:**
```
LLM Worker:
  POST /v1/jobs { "operation": "infer", "prompt": "...", ... }
  GET /v1/jobs/{job_id}/stream

SD Worker (SHOULD BE):
  POST /v1/jobs { "operation": "image_generation", "prompt": "...", ... }
  GET /v1/jobs/{job_id}/stream
```

**Same endpoints, different operation types.**

---

## 📋 Correct Implementation Plan

### Phase 1: Update operations-contract (1-2 hours)

**File:** `bin/97_contracts/operations-contract/src/lib.rs`

Add new operations:
```rust
pub enum Operation {
    // ... existing operations
    
    /// Generate image from text prompt
    ImageGeneration(ImageGenerationRequest),
    
    /// Transform image (img2img)
    ImageTransform(ImageTransformRequest),
    
    /// Inpaint image with mask
    ImageInpaint(ImageInpaintRequest),
}
```

**File:** `bin/97_contracts/operations-contract/src/requests.rs`

Add request types:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageGenerationRequest {
    /// Target hive (for routing)
    pub hive_id: String,
    /// Model to use
    pub model: String,
    /// Text prompt
    pub prompt: String,
    /// Negative prompt (optional)
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
    pub seed: Option<u64>,
    /// Worker ID to use (optional)
    pub worker_id: Option<String>,
}
```

**File:** `bin/97_contracts/operations-contract/src/operation_impl.rs`

Add routing logic:
```rust
impl Operation {
    pub fn target_server(&self) -> TargetServer {
        match self {
            Operation::Infer(_) => TargetServer::Queen,
            Operation::ImageGeneration(_) => TargetServer::Queen,  // NEW
            // ... rest
        }
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Operation::ImageGeneration(_) => "image_generation",  // NEW
            // ... rest
        }
    }
}
```

### Phase 2: Update sd-worker-rbee (2-3 hours)

**File:** `bin/31_sd_worker_rbee/src/job_router.rs` (NEW FILE)

Create job router (mirrors LLM worker):
```rust
use operations_contract::Operation;

pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let operation: Operation = serde_json::from_value(payload)?;
    
    match operation {
        Operation::ImageGeneration(req) => execute_image_generation(state, req).await,
        Operation::ImageTransform(req) => execute_image_transform(state, req).await,
        Operation::ImageInpaint(req) => execute_inpaint(state, req).await,
        _ => Err(anyhow!("Unsupported operation for SD worker")),
    }
}
```

**DELETE:**
- `src/http/jobs.rs` (TEAM-395's ad-hoc endpoint)
- `src/http/stream.rs` (TEAM-395's ad-hoc endpoint)

**KEEP:**
- `src/http/backend.rs`, `server.rs`, `routes.rs`, `health.rs`, `ready.rs` (TEAM-394's infrastructure)

**UPDATE:**
- `src/http/mod.rs` - Remove jobs/stream modules, add job_router
- `src/http/routes.rs` - Keep POST /v1/jobs, GET /v1/jobs/{job_id}/stream (same as LLM worker)

### Phase 3: Update Queen (1 hour)

**File:** `bin/10_queen_rbee/src/job_router.rs`

Add routing for new operations:
```rust
async fn route_operation(...) -> Result<()> {
    match operation {
        Operation::Infer(req) => {
            // Find LLM worker, forward
        }
        Operation::ImageGeneration(req) => {  // NEW
            // Find SD worker, forward
        }
        // ... rest
    }
}
```

### Phase 4: Integration Testing (1 hour)

Test end-to-end:
```bash
# Start SD worker
./target/debug/sd-worker-cuda --model sd-1.5 --port 8081

# Submit via rbee-keeper (after CLI update)
./rbee image generate --prompt "sunset" --model sd-1.5

# Or directly via Queen
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "image_generation",
    "hive_id": "localhost",
    "model": "sd-1.5",
    "prompt": "a beautiful sunset",
    "steps": 20
  }'
```

---

## 🎯 Benefits of Correct Approach

1. **Unified Client** - rbee-keeper uses same pattern for all workers
2. **Type Safety** - Compile-time guarantees via operations-contract
3. **Extensibility** - Easy to add ImageTransform, ImageInpaint
4. **Consistency** - All workers use same job router pattern
5. **Maintainability** - Single source of truth for operations

---

## 📊 Comparison

### ❌ TEAM-395's Approach (WRONG)

```
SD Worker has custom endpoints:
  POST /v1/jobs (custom CreateJobRequest)
  GET /v1/jobs/{job_id}/stream (custom SSE)

Problems:
- Not in operations-contract
- Different from LLM worker
- Client needs SD-specific code
- Can't route via Queen
```

### ✅ Correct Approach

```
SD Worker uses operations-contract:
  POST /v1/jobs { "operation": "image_generation", ... }
  GET /v1/jobs/{job_id}/stream

Benefits:
- In operations-contract (single source of truth)
- Same pattern as LLM worker
- Client reuses existing code
- Queen can route operations
```

---

## 🚨 Action Items

1. **STOP** all work on TEAM-395's implementation
2. **DELETE** jobs.rs and stream.rs (ad-hoc endpoints)
3. **UPDATE** operations-contract with ImageGeneration operations
4. **CREATE** job_router.rs for SD worker (mirrors LLM worker)
5. **UPDATE** Queen to route new operations
6. **TEST** end-to-end flow

---

## 📝 Notes for TEAM-396

**TEAM-396 should NOT work on "job registry" as described in TEAM-395's handoff.**

Instead, TEAM-396 should:
1. Review this analysis
2. Implement operations-contract updates
3. Create proper job_router.rs for SD worker
4. Follow LLM worker pattern exactly

The "job registry" issue TEAM-395 mentioned is already solved in the LLM worker via `JobRegistry<T>` from job-server crate.

---

**This analysis must be reviewed before any further work on SD worker.**
