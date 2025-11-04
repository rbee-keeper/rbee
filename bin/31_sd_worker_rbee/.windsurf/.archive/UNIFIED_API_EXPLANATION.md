# Unified API: LLM vs Image Operations

**Date:** 2025-11-03  
**Status:** 📋 ARCHITECTURAL DESIGN

---

## Current State (Operations-Contract)

### ✅ LLM Operations (IMPLEMENTED)

**Operation:** `Operation::Infer(InferRequest)`

**Request Structure:**
```rust
pub struct InferRequest {
    pub hive_id: String,      // Which hive to route to
    pub model: String,         // Model name (e.g., "llama-3.2-1b")
    pub prompt: String,        // Text prompt
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}
```

**Handled By:**
- **Queen** (`bin/10_queen_rbee`) - Routes to appropriate LLM worker
- **LLM Worker** (`bin/30_llm_worker_rbee`) - Executes inference

**CLI Usage:**
```bash
# Using rbee-keeper CLI
rbee-keeper infer \
  --hive localhost \
  --model llama-3.2-1b \
  --prompt "Hello, world!" \
  --max-tokens 100
```

**HTTP API:**
```bash
# Direct to Queen
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "infer",
    "hive_id": "localhost",
    "model": "llama-3.2-1b",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'

# Returns: { "job_id": "uuid", "sse_url": "/v1/jobs/{job_id}/stream" }

# Stream results
curl http://localhost:7833/v1/jobs/{job_id}/stream
```

---

### ❌ Image Operations (NOT YET IN CONTRACT)

**TEAM-397 TODO:** Add these to `operations-contract`:

#### Operation 1: Image Generation (Text-to-Image)

```rust
pub enum Operation {
    // ... existing operations
    
    /// Generate image from text prompt (Stable Diffusion)
    ImageGeneration(ImageGenerationRequest),
}

pub struct ImageGenerationRequest {
    pub hive_id: String,           // Which hive to route to
    pub model: String,              // Model name (e.g., "stable-diffusion-v1-5")
    pub prompt: String,             // Text prompt
    pub negative_prompt: Option<String>,
    pub steps: usize,               // Default: 20
    pub guidance_scale: f64,        // Default: 7.5
    pub width: usize,               // Default: 512
    pub height: usize,              // Default: 512
    pub seed: Option<u64>,
    pub worker_id: Option<String>,  // Optional: direct routing
}
```

**Would Be Handled By:**
- **Queen** - Routes to SD worker with model loaded
- **SD Worker** (`bin/31_sd_worker_rbee`) - Generates image

**CLI Usage (FUTURE):**
```bash
# Using rbee-keeper CLI
rbee-keeper image generate \
  --hive localhost \
  --model stable-diffusion-v1-5 \
  --prompt "A beautiful sunset over mountains" \
  --steps 20 \
  --width 512 \
  --height 512 \
  --output sunset.png
```

**HTTP API (FUTURE):**
```bash
# Direct to Queen
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "image_generation",
    "hive_id": "localhost",
    "model": "stable-diffusion-v1-5",
    "prompt": "A beautiful sunset over mountains",
    "steps": 20,
    "width": 512,
    "height": 512
  }'

# Returns: { "job_id": "uuid", "sse_url": "/v1/jobs/{job_id}/stream" }

# Stream progress and get result
curl http://localhost:7833/v1/jobs/{job_id}/stream
# Events:
# - progress: {"type": "progress", "step": 5, "total": 20}
# - complete: {"type": "complete", "image": "base64...", "format": "png"}
```

#### Operation 2: Image Transform (Image-to-Image)

```rust
pub enum Operation {
    /// Transform image based on prompt (img2img)
    ImageTransform(ImageTransformRequest),
}

pub struct ImageTransformRequest {
    pub hive_id: String,
    pub model: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub init_image: String,         // Base64-encoded input image
    pub strength: f64,              // 0.0-1.0, how much to transform
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub worker_id: Option<String>,
}
```

**CLI Usage (FUTURE):**
```bash
rbee-keeper image transform \
  --hive localhost \
  --model stable-diffusion-v1-5 \
  --prompt "Make it look like a painting" \
  --input photo.jpg \
  --strength 0.8 \
  --output painting.png
```

#### Operation 3: Image Inpaint

```rust
pub enum Operation {
    /// Inpaint masked regions of an image
    ImageInpaint(ImageInpaintRequest),
}

pub struct ImageInpaintRequest {
    pub hive_id: String,
    pub model: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub init_image: String,         // Base64-encoded input image
    pub mask_image: String,         // Base64-encoded mask (white = inpaint)
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub worker_id: Option<String>,
}
```

**CLI Usage (FUTURE):**
```bash
rbee-keeper image inpaint \
  --hive localhost \
  --model stable-diffusion-v1-5 \
  --prompt "A red car" \
  --input photo.jpg \
  --mask mask.png \
  --output inpainted.png
```

---

## Unified API Architecture

### At the Queen Level (Orchestrator)

**Queen receives ALL operations via:**
- HTTP: `POST http://localhost:7833/v1/jobs`
- Payload: `{ "operation": "...", ...request fields... }`

**Queen's job_router.rs routes based on operation type:**

```rust
// bin/10_queen_rbee/src/job_router.rs
pub async fn route_operation(operation: Operation, state: JobState) -> Result<JobResponse> {
    match operation {
        // LLM Operations
        Operation::Infer(req) => {
            // 1. Find LLM worker with model loaded
            let worker = state.hive_registry
                .find_workers_with_model(&req.model)
                .first()
                .ok_or_else(|| anyhow!("No worker with model {}", req.model))?;
            
            // 2. Forward to worker
            let worker_url = format!("http://localhost:{}/v1/jobs", worker.port);
            // ... HTTP POST to worker
        }
        
        // Image Operations (FUTURE)
        Operation::ImageGeneration(req) => {
            // 1. Find SD worker with model loaded
            let worker = state.hive_registry
                .find_workers_with_model(&req.model)
                .first()
                .ok_or_else(|| anyhow!("No SD worker with model {}", req.model))?;
            
            // 2. Forward to worker
            let worker_url = format!("http://localhost:{}/v1/jobs", worker.port);
            // ... HTTP POST to worker
        }
        
        // Hive Operations (worker management, model management)
        Operation::WorkerSpawn(req) => {
            // Route to hive
            let hive_url = format!("http://{}:7835/v1/jobs", req.hive_id);
            // ... HTTP POST to hive
        }
        
        _ => Err(anyhow!("Unsupported operation")),
    }
}
```

**Key Point:** Queen doesn't care if it's LLM or Image - it just routes based on:
1. **Operation type** (Infer vs ImageGeneration)
2. **Model availability** (which worker has the model loaded)
3. **Worker type** (LLM worker vs SD worker)

---

### At the Worker Level

**LLM Worker** (`bin/30_llm_worker_rbee`):
```rust
// Accepts: Operation::Infer
// Endpoint: POST http://localhost:8080/v1/jobs
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let operation: Operation = serde_json::from_value(payload)?;
    
    match operation {
        Operation::Infer(req) => execute_infer(state, req).await,
        _ => Err(anyhow!("LLM worker only handles Infer operations")),
    }
}
```

**SD Worker** (`bin/31_sd_worker_rbee`):
```rust
// Accepts: Operation::ImageGeneration, ImageTransform, ImageInpaint
// Endpoint: POST http://localhost:8081/v1/jobs
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let operation: Operation = serde_json::from_value(payload)?;
    
    match operation {
        Operation::ImageGeneration(req) => execute_image_generation(state, req).await,
        Operation::ImageTransform(req) => execute_image_transform(state, req).await,
        Operation::ImageInpaint(req) => execute_inpaint(state, req).await,
        _ => Err(anyhow!("SD worker only handles image operations")),
    }
}
```

**Key Point:** Workers are specialized but use the same:
- HTTP endpoints (`POST /v1/jobs`, `GET /v1/jobs/{id}/stream`)
- Operation enum (from operations-contract)
- Job routing pattern

---

### At the CLI Level (rbee-keeper)

**Current Structure:**
```
rbee-keeper
├── infer                    # LLM inference (IMPLEMENTED)
├── hive                     # Hive management (IMPLEMENTED)
│   ├── start
│   ├── stop
│   ├── status
│   └── ...
├── worker                   # Worker management (IMPLEMENTED)
│   ├── spawn
│   ├── list
│   └── ...
└── image                    # Image generation (TODO TEAM-397)
    ├── generate             # Text-to-image
    ├── transform            # Image-to-image
    └── inpaint              # Inpainting
```

**Implementation Pattern (Same for All):**

```rust
// bin/00_rbee_keeper/src/handlers/infer.rs (EXISTING)
pub async fn handle_infer(
    hive_id: String,
    model: String,
    prompt: String,
    max_tokens: Option<usize>,
    queen_url: &str,
) -> Result<()> {
    // 1. Create Operation
    let operation = Operation::Infer(InferRequest {
        hive_id,
        model,
        prompt,
        max_tokens,
        // ... other fields
    });
    
    // 2. Send to Queen
    let response = client
        .post(format!("{}/v1/jobs", queen_url))
        .json(&operation)
        .send()
        .await?;
    
    // 3. Get job_id and SSE URL
    let job_response: JobResponse = response.json().await?;
    
    // 4. Stream results
    stream_sse_results(&job_response.sse_url).await?;
    
    Ok(())
}

// bin/00_rbee_keeper/src/handlers/image.rs (TODO TEAM-397)
pub async fn handle_image_generate(
    hive_id: String,
    model: String,
    prompt: String,
    steps: usize,
    width: usize,
    height: usize,
    output: PathBuf,
    queen_url: &str,
) -> Result<()> {
    // 1. Create Operation
    let operation = Operation::ImageGeneration(ImageGenerationRequest {
        hive_id,
        model,
        prompt,
        steps,
        width,
        height,
        // ... other fields
    });
    
    // 2. Send to Queen (SAME AS LLM!)
    let response = client
        .post(format!("{}/v1/jobs", queen_url))
        .json(&operation)
        .send()
        .await?;
    
    // 3. Get job_id and SSE URL (SAME AS LLM!)
    let job_response: JobResponse = response.json().await?;
    
    // 4. Stream results and save image
    stream_image_results(&job_response.sse_url, &output).await?;
    
    Ok(())
}
```

---

## Key Differences: LLM vs Image

| Aspect | LLM Operations | Image Operations |
|--------|---------------|------------------|
| **Operation Types** | `Infer` | `ImageGeneration`, `ImageTransform`, `ImageInpaint` |
| **Request Fields** | prompt, max_tokens, temperature | prompt, steps, width, height, guidance_scale |
| **Worker Type** | LLM Worker (llm-worker-rbee) | SD Worker (sd-worker-rbee) |
| **Response Type** | Text tokens (streaming) | Image (base64 PNG) + progress |
| **SSE Events** | `token`, `done` | `progress`, `complete` |
| **Model Format** | GGUF | SafeTensors |
| **Execution** | Token-by-token generation | Step-by-step diffusion |

---

## Unified API Benefits

### 1. **Same Endpoints Everywhere**
```
Queen:       POST http://localhost:7833/v1/jobs
LLM Worker:  POST http://localhost:8080/v1/jobs
SD Worker:   POST http://localhost:8081/v1/jobs
Hive:        POST http://localhost:7835/v1/jobs
```

### 2. **Same Operation Enum**
```rust
// Client sends Operation enum
// Queen routes Operation enum
// Worker receives Operation enum
// All use operations-contract crate
```

### 3. **Same Job Flow**
```
1. Client → Queen (POST /v1/jobs with Operation)
2. Queen → Worker (POST /v1/jobs with Operation)
3. Worker creates job_id
4. Client streams (GET /v1/jobs/{job_id}/stream)
5. Worker sends SSE events
```

### 4. **Same CLI Pattern**
```bash
# LLM
rbee-keeper infer --hive X --model Y --prompt Z

# Image (future)
rbee-keeper image generate --hive X --model Y --prompt Z
```

---

## Implementation Checklist for TEAM-397

### Phase 1: Add Operations to Contract

**File:** `bin/97_contracts/operations-contract/src/lib.rs`

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

Add `ImageGenerationRequest`, `ImageTransformRequest`, `ImageInpaintRequest` structs.

### Phase 2: Update Queen Router

**File:** `bin/10_queen_rbee/src/job_router.rs`

Add routing for image operations (find SD worker, forward request).

### Phase 3: Implement SD Worker Handlers

**File:** `bin/31_sd_worker_rbee/src/job_router.rs`

Uncomment and complete the handler implementations (already scaffolded).

### Phase 4: Add CLI Commands

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
enum Commands {
    Infer(InferArgs),
    Image(ImageCommands),  // NEW
    // ... rest
}

enum ImageCommands {
    Generate(ImageGenerateArgs),
    Transform(ImageTransformArgs),
    Inpaint(ImageInpaintArgs),
}
```

**File:** `bin/00_rbee_keeper/src/handlers/image.rs` (NEW)

Implement handlers following the same pattern as `infer.rs`.

---

## Example: Complete Flow

### User wants to generate an image:

```bash
# 1. User runs CLI
rbee-keeper image generate \
  --hive localhost \
  --model stable-diffusion-v1-5 \
  --prompt "A cat wearing a hat" \
  --output cat.png
```

### Behind the scenes:

```
1. rbee-keeper creates Operation::ImageGeneration
   ↓
2. POST http://localhost:7833/v1/jobs (Queen)
   ↓
3. Queen finds SD worker with model loaded
   ↓
4. POST http://localhost:8081/v1/jobs (SD Worker)
   ↓
5. SD Worker creates job, starts generation
   ↓
6. Returns { "job_id": "abc123", "sse_url": "/v1/jobs/abc123/stream" }
   ↓
7. rbee-keeper connects to SSE stream
   ↓
8. SD Worker sends progress events:
   - {"type": "progress", "step": 5, "total": 20}
   - {"type": "progress", "step": 10, "total": 20}
   - ...
   ↓
9. SD Worker sends complete event:
   - {"type": "complete", "image": "base64...", "format": "png"}
   ↓
10. rbee-keeper decodes base64, saves to cat.png
```

**Same flow as LLM inference, just different operation type and response format!**

---

## Conclusion

**Are LLM and Image operations different?**
- ✅ **Yes** - Different operation types, request fields, and response formats
- ✅ **No** - Same unified API, same endpoints, same routing pattern

**How does it look at Queen level?**
- Queen receives `Operation` enum
- Routes based on operation type and model availability
- Doesn't care if it's LLM or Image - just forwards to appropriate worker

**How does it look at CLI level?**
- Different subcommands (`infer` vs `image generate`)
- Same underlying pattern (create Operation, POST to Queen, stream results)
- User doesn't need to know about workers, ports, or routing

**The beauty of operations-contract:**
- Single source of truth for all operations
- Compile-time guarantees
- Unified API across all components
- Easy to add new operation types

**TEAM-397 just needs to:**
1. Add 3 new operation variants to the enum
2. Add 3 new request structs
3. Update Queen router (5 lines)
4. Implement SD worker handlers (already scaffolded)
5. Add CLI commands (copy infer.rs pattern)

That's it! The architecture is already correct. 🎉
