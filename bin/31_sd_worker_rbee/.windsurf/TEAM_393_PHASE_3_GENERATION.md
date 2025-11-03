# TEAM-393: Phase 3 - Generation Engine & Queue

**Team:** TEAM-393  
**Phase:** 3 - Generation Engine  
**Duration:** 40 hours  
**Dependencies:** TEAM-392 (needs inference pipeline)  
**Parallel Work:** None (sequential after TEAM-392)

---

## 🎯 Mission

Build the asynchronous generation engine with request queue, progress reporting, and image processing utilities. Enable background generation with real-time progress updates.

---

## 📦 What You're Building

### Files to Create (3 files, ~500 LOC total)

1. **`src/backend/generation_engine.rs`** (~250 LOC)
   - Background generation task
   - Request queue processing
   - Progress reporting
   - Cancellation support

2. **`src/backend/request_queue.rs`** (~150 LOC)
   - MPSC channel-based queue
   - Request/response types
   - Queue management

3. **`src/backend/image_utils.rs`** (~100 LOC)
   - Base64 encoding/decoding
   - Image resizing
   - Mask processing
   - Tensor ↔ Image conversion

---

## 📋 Task Breakdown

### Day 1: Study & Design (8 hours)

**Morning (4 hours):**
- [ ] Study TEAM-392's inference pipeline (1 hour)
- [ ] Read `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/backend/generation_engine.rs` (2 hours)
- [ ] Read `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/backend/request_queue.rs` (1 hour)

**Afternoon (4 hours):**
- [ ] Design request/response types (2 hours)
- [ ] Design progress reporting mechanism (1 hour)
- [ ] Design cancellation strategy (1 hour)

**Output:** Design document, type definitions

---

### Day 2: Request Queue (8 hours)

**Morning (4 hours):**
- [ ] Create `src/backend/request_queue.rs` (30 min)
- [ ] Define `GenerationRequest` struct (1 hour)
- [ ] Define `GenerationResponse` enum (1 hour)
- [ ] Implement MPSC channel queue (1.5 hours)

**Afternoon (4 hours):**
- [ ] Add request submission (1 hour)
- [ ] Add response handling (1 hour)
- [ ] Add queue shutdown (1 hour)
- [ ] Write unit tests (1 hour)

**Output:** Working request queue, tests passing

---

### Day 3: Generation Engine Core (8 hours)

**Morning (4 hours):**
- [ ] Create `src/backend/generation_engine.rs` (30 min)
- [ ] Implement `GenerationEngine` struct (1 hour)
- [ ] Implement background task loop (1.5 hours)
- [ ] Add request processing (1 hour)

**Afternoon (4 hours):**
- [ ] Integrate with TEAM-392's inference (2 hours)
- [ ] Add error handling (1 hour)
- [ ] Add graceful shutdown (1 hour)

**Output:** Engine processes requests in background

---

### Day 4: Progress Reporting (8 hours)

**Morning (4 hours):**
- [ ] Design progress event types (1 hour)
- [ ] Implement progress callbacks (2 hours)
- [ ] Add step-by-step progress (1 hour)

**Afternoon (4 hours):**
- [ ] Add completion events (1 hour)
- [ ] Add error events (1 hour)
- [ ] Test progress flow (2 hours)

**Output:** Real-time progress reporting working

---

### Day 5: Image Utilities (8 hours)

**Morning (4 hours):**
- [ ] Create `src/backend/image_utils.rs` (30 min)
- [ ] Implement base64 encoding (1 hour)
- [ ] Implement base64 decoding (1 hour)
- [ ] Implement image resizing (1.5 hours)

**Afternoon (4 hours):**
- [ ] Implement mask processing (2 hours)
- [ ] Implement tensor ↔ image conversion (1 hour)
- [ ] Write unit tests (1 hour)

**Output:** Image utilities complete, tests passing

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] Request queue accepts generation requests
- [ ] Generation engine processes requests in background
- [ ] Progress events fire for each diffusion step
- [ ] Completion events include base64-encoded image
- [ ] Cancellation works (can stop mid-generation)
- [ ] Multiple requests can be queued
- [ ] Image utilities handle all formats (PNG, JPEG)
- [ ] Base64 encoding/decoding works correctly
- [ ] All unit tests passing
- [ ] Clean compilation (0 warnings)
- [ ] Can generate 5 images sequentially without issues

---

## 🧪 Testing Requirements

### Unit Tests (Required)

1. **Request Queue Tests** (`src/backend/request_queue.rs`)
   - Test request submission
   - Test response retrieval
   - Test queue shutdown
   - Test concurrent access

2. **Generation Engine Tests** (`src/backend/generation_engine.rs`)
   - Test request processing
   - Test progress events
   - Test completion events
   - Test error handling
   - Test cancellation

3. **Image Utils Tests** (`src/backend/image_utils.rs`)
   - Test base64 encoding
   - Test base64 decoding
   - Test image resizing
   - Test mask processing

### Integration Test

```rust
#[tokio::test]
async fn test_generation_engine_end_to_end() {
    let engine = GenerationEngine::new(backend).await;
    
    let request = GenerationRequest {
        prompt: "a photo of a dog".to_string(),
        steps: 20,
        seed: Some(42),
    };
    
    let (tx, mut rx) = mpsc::channel(10);
    engine.submit(request, tx).await.unwrap();
    
    let mut progress_count = 0;
    while let Some(event) = rx.recv().await {
        match event {
            GenerationEvent::Progress { step, total } => {
                progress_count += 1;
                assert!(step <= total);
            }
            GenerationEvent::Complete { image_base64 } => {
                assert!(!image_base64.is_empty());
                break;
            }
            GenerationEvent::Error { .. } => panic!("Unexpected error"),
        }
    }
    
    assert_eq!(progress_count, 20); // One per step
}
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **LLM Worker Generation Engine** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/backend/generation_engine.rs`
   - Focus: Background task pattern, progress reporting

2. **LLM Worker Request Queue** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/backend/request_queue.rs`
   - Focus: MPSC channel usage, request/response types

3. **TEAM-392's Inference** (Your Dependency)
   - Path: `src/backend/inference.rs`
   - Usage: `CandleSDBackend::text_to_image()`

---

## 🔧 Implementation Notes

### Request Queue Pattern

```rust
pub struct GenerationRequest {
    pub job_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub width: usize,
    pub height: usize,
}

pub enum GenerationResponse {
    Progress { step: usize, total: usize },
    Complete { image_base64: String },
    Error { message: String },
}

pub struct RequestQueue {
    tx: mpsc::Sender<(GenerationRequest, mpsc::Sender<GenerationResponse>)>,
}
```

### Generation Engine Pattern

```rust
pub struct GenerationEngine {
    backend: Arc<CandleSDBackend>,
    queue: RequestQueue,
    shutdown: Arc<AtomicBool>,
}

impl GenerationEngine {
    pub async fn start(&self) {
        tokio::spawn(async move {
            while !self.shutdown.load(Ordering::Relaxed) {
                if let Some((request, response_tx)) = self.queue.recv().await {
                    self.process_request(request, response_tx).await;
                }
            }
        });
    }
    
    async fn process_request(&self, request: GenerationRequest, tx: mpsc::Sender<GenerationResponse>) {
        // Progress callback
        let progress_fn = |step, total| {
            tx.send(GenerationResponse::Progress { step, total }).await.ok();
        };
        
        // Generate image
        match self.backend.text_to_image_with_progress(request, progress_fn).await {
            Ok(image) => {
                let base64 = image_to_base64(&image);
                tx.send(GenerationResponse::Complete { image_base64: base64 }).await.ok();
            }
            Err(e) => {
                tx.send(GenerationResponse::Error { message: e.to_string() }).await.ok();
            }
        }
    }
}
```

### Image Utilities

```rust
pub fn image_to_base64(image: &DynamicImage) -> String {
    let mut buffer = Vec::new();
    image.write_to(&mut Cursor::new(&mut buffer), ImageFormat::Png).unwrap();
    base64::encode(&buffer)
}

pub fn base64_to_image(base64: &str) -> Result<DynamicImage> {
    let bytes = base64::decode(base64)?;
    let image = image::load_from_memory(&bytes)?;
    Ok(image)
}
```

---

## 🚨 Common Pitfalls

1. **Channel Deadlocks**
   - Problem: Sender waiting on full channel
   - Solution: Use bounded channels with appropriate capacity

2. **Progress Callback Blocking**
   - Problem: Slow progress callback blocks generation
   - Solution: Use `try_send()` or separate task

3. **Memory Leaks**
   - Problem: Completed requests not cleaned up
   - Solution: Drop response channels after completion

4. **Cancellation Race Conditions**
   - Problem: Cancel arrives after completion
   - Solution: Check cancellation flag before each step

---

## 🎯 Handoff to TEAM-395

**What TEAM-395 needs from you:**

### Files Created
- `src/backend/generation_engine.rs` - Background generation
- `src/backend/request_queue.rs` - Request queue
- `src/backend/image_utils.rs` - Image utilities

### APIs Exposed

```rust
// Generation engine
pub struct GenerationEngine {
    pub async fn submit(&self, request: GenerationRequest, response_tx: mpsc::Sender<GenerationResponse>) -> Result<()>;
    pub async fn shutdown(&self);
}

// Request/Response types
pub struct GenerationRequest { /* ... */ }
pub enum GenerationResponse {
    Progress { step: usize, total: usize },
    Complete { image_base64: String },
    Error { message: String },
}

// Image utilities
pub fn image_to_base64(image: &DynamicImage) -> String;
pub fn base64_to_image(base64: &str) -> Result<DynamicImage>;
```

### What Works
- Async generation in background
- Real-time progress reporting
- Base64 image encoding
- Multiple queued requests
- Graceful shutdown

### What TEAM-395 Will Add
- HTTP endpoints for job submission
- SSE streaming for progress
- Job registry integration

---

## 📊 Progress Tracking

Track your progress:

- [ ] Day 1: Design complete
- [ ] Day 2: Request queue working
- [ ] Day 3: Engine processing requests
- [ ] Day 4: Progress reporting working
- [ ] Day 5: Image utilities complete, ready for handoff

---

**TEAM-393: You're building the async engine that makes everything smooth. Make it rock-solid.** 🚀
