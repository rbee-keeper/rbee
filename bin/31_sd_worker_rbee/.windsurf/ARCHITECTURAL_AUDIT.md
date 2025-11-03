# SD Worker Architectural Audit

**Date:** 2025-11-03  
**Auditor:** Architecture Review  
**Status:** 🚨 MULTIPLE DEVIATIONS FOUND

---

## Executive Summary

Previous teams (390-394) implemented patterns that DEVIATE from the established LLM worker architecture. While the code compiles and might work, it creates inconsistency across workers and violates established patterns.

**Critical Finding:** The `RequestQueue` and `GenerationEngine` architecture is fundamentally different from the LLM worker.

---

## Issue 1: RequestQueue Ownership Pattern

### ❌ SD Worker (TEAM-393) - WRONG

**File:** `src/backend/request_queue.rs`

```rust
pub struct RequestQueue {
    tx: mpsc::Sender<QueueItem>,
    rx: Mutex<Option<mpsc::Receiver<QueueItem>>>,  // OWNS receiver!
}

impl RequestQueue {
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self {
            tx,
            rx: Mutex::new(Some(rx)),  // Wraps in Mutex
        }
    }
    
    pub fn take_receiver(&self) -> Option<mpsc::Receiver<QueueItem>> {
        self.rx.lock().unwrap().take()  // Complex interior mutability
    }
}
```

**Problems:**
1. RequestQueue OWNS the receiver (anti-pattern)
2. Requires Mutex for interior mutability
3. Bounded channel with capacity limits
4. `take_receiver()` can panic on lock poisoning
5. No clear ownership model

### ✅ LLM Worker - CORRECT

**File:** `bin/30_llm_worker_rbee/src/backend/request_queue.rs`

```rust
pub struct RequestQueue {
    tx: mpsc::UnboundedSender<GenerationRequest>,  // Just sender!
}

impl RequestQueue {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<GenerationRequest>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)  // Returns BOTH, doesn't own receiver
    }
    
    pub fn add_request(&self, request: GenerationRequest) -> Result<(), String> {
        self.tx.send(request)
            .map_err(|e| format!("Queue send failed: {e}"))
    }
}
```

**Benefits:**
1. Clear separation: sender in RequestQueue, receiver given to engine
2. No Mutex needed (just sender which is Clone + Send)
3. Unbounded channel (no artificial limits)
4. Simple, idiomatic pattern
5. Clear ownership transfer

---

## Issue 2: GenerationEngine Construction

### ❌ SD Worker (TEAM-393) - WRONG

**File:** `src/backend/generation_engine.rs`

```rust
pub struct GenerationEngine {
    queue: Arc<RequestQueue>,  // OWNS queue!
    shutdown: Arc<AtomicBool>,
}

impl GenerationEngine {
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            queue: Arc::new(RequestQueue::new(queue_capacity)),  // Creates queue!
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }
    
    pub fn start(&mut self, pipeline: Arc<InferencePipeline>) {
        let mut rx = self.queue.take_receiver()  // Takes from queue
            .expect("Receiver already taken");
        // ...
    }
    
    pub async fn submit(
        &self,
        request: GenerationRequest,
        response_tx: mpsc::Sender<GenerationResponse>,  // Passed separately!
    ) -> Result<()> {
        self.queue.submit(request, response_tx).await
    }
}
```

**Problems:**
1. GenerationEngine creates RequestQueue (tight coupling)
2. `start()` requires `&mut self` (ownership issues)
3. `response_tx` passed separately (not in request)
4. Complex lifetime management
5. Hard to test (can't mock queue easily)

### ✅ LLM Worker - CORRECT

**File:** `bin/30_llm_worker_rbee/src/backend/generation_engine.rs`

```rust
pub struct GenerationEngine {
    backend: Arc<Mutex<CandleInferenceBackend>>,
    request_rx: mpsc::UnboundedReceiver<GenerationRequest>,  // Receives RX!
}

impl GenerationEngine {
    pub fn new(
        backend: Arc<Mutex<CandleInferenceBackend>>,
        request_rx: mpsc::UnboundedReceiver<GenerationRequest>,  // Passed in!
    ) -> Self {
        Self { backend, request_rx }
    }
    
    pub fn start(mut self) {  // Consumes self!
        tokio::task::spawn_blocking(move || {
            loop {
                let request = rt.block_on(self.request_rx.recv());
                // response_tx is IN the request!
                Self::generate_streaming(
                    &mut backend,
                    &request.prompt,
                    &request.config,
                    request.response_tx,  // Part of request!
                )
            }
        });
    }
}
```

**Benefits:**
1. Clear dependency injection (rx passed in)
2. `start()` consumes self (clean ownership)
3. `response_tx` is part of GenerationRequest (cohesive)
4. Easy to test (pass mock receiver)
5. No shared state management

---

## Issue 3: Request Structure

### ❌ SD Worker - WRONG

```rust
pub struct GenerationRequest {
    pub job_id: String,
    pub config: SamplingConfig,
    // No response_tx!
}

// response_tx passed separately to submit()
```

### ✅ LLM Worker - CORRECT

```rust
pub struct GenerationRequest {
    pub request_id: String,
    pub prompt: String,
    pub config: SamplingConfig,
    pub response_tx: mpsc::UnboundedSender<TokenResponse>,  // Included!
}
```

**Why this matters:** The request is self-contained. You can serialize it, queue it, or pass it around without needing additional context.

---

## Issue 4: spawn_blocking vs tokio::spawn

### ❌ SD Worker - WRONG

```rust
tokio::spawn(async move {  // Regular async spawn
    // Image generation (CPU-intensive!) runs here
    pipeline.text_to_image(...)
})
```

**Problem:** Image generation is CPU-intensive. Running in regular `tokio::spawn` blocks the async runtime.

### ✅ LLM Worker - CORRECT

```rust
tokio::task::spawn_blocking(move || {  // Blocking task
    // CPU-intensive work runs here
    loop {
        let request = rt.block_on(self.request_rx.recv());
        backend.generate(...)  // CPU work
    }
});
```

**Why this matters:** `spawn_blocking` moves CPU work to a separate thread pool, preventing it from blocking async I/O operations.

---

## Issue 5: Channel Types

### SD Worker

```rust
mpsc::channel(capacity)  // Bounded, requires capacity argument
```

### LLM Worker

```rust
mpsc::unbounded_channel()  // Unbounded, simpler
```

**Trade-offs:**
- **Bounded:** Provides backpressure, prevents unbounded growth
- **Unbounded:** Simpler, no capacity tuning needed

**LLM worker's choice is better for workers:** The queue is internal and short-lived (requests are processed quickly). Bounded channels add complexity for minimal benefit.

---

## Recommended Fixes

### Priority 1: RequestQueue Pattern (BREAKING CHANGE)

**Delete:** Current `RequestQueue` implementation  
**Replace with:**

```rust
// src/backend/request_queue.rs
pub struct GenerationRequest {
    pub request_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub config: SamplingConfig,
    pub response_tx: mpsc::UnboundedSender<GenerationResponse>,  // Add this!
}

pub struct RequestQueue {
    tx: mpsc::UnboundedSender<GenerationRequest>,
}

impl RequestQueue {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<GenerationRequest>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }
    
    pub fn add_request(&self, request: GenerationRequest) -> Result<(), String> {
        self.tx.send(request)
            .map_err(|e| format!("Queue send failed: {e}"))
    }
}
```

### Priority 2: GenerationEngine Pattern (BREAKING CHANGE)

```rust
// src/backend/generation_engine.rs
pub struct GenerationEngine {
    pipeline: Arc<InferencePipeline>,
    request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
}

impl GenerationEngine {
    pub fn new(
        pipeline: Arc<InferencePipeline>,
        request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
    ) -> Self {
        Self { pipeline, request_rx }
    }
    
    pub fn start(mut self) {
        tokio::task::spawn_blocking(move || {  // Use spawn_blocking!
            let rt = tokio::runtime::Handle::current();
            
            loop {
                let request = match rt.block_on(self.request_rx.recv()) {
                    Some(req) => req,
                    None => break,
                };
                
                // response_tx is in the request!
                Self::process_request(&self.pipeline, request);
            }
        });
    }
    
    fn process_request(
        pipeline: &InferencePipeline,
        request: GenerationRequest,
    ) {
        // Generate image
        match pipeline.text_to_image(...) {
            Ok(image) => {
                let _ = request.response_tx.send(GenerationResponse::Complete { image });
            }
            Err(e) => {
                let _ = request.response_tx.send(GenerationResponse::Error { 
                    message: e.to_string() 
                });
            }
        }
    }
}
```

### Priority 3: Main Setup

```rust
// src/main.rs
let pipeline = Arc::new(load_pipeline()?);

// Create queue and engine
let (request_queue, request_rx) = RequestQueue::new();
let engine = GenerationEngine::new(Arc::clone(&pipeline), request_rx);
engine.start();

// Create AppState for HTTP handlers
let app_state = AppState::new(request_queue);
```

---

## Code Size Impact

**Current SD Worker Pattern:**
- `request_queue.rs`: 110 LOC (with Mutex complexity)
- `generation_engine.rs`: 146 LOC (with Arc<RequestQueue>)
- **Total:** 256 LOC

**LLM Worker Pattern:**
- `request_queue.rs`: 77 LOC (simpler)
- `generation_engine.rs`: 215 LOC (more complete)
- **Total:** 292 LOC (+36 LOC but cleaner)

**Net:** Slightly more code but MUCH cleaner architecture.

---

## Migration Strategy

### Option A: Fix Now (Recommended)

**Impact:** Breaking changes to TEAM-393's work  
**Time:** 2-3 hours  
**Benefit:** Clean architecture from the start

### Option B: Technical Debt

**Impact:** Accept the deviation, document it  
**Risk:** Future confusion, harder to maintain  
**Cost:** Will need refactoring eventually

---

## Other Findings

### ✅ Good Decisions (Keep These)

1. **Image utilities module** - Good separation of concerns
2. **Model loader** - Clear abstraction
3. **CLIP/VAE/Scheduler modules** - Well-organized
4. **SamplingConfig** - Proper configuration struct

### 🟡 Minor Issues (Lower Priority)

1. **No `common/` directory** - LLM worker has this for shared types
2. **No `device.rs`** - Relies on shared-worker-rbee instead (acceptable)
3. **Error types** - Could be more specific

---

## Comparison Table

| Aspect | LLM Worker | SD Worker | Verdict |
|--------|-----------|-----------|---------|
| RequestQueue ownership | Queue doesn't own RX ✅ | Queue owns RX in Mutex ❌ | LLM correct |
| Engine construction | RX passed in ✅ | Creates queue internally ❌ | LLM correct |
| response_tx location | In GenerationRequest ✅ | Passed separately ❌ | LLM correct |
| Channel type | Unbounded ✅ | Bounded 🟡 | LLM simpler |
| Execution | spawn_blocking ✅ | tokio::spawn ❌ | LLM correct |
| Clone-ability | RequestQueue is Clone ✅ | Complex Arc pattern ❌ | LLM correct |

**Score: LLM Pattern wins 5/6**

---

## Recommendation

**REFACTOR to match LLM worker pattern.** The current SD worker architecture:
1. Deviates from established patterns
2. Is more complex than necessary
3. Uses anti-patterns (queue owning receiver, Mutex for interior mutability)
4. Harder to test and maintain

**Cost:** 2-3 hours of refactoring  
**Benefit:** Consistent, maintainable codebase  
**Risk of not fixing:** Technical debt accumulation

---

## Verdict

🚨 **ARCHITECTURAL VIOLATIONS FOUND**

**Teams responsible:** TEAM-393 (GenerationEngine, RequestQueue patterns)

**Recommendation:** Refactor before TEAM-396 builds on top of this foundation.

**Note:** TEAM-394's HTTP infrastructure is fine - that's not affected by these backend issues.
