# TEAM-393 → TEAM-394 Knowledge Transfer

**From:** TEAM-393 (Generation Engine)  
**To:** TEAM-394 (HTTP Infrastructure)  
**Date:** 2025-11-03

---

## 📚 Required Reading List (Priority Order)

### 1. CRITICAL - Read These FIRST (4 hours)

**Your Team's Deliverables:**
1. `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/src/backend/request_queue.rs` (30 min)
   - **Why:** You need to understand GenerationRequest/Response types
   - **Focus:** How requests flow through the queue

2. `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/src/backend/generation_engine.rs` (30 min)
   - **Why:** You'll integrate this into AppState
   - **Focus:** How to start the engine, submit requests, handle responses

3. `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/src/backend/image_utils.rs` (15 min)
   - **Why:** You'll use image_to_base64() in HTTP responses
   - **Focus:** Base64 encoding for JSON responses

**LLM Worker Reference (MUST READ):**
4. `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/backend.rs` (45 min)
   - **Why:** Exact pattern you should follow
   - **Focus:** AppState structure, InferenceBackend trait

5. `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/server.rs` (45 min)
   - **Why:** Server lifecycle pattern
   - **Focus:** Graceful shutdown, signal handling

6. `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/routes.rs` (30 min)
   - **Why:** Router configuration pattern
   - **Focus:** Middleware stack order

### 2. IMPORTANT - Read These SECOND (2 hours)

7. `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/.windsurf/TEAM_392_FINAL_HANDOFF.md` (15 min)
   - **Why:** Understand inference pipeline APIs
   - **Focus:** What TEAM-392 delivered

8. `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/.windsurf/TEAM_393_HANDOFF.md` (15 min)
   - **Why:** Complete context on generation engine
   - **Focus:** Integration patterns, example usage

9. Axum Documentation - State Management (30 min)
   - **Why:** You need to understand Axum's State extractor
   - **URL:** https://docs.rs/axum/latest/axum/extract/struct.State.html

10. Tower-HTTP Middleware (45 min)
    - **Why:** CORS, logging, timeout configuration
    - **URL:** https://docs.rs/tower-http/latest/tower_http/

### 3. OPTIONAL - Read If Time Permits

11. `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/src/backend/inference.rs`
    - **Why:** Understand the full inference pipeline
    
12. `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/src/backend/sampling.rs`
    - **Why:** Understand SamplingConfig validation

---

## 🎯 What You MUST Know

### 1. Generation Engine Integration

**How to use it in AppState:**

```rust
use crate::backend::generation_engine::GenerationEngine;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    config: Arc<Config>,
    generation_engine: Arc<GenerationEngine>,
}

impl AppState {
    pub fn new(config: Config, pipeline: Arc<InferencePipeline>) -> Self {
        let mut engine = GenerationEngine::new(10); // Queue capacity
        engine.start(pipeline); // Start background task
        
        Self {
            config: Arc::new(config),
            generation_engine: Arc::new(engine),
        }
    }
}
```

**CRITICAL:** Call `engine.start(pipeline)` BEFORE wrapping in Arc!

### 2. Request/Response Flow

```rust
use crate::backend::request_queue::{GenerationRequest, GenerationResponse};
use tokio::sync::mpsc;

// In your HTTP handler (TEAM-395 will implement this):
async fn submit_job(State(state): State<AppState>, Json(req): Json<JobRequest>) -> Result<Json<JobResponse>> {
    // Create response channel
    let (response_tx, mut response_rx) = mpsc::channel(10);
    
    // Create generation request
    let gen_request = GenerationRequest {
        job_id: req.job_id.clone(),
        config: req.into_sampling_config(),
    };
    
    // Submit to engine
    state.generation_engine.submit(gen_request, response_tx).await?;
    
    // TEAM-395 will handle streaming responses via SSE
    Ok(Json(JobResponse { job_id: req.job_id }))
}
```

### 3. Image Encoding for HTTP

```rust
use crate::backend::image_utils::image_to_base64;

// When you get GenerationResponse::Complete { image }:
match response {
    GenerationResponse::Complete { image } => {
        let base64 = image_to_base64(&image)?;
        // Send in JSON: { "image": base64, "format": "png" }
    }
    _ => { /* handle other cases */ }
}
```

### 4. Middleware Stack Order (CRITICAL!)

```rust
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer, timeout::TimeoutLayer};

Router::new()
    .route("/health", get(health_check))
    .route("/ready", get(readiness_check))
    .layer(
        ServiceBuilder::new()
            .layer(CorsLayer::permissive())      // 1. CORS first
            .layer(TraceLayer::new_for_http())   // 2. Logging second
            .layer(TimeoutLayer::new(Duration::from_secs(300))) // 3. Timeout third
    )
    .with_state(state)
```

**Why this order?**
- CORS must be outermost (handles preflight)
- Logging should see all requests
- Timeout applies to actual processing

---

## ⚠️ Critical Lessons Learned

### 1. MPSC Channel Capacity

**Problem:** If channel fills up, `submit()` will block.

**Solution:** Use bounded channels with appropriate capacity:
- Queue capacity: 10 (for pending requests)
- Response channel: 10 (for progress events)

**Why 10?** Balances memory usage vs. throughput. SD generation is slow (~20-30s), so 10 pending requests is plenty.

### 2. Progress Callbacks Don't Block

**Implementation in generation_engine.rs:**
```rust
let _ = progress_tx.try_send(GenerationResponse::Progress { step, total });
```

**Why try_send()?** If the receiver is slow/gone, we don't want to block image generation. Progress is "best effort".

### 3. Graceful Shutdown Pattern

**You MUST implement this:**

```rust
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Ctrl+C handler");
    };
    
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
```

**Why?** Kubernetes/Docker send SIGTERM. You need to handle it gracefully.

### 4. AppState Must Be Clone

**CRITICAL:** Axum requires `State<AppState>` to be Clone.

**Solution:** Wrap everything in Arc:

```rust
#[derive(Clone)]
pub struct AppState {
    config: Arc<Config>,           // ← Arc
    engine: Arc<GenerationEngine>, // ← Arc
}
```

**Why?** Axum clones state for each request. Arc makes it cheap.

---

## 🔧 Specific Implementation Advice

### 1. Health Endpoint (/health)

**Simple version (start with this):**

```rust
pub async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}
```

**Advanced version (optional):**
- Check if generation engine is responsive
- Check if model is loaded
- Return 503 if unhealthy

### 2. Ready Endpoint (/ready)

**You need to track model loading state:**

```rust
pub struct AppState {
    config: Arc<Config>,
    engine: Arc<GenerationEngine>,
    model_loaded: Arc<AtomicBool>, // ← Add this
}

pub async fn readiness_check(State(state): State<AppState>) -> impl IntoResponse {
    if state.model_loaded.load(Ordering::Relaxed) {
        (StatusCode::OK, Json(json!({ "ready": true })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ 
            "ready": false, 
            "reason": "model loading" 
        })))
    }
}
```

### 3. CORS Configuration

**Development (permissive):**
```rust
CorsLayer::permissive()
```

**Production (restrictive):**
```rust
CorsLayer::new()
    .allow_origin("http://localhost:5174".parse::<HeaderValue>().unwrap())
    .allow_methods([Method::GET, Method::POST])
    .allow_headers([CONTENT_TYPE, AUTHORIZATION])
```

**For now:** Use permissive. TEAM-396 will tighten it.

### 4. Port Binding

**Use 0.0.0.0 for containers:**

```rust
let bind_addr = format!("0.0.0.0:{}", config.port);
let listener = TcpListener::bind(&bind_addr).await?;
```

**Why 0.0.0.0?** Allows external connections (Docker, Kubernetes).

---

## 🚨 Common Pitfalls to Avoid

### 1. ❌ DON'T: Start engine after Arc wrapping

```rust
// WRONG - engine.start() won't work after Arc::new()
let engine = Arc::new(GenerationEngine::new(10));
engine.start(pipeline); // ← Compile error!
```

```rust
// CORRECT - start before Arc
let mut engine = GenerationEngine::new(10);
engine.start(pipeline);
let engine = Arc::new(engine); // ← Now wrap it
```

### 2. ❌ DON'T: Forget to handle SIGTERM

```rust
// WRONG - only handles Ctrl+C
axum::serve(listener, app).await?;
```

```rust
// CORRECT - handles SIGTERM and Ctrl+C
axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())
    .await?;
```

### 3. ❌ DON'T: Block in handlers

```rust
// WRONG - blocks the handler
let image = generate_image_sync(); // ← Blocks!
```

```rust
// CORRECT - use async
let image = generate_image_async().await?; // ← Non-blocking
```

### 4. ❌ DON'T: Forget error handling

```rust
// WRONG - panics on error
let result = something().unwrap();
```

```rust
// CORRECT - returns error to client
let result = something()
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
```

---

## 📊 Testing Strategy

### Unit Tests (Required)

1. **AppState Clone Test:**
```rust
#[test]
fn test_appstate_clone() {
    let state = AppState::new(/* ... */);
    let cloned = state.clone();
    // Verify Arc pointers are the same
}
```

2. **Health Endpoint Test:**
```rust
#[tokio::test]
async fn test_health_endpoint() {
    let response = health_check().await;
    // Verify 200 OK
}
```

### Integration Tests (Required)

1. **Server Lifecycle:**
   - Start server
   - Make request to /health
   - Shutdown server
   - Verify graceful shutdown

2. **Concurrent Requests:**
   - Start server
   - Make 100 concurrent /health requests
   - Verify all succeed

---

## 🎯 Success Checklist

Before handing off to TEAM-395, verify:

- [ ] HTTP server starts on configured port
- [ ] GET /health returns 200 OK with JSON
- [ ] GET /ready returns 200 when model loaded, 503 otherwise
- [ ] CORS headers present on all responses
- [ ] Server handles SIGTERM gracefully
- [ ] Server handles Ctrl+C gracefully
- [ ] AppState is Clone
- [ ] GenerationEngine integrated into AppState
- [ ] All unit tests pass
- [ ] Integration test passes (server lifecycle)
- [ ] Can handle 100 concurrent /health requests
- [ ] Clean compilation (0 warnings in your code)

---

## 📞 Questions to Ask

If you're stuck, ask yourself:

1. **"Does AppState have everything TEAM-395 needs?"**
   - GenerationEngine? ✓
   - Config? ✓
   - Model loaded status? ✓

2. **"Can the server shut down gracefully?"**
   - Handles SIGTERM? ✓
   - Waits for in-flight requests? ✓

3. **"Is the middleware stack correct?"**
   - CORS → Logging → Timeout? ✓

4. **"Are all Arc wrappers in place?"**
   - AppState Clone works? ✓

---

## 🎁 What You're Giving TEAM-395

**They will receive from you:**

1. **AppState** with:
   - GenerationEngine (started and ready)
   - Config access
   - Model loading status

2. **HTTP Server** that:
   - Starts on configured port
   - Handles graceful shutdown
   - Has middleware stack configured

3. **Router** with:
   - /health endpoint working
   - /ready endpoint working
   - Extension points for /v1/jobs routes

4. **Documentation** showing:
   - How to add new routes
   - How to access AppState
   - How to submit to GenerationEngine

---

## 🚀 Final Advice

1. **Start Simple:** Get /health working first, then add complexity
2. **Test Early:** Write tests as you go, not at the end
3. **Follow LLM Worker:** It's your blueprint - copy the patterns
4. **Ask Questions:** If something is unclear, check the LLM worker code
5. **Document Decisions:** Add comments explaining non-obvious choices

**You're building the foundation. Make it solid!** 🏗️

---

**TEAM-393 signing off. Good luck, TEAM-394!** 🚀
