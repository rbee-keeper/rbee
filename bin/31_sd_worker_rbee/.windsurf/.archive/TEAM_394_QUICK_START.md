# TEAM-394 Quick Start Guide

## 🚨 READ THIS FIRST

**Before writing any code:**

1. **Read Knowledge Transfer** (1.5 hours)
   - `.windsurf/TEAM_393_TO_394_KNOWLEDGE_TRANSFER.md`
   - Contains everything you need to know

2. **Study These Files** (1 hour)
   - `src/backend/request_queue.rs`
   - `src/backend/generation_engine.rs`
   - `src/backend/image_utils.rs`

3. **Reference LLM Worker** (2 hours)
   - `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/`
   - Copy the patterns, don't reinvent

---

## 📋 Your Checklist

### Day 1: AppState (8 hours)
- [ ] Create `src/http/backend.rs`
- [ ] Define AppState with GenerationEngine
- [ ] **CRITICAL:** Start engine BEFORE Arc wrapping
- [ ] Add `is_ready()` method
- [ ] Write Clone test

### Day 2: HTTP Server (8 hours)
- [ ] Create `src/http/server.rs`
- [ ] Implement server startup
- [ ] Add graceful shutdown (SIGTERM + Ctrl+C)
- [ ] Test server lifecycle

### Day 3: Routes & Middleware (8 hours)
- [ ] Create `src/http/routes.rs`
- [ ] Configure router
- [ ] Add middleware: CORS → Logging → Timeout
- [ ] Test middleware chain

### Day 4: Health & Ready (8 hours)
- [ ] Create `src/http/health.rs`
- [ ] Implement GET /health
- [ ] Create `src/http/ready.rs`
- [ ] Implement GET /ready
- [ ] Write endpoint tests

### Day 5: Integration & Testing (8 hours)
- [ ] Integration test: server lifecycle
- [ ] Integration test: 100 concurrent requests
- [ ] Verify all success criteria
- [ ] Write handoff document

---

## 🎯 Critical Code Snippets

### AppState (Copy This!)

```rust
use crate::backend::generation_engine::GenerationEngine;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
pub struct AppState {
    config: Arc<Config>,
    generation_engine: Arc<GenerationEngine>,
    model_loaded: Arc<AtomicBool>,
}

impl AppState {
    pub fn new(config: Config, pipeline: Arc<InferencePipeline>) -> Self {
        // CRITICAL: Start BEFORE Arc!
        let mut engine = GenerationEngine::new(10);
        engine.start(pipeline);
        
        Self {
            config: Arc::new(config),
            generation_engine: Arc::new(engine),
            model_loaded: Arc::new(AtomicBool::new(true)),
        }
    }
    
    pub fn is_ready(&self) -> bool {
        self.model_loaded.load(Ordering::Relaxed)
    }
}
```

### Graceful Shutdown (Copy This!)

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

### Middleware Stack (Copy This!)

```rust
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer, timeout::TimeoutLayer};

Router::new()
    .route("/health", get(health_check))
    .route("/ready", get(readiness_check))
    .layer(
        ServiceBuilder::new()
            .layer(CorsLayer::permissive())
            .layer(TraceLayer::new_for_http())
            .layer(TimeoutLayer::new(Duration::from_secs(300)))
    )
    .with_state(state)
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T: Start engine after Arc
```rust
let engine = Arc::new(GenerationEngine::new(10));
engine.start(pipeline); // ← Won't compile!
```

### ✅ DO: Start before Arc
```rust
let mut engine = GenerationEngine::new(10);
engine.start(pipeline);
let engine = Arc::new(engine);
```

### ❌ DON'T: Forget SIGTERM
```rust
axum::serve(listener, app).await?; // ← Only handles Ctrl+C
```

### ✅ DO: Handle both signals
```rust
axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())
    .await?;
```

### ❌ DON'T: Wrong middleware order
```rust
.layer(TimeoutLayer::new(...))
.layer(TraceLayer::new_for_http())
.layer(CorsLayer::permissive())
```

### ✅ DO: Correct order
```rust
.layer(CorsLayer::permissive())      // 1. CORS first
.layer(TraceLayer::new_for_http())   // 2. Logging second
.layer(TimeoutLayer::new(...))       // 3. Timeout third
```

---

## ✅ Success Criteria

Before handing off to TEAM-395:

- [ ] Server starts on port 8080
- [ ] GET /health returns 200 OK
- [ ] GET /ready returns 200 when ready, 503 when not
- [ ] CORS headers on all responses
- [ ] Server handles SIGTERM gracefully
- [ ] Server handles Ctrl+C gracefully
- [ ] Can handle 100 concurrent /health requests
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Clean compilation (0 warnings in your code)

---

## 📞 If You Get Stuck

1. **Check Knowledge Transfer:** `.windsurf/TEAM_393_TO_394_KNOWLEDGE_TRANSFER.md`
2. **Check LLM Worker:** Copy the pattern from there
3. **Check TEAM-393 Code:** See how they did it
4. **Ask Questions:** Document what's unclear

---

## 🎁 What You're Giving TEAM-395

- ✅ AppState with GenerationEngine ready
- ✅ HTTP server with graceful shutdown
- ✅ Router with middleware configured
- ✅ /health and /ready endpoints working
- ✅ Extension points for /v1/jobs routes

---

**Start with the knowledge transfer document. It will save you hours!** 📚

**Good luck, TEAM-394!** 🚀
