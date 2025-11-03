# TEAM-394: Phase 4 - HTTP Infrastructure

**Team:** TEAM-394  
**Phase:** 4 - HTTP Infrastructure  
**Duration:** 40 hours  
**Dependencies:** None (can work parallel to TEAM-392/393)  
**Parallel Work:** ✅ Can work independently

---

## 🚨 CRITICAL: Read TEAM-393's Knowledge Transfer FIRST!

**Before starting, read this file:**
📄 `.windsurf/TEAM_393_TO_394_KNOWLEDGE_TRANSFER.md`

**It contains:**
- ✅ Complete reading list (prioritized)
- ✅ Integration patterns for GenerationEngine
- ✅ Critical lessons learned
- ✅ Common pitfalls to avoid
- ✅ Specific implementation advice
- ✅ Testing strategy
- ✅ Success checklist

**Estimated reading time:** 1.5 hours  
**This will save you 10+ hours of debugging!**

---

## 🎯 Mission

Build the HTTP server infrastructure: AppState, route configuration, health/ready endpoints, and CORS middleware. Create the foundation for all HTTP endpoints.

---

## 📦 What You're Building

### Files to Create (5 files, ~400 LOC total)

1. **`src/http/backend.rs`** (~100 LOC)
   - AppState definition
   - InferenceBackend trait
   - Shared state management

2. **`src/http/server.rs`** (~100 LOC)
   - HTTP server lifecycle
   - Graceful shutdown
   - Port binding

3. **`src/http/routes.rs`** (~80 LOC)
   - Axum router configuration
   - Route registration
   - Middleware stack

4. **`src/http/health.rs`** (~60 LOC)
   - Health check endpoint
   - Liveness probe

5. **`src/http/ready.rs`** (~60 LOC)
   - Readiness check endpoint
   - Model loading status

---

## 📋 Task Breakdown

### Day 1: Study & Design (8 hours)

**Morning (4 hours):**
- [ ] Read `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/` (2 hours)
- [ ] Study Axum framework basics (1 hour)
- [ ] Study tower-http middleware (1 hour)

**Afternoon (4 hours):**
- [ ] Design AppState structure (1 hour)
- [ ] Design route hierarchy (1 hour)
- [ ] Design middleware stack (1 hour)
- [ ] Create module structure (1 hour)

**Output:** Design document, module skeleton

---

### Day 2: AppState & Backend Trait (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/backend.rs` (30 min)
- [ ] Define `AppState` struct (1 hour)
- [ ] Define `InferenceBackend` trait (1.5 hours)
- [ ] Add Arc wrapping for thread safety (1 hour)

**Afternoon (4 hours):**
- [ ] Implement Clone for AppState (1 hour)
- [ ] Add configuration access (1 hour)
- [ ] Add backend access (1 hour)
- [ ] Write unit tests (1 hour)

**Output:** AppState and trait defined, tests passing

---

### Day 3: HTTP Server (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/server.rs` (30 min)
- [ ] Implement server startup (1.5 hours)
- [ ] Add port binding logic (1 hour)
- [ ] Add address resolution (1 hour)

**Afternoon (4 hours):**
- [ ] Implement graceful shutdown (2 hours)
- [ ] Add signal handling (SIGTERM, SIGINT) (1 hour)
- [ ] Write integration tests (1 hour)

**Output:** HTTP server starts and stops cleanly

---

### Day 4: Routes & Middleware (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/routes.rs` (30 min)
- [ ] Define route structure (1 hour)
- [ ] Add CORS middleware (1 hour)
- [ ] Add logging middleware (1.5 hours)

**Afternoon (4 hours):**
- [ ] Add request ID middleware (1 hour)
- [ ] Add timeout middleware (1 hour)
- [ ] Configure middleware stack (1 hour)
- [ ] Test middleware chain (1 hour)

**Output:** Router with middleware stack working

---

### Day 5: Health & Ready Endpoints (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/health.rs` (30 min)
- [ ] Implement GET /health endpoint (1 hour)
- [ ] Add health check logic (1.5 hours)
- [ ] Write tests (1 hour)

**Afternoon (4 hours):**
- [ ] Create `src/http/ready.rs` (30 min)
- [ ] Implement GET /ready endpoint (1 hour)
- [ ] Add readiness check logic (1.5 hours)
- [ ] Write tests (1 hour)

**Output:** Health and ready endpoints working

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] HTTP server starts on specified port
- [ ] GET /health returns 200 OK
- [ ] GET /ready returns 200 when model loaded, 503 otherwise
- [ ] CORS headers present on all responses
- [ ] Request logging works
- [ ] Graceful shutdown works (waits for in-flight requests)
- [ ] AppState accessible in all handlers
- [ ] All unit tests passing
- [ ] Clean compilation (0 warnings)
- [ ] Can handle 100 concurrent requests without issues

---

## 🧪 Testing Requirements

### Unit Tests (Required)

1. **AppState Tests** (`src/http/backend.rs`)
   - Test Clone implementation
   - Test thread safety (Arc)
   - Test configuration access

2. **Server Tests** (`src/http/server.rs`)
   - Test server startup
   - Test port binding
   - Test graceful shutdown

3. **Health Tests** (`src/http/health.rs`)
   - Test health endpoint
   - Test response format

4. **Ready Tests** (`src/http/ready.rs`)
   - Test ready endpoint
   - Test model loading status

### Integration Test

```rust
#[tokio::test]
async fn test_http_server_lifecycle() {
    let config = Config::default();
    let backend = MockBackend::new();
    let state = AppState::new(config, backend);
    
    let server = HttpServer::new(state, "127.0.0.1:0").await.unwrap();
    let addr = server.local_addr();
    
    tokio::spawn(async move {
        server.serve().await.unwrap();
    });
    
    // Test health endpoint
    let response = reqwest::get(format!("http://{}/health", addr))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    
    // Test ready endpoint
    let response = reqwest::get(format!("http://{}/ready", addr))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **LLM Worker HTTP** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/`
   - Focus: AppState pattern, server lifecycle

2. **Axum Documentation**
   - Router configuration
   - State management
   - Middleware

3. **Tower-HTTP Middleware**
   - CORS
   - Logging
   - Timeout

---

## 🔧 Implementation Notes

### AppState Pattern (UPDATED FOR TEAM-393 INTEGRATION)

```rust
use crate::backend::generation_engine::GenerationEngine;
use crate::backend::inference::InferencePipeline;
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
        // CRITICAL: Start engine BEFORE wrapping in Arc!
        let mut engine = GenerationEngine::new(10); // Queue capacity
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

**Key Changes from Original Plan:**
- ✅ Uses TEAM-393's GenerationEngine (not a trait)
- ✅ Starts engine before Arc wrapping
- ✅ Tracks model loading with AtomicBool
- ✅ Simpler than original InferenceBackend trait approach

### Server Pattern

```rust
pub struct HttpServer {
    app: Router,
    addr: SocketAddr,
}

impl HttpServer {
    pub async fn new(state: AppState, bind_addr: &str) -> Result<Self> {
        let router = create_router(state);
        let addr = bind_addr.parse()?;
        Ok(Self { app: router, addr })
    }
    
    pub async fn serve(self) -> Result<()> {
        let listener = TcpListener::bind(self.addr).await?;
        
        axum::serve(listener, self.app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        
        Ok(())
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };
    
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
```

### Routes Pattern

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health_check))
        .route("/ready", get(ready::readiness_check))
        // More routes added by TEAM-395
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .layer(TraceLayer::new_for_http())
                .layer(TimeoutLayer::new(Duration::from_secs(300)))
        )
        .with_state(state)
}
```

### Health Endpoint

```rust
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    match state.backend.health_check().await {
        Ok(_) => (StatusCode::OK, Json(json!({ "status": "healthy" }))),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "status": "unhealthy", "error": e.to_string() }))),
    }
}
```

### Ready Endpoint (UPDATED)

```rust
pub async fn readiness_check(State(state): State<AppState>) -> impl IntoResponse {
    if state.is_ready() {
        (StatusCode::OK, Json(json!({ 
            "ready": true,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ 
            "ready": false, 
            "reason": "model loading",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }
}
```

**Note:** Uses `state.is_ready()` method from AppState (checks AtomicBool)

---

## 🚨 Common Pitfalls

1. **State Cloning Issues**
   - Problem: AppState not Clone
   - Solution: Wrap everything in Arc

2. **Middleware Order**
   - Problem: Wrong middleware order breaks functionality
   - Solution: CORS → Logging → Timeout → Routes

3. **Graceful Shutdown**
   - Problem: Server kills in-flight requests
   - Solution: Use `with_graceful_shutdown()`

4. **Port Binding**
   - Problem: Port already in use
   - Solution: Proper error handling, configurable port

---

## 🎯 Handoff to TEAM-395 & TEAM-396

**What they need from you:**

### Files Created
- `src/http/backend.rs` - AppState and trait
- `src/http/server.rs` - HTTP server
- `src/http/routes.rs` - Router
- `src/http/health.rs` - Health endpoint
- `src/http/ready.rs` - Ready endpoint

### APIs Exposed

```rust
// AppState for all handlers
pub struct AppState {
    pub config: Arc<Config>,
    pub backend: Arc<dyn InferenceBackend>,
    pub generation_engine: Arc<GenerationEngine>,
}

// Server lifecycle
pub struct HttpServer {
    pub async fn new(state: AppState, bind_addr: &str) -> Result<Self>;
    pub async fn serve(self) -> Result<()>;
}

// Router extension point
pub fn create_router(state: AppState) -> Router;
```

### What Works
- HTTP server starts and stops
- Health and ready endpoints
- CORS middleware
- Logging middleware
- Graceful shutdown

### What TEAM-395 Will Add
- POST /v1/jobs endpoint
- GET /v1/jobs/:id/stream endpoint
- SSE streaming

### What TEAM-396 Will Add
- Request validation
- Authentication middleware

---

## 📊 Progress Tracking

Track your progress:

- [ ] Day 1: Design complete
- [ ] Day 2: AppState working
- [ ] Day 3: Server working
- [ ] Day 4: Routes and middleware working
- [ ] Day 5: Health/ready working, ready for handoff

---

**TEAM-394: You're building the foundation. Make it solid and extensible.** 🏗️
