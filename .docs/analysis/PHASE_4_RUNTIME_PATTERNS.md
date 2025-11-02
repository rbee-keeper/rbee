# Phase 4: Service Runtime Patterns Analysis

**Analysis Date:** November 2, 2025  
**Scope:** Tokio runtime configs, HTTP servers, narration initialization  
**Status:** ✅ COMPLETE

---

## Executive Summary

All 4 main binaries use **multi-threaded tokio runtime** except `llm-worker-rbee` which uses **single-threaded** for optimal CPU inference performance. All binaries use **Axum** for HTTP serving and initialize narration before the async runtime.

---

## 1. rbee-keeper Runtime Analysis

### File: `bin/00_rbee_keeper/src/main.rs`

**Tokio Runtime:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
```

**Configuration:** Multi-threaded (default)

**HTTP Server:** ❌ No (CLI client only)

**Narration Initialization:**
```rust
// TEAM-295: If no subcommand provided, launch Tauri GUI instead
if cli.command.is_none() {
    launch_gui();
    return Ok(());
}
```

**Pattern:** Narration initialized implicitly via `n!()` macro (no explicit init)

**Key Characteristics:**
- **Dual-mode:** CLI + Tauri GUI
- **Client-only:** Makes HTTP requests to queen
- **No server:** Does not bind to any port
- **Narration:** Uses `n!()` macro for CLI feedback

---

## 2. queen-rbee Runtime Analysis

### File: `bin/10_queen_rbee/src/main.rs`

**Tokio Runtime:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
```

**Configuration:** Multi-threaded (default)

**HTTP Server:** ✅ Yes (Axum)

**Server Initialization:**
```rust
let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
let router = create_router(job_server, telemetry);

n!("listen", "Listening on http://{}", addr);
n!("ready", "Ready to accept connections");

let listener = tokio::net::TcpListener::bind(addr).await?;
axum::serve(listener, router).await.map_err(|e| {
    n!("error", "Server error: {}", e);
    anyhow::anyhow!("Server failed: {}", e)
})
```

**Default Port:** 7833

**Narration Initialization:**
```rust
// TEAM-164: Initialize SSE sink for distributed narration
// TEAM-204: Removed init() - no global channel, job channels created on-demand

n!("start", "Queen-rbee starting on port {} (localhost-only mode)", args.port);
```

**Pattern:** Narration initialized implicitly (no explicit init needed)

**Key Characteristics:**
- **HTTP daemon:** Axum server on port 7833
- **Job registry:** `Arc<JobRegistry<String>>`
- **Telemetry registry:** `Arc<TelemetryRegistry>`
- **Hive discovery:** Background task after 5s delay
- **Static files:** Embedded via `rust-embed` (production)
- **Dev proxy:** `/dev/*` → Vite server (port 7834)

**Router Structure:**
```rust
Router::new()
    .route("/health", get(health))
    .route("/v1/jobs", post(submit_job))
    .route("/v1/jobs/:job_id/stream", get(stream_job))
    .route("/v1/hive-heartbeat", post(hive_heartbeat))
    .layer(CorsLayer::permissive())
```

---

## 3. rbee-hive Runtime Analysis

### File: `bin/20_rbee_hive/src/main.rs`

**Tokio Runtime:**
```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
```

**Configuration:** Multi-threaded (default)

**HTTP Server:** ✅ Yes (Axum)

**Server Initialization:**
```rust
let addr = SocketAddr::from(([0, 0, 0, 0], args.port));

n!("listen", "✅ Listening on http://{}", addr);

let listener = tokio::net::TcpListener::bind(addr).await?;

n!("ready", "✅ Hive ready");

// Start heartbeat task
n!("heartbeat_start", "💓 Heartbeat task started (sending to {})", args.queen_url);

axum::serve(listener, app).await?;
```

**Default Port:** 7835

**Narration Initialization:**
```rust
// TEAM-385: Tracing subscriber initialization removed
// The narration system handles its own output formatting
// We don't need to suppress tracing since narration uses its own channels

n!("startup", "🐝 Starting rbee-hive on port {}", args.port);
```

**Pattern:** Narration initialized implicitly (no explicit init)

**Key Characteristics:**
- **HTTP daemon:** Axum server on port 7835
- **Job registry:** `Arc<JobRegistry<String>>`
- **Model catalog:** `Arc<ModelCatalog>`
- **Worker catalog:** `Arc<WorkerCatalog>`
- **Model provisioner:** `Arc<ModelProvisioner>`
- **Heartbeat:** Background task to queen
- **Static files:** Embedded via `rust-embed` (production)
- **Dev proxy:** `/dev/*` → Vite server (port 7836)

**Router Structure:**
```rust
Router::new()
    .route("/health", get(health))
    .route("/v1/jobs", post(submit_job))
    .route("/v1/jobs/:job_id/stream", get(stream_job))
    .route("/v1/capabilities", get(capabilities))
    .route("/v1/heartbeats/stream", get(heartbeat_stream))
    .layer(CorsLayer::permissive())
```

---

## 4. llm-worker-rbee Runtime Analysis

### File: `bin/30_llm_worker_rbee/src/main.rs`

**Tokio Runtime:**
```rust
#[tokio::main(flavor = "current_thread")] // CRITICAL: Single-threaded!
async fn main() -> anyhow::Result<()> {
```

**Configuration:** ⚠️ **Single-threaded** (for optimal CPU performance)

**Why Single-Threaded:**
```rust
/// CRITICAL: Uses single-threaded tokio runtime for SPEED
/// - flavor = "current_thread" ensures NO thread pool
/// - All async operations run on ONE thread
/// - No context switching overhead
/// - Optimal for CPU-bound inference
```

**HTTP Server:** ✅ Yes (Axum)

**Server Initialization:**
```rust
n!(ACTION_STARTUP, "Starting Candle worker on port {}", args.port);

// Load model first
n!(ACTION_MODEL_LOAD, "Loading Llama model from {}", args.model);

let backend = match CandleInferenceBackend::load(&args.model) {
    Ok(backend) => {
        n!("model_load_success", "Model loaded successfully");
        backend
    }
    Err(e) => {
        n!("model_load_failed", "Model load failed: {}", error_msg);
        return Err(e);
    }
};

// Create router
let router = create_router(/* ... */);

// Start server
let addr = if args.local_mode {
    SocketAddr::from(([127, 0, 0, 1], args.port))
} else {
    SocketAddr::from(([0, 0, 0, 0], args.port))
};

let listener = tokio::net::TcpListener::bind(addr).await?;
axum::serve(listener, router).await?;
```

**Default Ports:** 9300+ (dynamic allocation by hive)

**Narration Initialization:**
```rust
// No explicit initialization - uses n!() macro directly
n!(ACTION_STARTUP, "Starting Candle worker on port {}", args.port);
```

**Pattern:** Narration initialized implicitly

**Key Characteristics:**
- **HTTP daemon:** Axum server on dynamic port
- **Single-threaded:** Optimal for CPU inference
- **Model loading:** Blocking operation before server start
- **Job registry:** `Arc<JobRegistry<TokenResponse>>`
- **Request queue:** `Arc<Mutex<RequestQueue>>`
- **Generation engine:** Background task for inference
- **Heartbeat:** Background task to queen
- **Local mode:** Binds to 127.0.0.1 only (no auth)
- **Network mode:** Binds to 0.0.0.0 (requires auth token)

**Router Structure:**
```rust
Router::new()
    .route("/health", get(health))
    .route("/ready", get(ready))
    .route("/v1/infer", post(infer))
    .route("/v1/jobs", post(submit_job))
    .route("/v1/jobs/:job_id/stream", get(stream_job))
```

---

## 5. Runtime Comparison Matrix

| Binary | Tokio Runtime | HTTP Server | Default Port | Narration Init | Static Files |
|--------|---------------|-------------|--------------|----------------|--------------|
| `rbee-keeper` | Multi-threaded | ❌ No | N/A | Implicit | N/A |
| `queen-rbee` | Multi-threaded | ✅ Axum | 7833 | Implicit | ✅ Embedded |
| `rbee-hive` | Multi-threaded | ✅ Axum | 7835 | Implicit | ✅ Embedded |
| `llm-worker-rbee` | **Single-threaded** | ✅ Axum | 9300+ | Implicit | ❌ No |

---

## 6. Narration Initialization Pattern

### All Binaries Use Implicit Initialization

**Pattern:**
```rust
// No explicit init() call needed
// Narration is initialized on first n!() macro use

n!("startup", "Service starting...");
```

**Why No Explicit Init:**
- TEAM-204: Removed global channel initialization
- Job channels created on-demand
- Thread-local context for job_id propagation
- SSE sink initialized per-job

**Historical Context:**
```rust
// TEAM-164: Initialize SSE sink for distributed narration
// TEAM-204: Removed init() - no global channel, job channels created on-demand
```

---

## 7. HTTP Server Patterns

### Common Axum Setup

**All HTTP servers follow this pattern:**

1. **Create router with routes**
2. **Add CORS layer** (permissive for development)
3. **Bind to socket address**
4. **Narrate listen address**
5. **Serve with axum::serve()**

**Example (queen-rbee):**
```rust
let router = create_router(job_server, telemetry);
let addr = SocketAddr::from(([0, 0, 0, 0], args.port));

n!("listen", "Listening on http://{}", addr);

let listener = tokio::net::TcpListener::bind(addr).await?;
axum::serve(listener, router).await?;
```

---

### Static File Serving

**Production (Release Mode):**
- Files embedded via `rust-embed`
- Served from binary (no external files needed)
- Pattern: `RustEmbed` derive macro

**Development (Debug Mode):**
- `/dev/*` proxied to Vite dev server
- Hot reload support
- Separate ports (7834 for queen, 7836 for hive)

**Code Pattern:**
```rust
#[cfg(debug_assertions)]
{
    eprintln!("🔧 [QUEEN] Running in DEBUG mode");
    eprintln!("   - /dev/{{*path}} → Proxy to Vite dev server (port 7834)");
    eprintln!("   - / → Embedded static files (may be stale, rebuild to update)");
}

#[cfg(not(debug_assertions))]
{
    eprintln!("🚀 [QUEEN] Running in RELEASE mode");
    eprintln!("   - / → Embedded static files (production)");
}
```

---

## 8. Background Tasks

### queen-rbee Background Tasks

1. **Hive Discovery** (after 5s delay)
   ```rust
   tokio::spawn(async move {
       if let Err(e) = discovery::discover_hives_on_startup(&queen_url).await {
           n!("discovery_error", "❌ Hive discovery failed: {}", e);
       }
   });
   ```

### rbee-hive Background Tasks

1. **Heartbeat to Queen**
   ```rust
   n!("heartbeat_start", "💓 Heartbeat task started (sending to {})", args.queen_url);
   heartbeat::start_heartbeat_task(hive_info, queen_url, running_flag);
   ```

### llm-worker-rbee Background Tasks

1. **Generation Engine** (inference loop)
2. **Heartbeat to Queen**

---

## 9. Port Configuration

### Default Ports

| Service | Port | Configurable | Usage |
|---------|------|--------------|-------|
| queen-rbee | 7833 | ✅ Yes | HTTP API |
| queen-rbee (dev) | 7834 | ❌ No | Vite dev server |
| rbee-hive | 7835 | ✅ Yes | HTTP API |
| rbee-hive (dev) | 7836 | ❌ No | Vite dev server |
| llm-worker-rbee | 9300+ | ✅ Yes | HTTP API (dynamic) |

### Port Allocation Strategy

**Static Ports:**
- queen-rbee: 7833 (fixed, well-known)
- rbee-hive: 7835 (fixed, well-known)

**Dynamic Ports:**
- llm-worker-rbee: Assigned by hive when spawning
- Range: 9300-9399 (100 workers max per hive)

---

## 10. Key Findings

### Runtime Optimization

1. **llm-worker-rbee uses single-threaded runtime**
   - Reason: Optimal for CPU-bound inference
   - Benefit: No context switching overhead
   - Trade-off: Cannot use thread pool for parallel tasks

2. **All other binaries use multi-threaded runtime**
   - Reason: I/O-bound operations (HTTP, SSE, SSH)
   - Benefit: Better concurrency for multiple requests
   - Default: Tokio auto-detects CPU count

### Narration Pattern

1. **No explicit initialization needed**
   - Removed in TEAM-204
   - Job channels created on-demand
   - Thread-local context for job_id

2. **Consistent usage across all binaries**
   - `n!()` macro everywhere
   - Action constants for common operations
   - Multi-mode support (human, cute, story)

### HTTP Server Pattern

1. **All use Axum framework**
   - Modern, fast, type-safe
   - Tower middleware support
   - CORS enabled for development

2. **Static file serving**
   - Embedded in production (rust-embed)
   - Proxied in development (Vite)
   - Separate dev/prod code paths

---

**Next Phase:** [PHASE_5_FRONTEND_PACKAGES.md](./PHASE_5_FRONTEND_PACKAGES.md)
