# ROCm CFG Attribute Audit

**Date:** 2025-11-13  
**Purpose:** Audit all `#[cfg(feature = "...")]` callsites for ROCm support  
**Status:** 🔴 **ROCm MISSING** - Need to add ROCm alongside CPU, CUDA, Metal

---

## Summary

**Current backends with cfg attributes:**
- ✅ **CPU** - Fully implemented
- ✅ **CUDA** - Fully implemented  
- ✅ **Metal** - Fully implemented
- 🔴 **ROCm** - **MISSING** (needs to be added)

**Files requiring ROCm cfg attributes:** 5 files

---

## 1. Cargo.toml - Feature Definitions

**File:** `/home/vince/Projects/rbee/bin/30_llm_worker_rbee/Cargo.toml`

### Current Features (Lines 231-244)

```toml
[features]
default = ["cpu"]

# Backend features (mutually exclusive at build time)
cpu = []
cuda = ["candle-kernels", "cudarc", "candle-core/cuda", "candle-nn/cuda", "flash-attn"]
metal = ["candle-core/metal", "candle-nn/metal"]

# Flash Attention feature (2-4x faster inference on CUDA)
flash-attn = ["candle-flash-attn", "candle-transformers/flash-attn"]
```

### ❌ MISSING: ROCm Feature

**Need to add:**
```toml
rocm = ["candle-kernels", "candle-core/rocm", "candle-nn/rocm"]
```

**Note:** ROCm support in `/home/vince/Projects/rbee/deps/candle` must expose:
- `candle-core/rocm` feature
- `candle-nn/rocm` feature
- ROCm kernels in `candle-kernels`

---

## 2. device.rs - Device Initialization

**File:** `/home/vince/Projects/rbee/bin/30_llm_worker_rbee/src/device.rs`

### Current Implementation

#### ✅ CPU (Lines 16-23)
```rust
#[cfg(feature = "cpu")]
pub fn init_cpu_device() -> CandleResult<Device> {
    tracing::info!("Initializing CPU device");
    n!(ACTION_DEVICE_INIT, "Initialized CPU device");
    Ok(Device::Cpu)
}
```

#### ✅ CUDA (Lines 26-35)
```rust
#[cfg(feature = "cuda")]
pub fn init_cuda_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing CUDA device {}", gpu_id);
    let device = Device::new_cuda(gpu_id)?;
    n!(ACTION_DEVICE_INIT, "Initialized CUDA device {}", gpu_id);
    Ok(device)
}
```

#### ✅ Metal (Lines 39-48)
```rust
#[cfg(feature = "metal")]
pub fn init_metal_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing Apple Metal device (GPU) {}", gpu_id);
    let device = Device::new_metal(gpu_id)?;
    n!(ACTION_DEVICE_INIT, "Initialized Apple Metal device {}", gpu_id);
    Ok(device)
}
```

### ❌ MISSING: ROCm Device Init

**Need to add:**
```rust
/// Initialize AMD ROCm device (GPU)
/// Note: ROCm is AMD's GPU API, equivalent to CUDA for NVIDIA
#[cfg(feature = "rocm")]
pub fn init_rocm_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing AMD ROCm device (GPU) {}", gpu_id);
    let device = Device::new_rocm(gpu_id)?;
    n!(ACTION_DEVICE_INIT, "Initialized AMD ROCm device {}", gpu_id);
    Ok(device)
}
```

### Tests (Lines 68-90)

#### ✅ CPU Test
```rust
#[test]
#[cfg(feature = "cpu")]
fn test_cpu_device_init() {
    let device = init_cpu_device().unwrap();
    verify_device(&device).unwrap();
}
```

#### ✅ CUDA Test
```rust
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_device_init() {
    if let Ok(device) = init_cuda_device(0) {
        verify_device(&device).unwrap();
    }
}
```

#### ✅ Metal Test
```rust
#[test]
#[cfg(feature = "metal")]
fn test_metal_device_init() {
    if let Ok(device) = init_metal_device(0) {
        verify_device(&device).unwrap();
    }
}
```

### ❌ MISSING: ROCm Test

**Need to add:**
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_rocm_device_init() {
    // Only run if ROCm is available
    if let Ok(device) = init_rocm_device(0) {
        verify_device(&device).unwrap();
    }
}
```

---

## 3. backend/inference.rs - Model Loading

**File:** `/home/vince/Projects/rbee/bin/30_llm_worker_rbee/src/backend/inference.rs`

### Current Implementation

#### ✅ CPU Load (Lines 106-140)
```rust
#[cfg(feature = "cpu")]
pub fn load(model_path: &str) -> Result<Self> {
    let path = Path::new(model_path);
    let device = Device::Cpu;
    
    let model = models::load_model(model_path, &device, None)?;
    let model_size_bytes = models::calculate_model_size(model_path)?;
    let tokenizer = tokenizer_loader::load_tokenizer(path)?;
    let cached_eos_token = tokenizer.token_to_id("</s>");
    
    // ... logging ...
    
    Ok(Self { model, tokenizer, device, model_size_bytes, cached_eos_token })
}
```

#### ✅ CUDA Load (Lines 142-176)
```rust
#[cfg(feature = "cuda")]
pub fn load(model_path: &str, gpu_id: usize) -> Result<Self> {
    let path = Path::new(model_path);
    let device = Device::new_cuda(gpu_id)?;
    
    let model = models::load_model(model_path, &device, None)?;
    let model_size_bytes = models::calculate_model_size(model_path)?;
    let tokenizer = tokenizer_loader::load_tokenizer(path)?;
    let cached_eos_token = tokenizer.token_to_id("</s>");
    
    // ... logging ...
    
    Ok(Self { model, tokenizer, device, model_size_bytes, cached_eos_token })
}
```

#### ✅ Metal Load (Lines 178-212)
```rust
#[cfg(feature = "metal")]
pub fn load(model_path: &str, gpu_id: usize) -> Result<Self> {
    let path = Path::new(model_path);
    let device = Device::new_metal(gpu_id)?;
    
    let model = models::load_model(model_path, &device, None)?;
    let model_size_bytes = models::calculate_model_size(model_path)?;
    let tokenizer = tokenizer_loader::load_tokenizer(path)?;
    let cached_eos_token = tokenizer.token_to_id("</s>");
    
    // ... logging ...
    
    Ok(Self { model, tokenizer, device, model_size_bytes, cached_eos_token })
}
```

### ❌ MISSING: ROCm Load

**Need to add:**
```rust
#[cfg(feature = "rocm")]
pub fn load(model_path: &str, gpu_id: usize) -> Result<Self> {
    let path = Path::new(model_path);
    let device = Device::new_rocm(gpu_id)?;
    
    let model = models::load_model(model_path, &device, None)?;
    let model_size_bytes = models::calculate_model_size(model_path)?;
    let tokenizer = tokenizer_loader::load_tokenizer(path)?;
    let cached_eos_token = tokenizer.token_to_id("</s>");
    
    tracing::info!(
        architecture = model.architecture(),
        vocab_size = model.vocab_size(),
        tokenizer_vocab = tokenizer.get_vocab_size(true),
        model_size_mb = model_size_bytes / 1_000_000,
        cached_eos_token = ?cached_eos_token,
        "Model and tokenizer loaded successfully"
    );
    
    n!(
        ACTION_MODEL_LOAD,
        "Loaded {} model ({} MB, vocab: {})",
        model.architecture(),
        model_size_bytes / 1_000_000,
        model.vocab_size()
    );
    
    Ok(Self { model, tokenizer, device, model_size_bytes, cached_eos_token })
}
```

### VRAM Usage (Lines 496-499)

#### ✅ CUDA VRAM
```rust
fn vram_usage(&self) -> u64 {
    #[cfg(feature = "cuda")]
    {
        self.model_size_bytes
    }
}
```

### ❌ MISSING: ROCm VRAM

**Need to add:**
```rust
fn vram_usage(&self) -> u64 {
    #[cfg(feature = "cuda")]
    {
        self.model_size_bytes
    }
    #[cfg(feature = "rocm")]
    {
        self.model_size_bytes
    }
}
```

---

## 4. Binary Entry Points

### ✅ CPU Binary (src/bin/cpu.rs)

**Lines 1-114** - Complete implementation
- No cfg needed (always available)
- Device: `"cpu:0"`
- Binary: `llm-worker-rbee-cpu`

### ✅ CUDA Binary (src/bin/cuda.rs)

**Lines 1-123** - Complete implementation
- Feature-gated: `required-features = ["cuda"]`
- Device: `format!("cuda:{}", args.cuda_device)`
- Binary: `llm-worker-rbee-cuda`
- CLI arg: `--cuda-device <N>`

### ✅ Metal Binary (src/bin/metal.rs)

**Lines 1-126** - Complete implementation
- Feature-gated: `required-features = ["metal"]`
- Device: `format!("metal:{}", args.metal_device)`
- Binary: `llm-worker-rbee-metal`
- CLI arg: `--metal-device <N>`

### ❌ MISSING: ROCm Binary (src/bin/rocm.rs)

**Need to create:** `/home/vince/Projects/rbee/bin/30_llm_worker_rbee/src/bin/rocm.rs`

**Template:**
```rust
// TEAM-XXX: ROCm GPU worker binary
//!
//! Uses AMD ROCm for GPU inference with strict device residency.
//! This binary is feature-gated to ROCm backend only.

use anyhow::Result;
use clap::Parser;
use llm_worker_rbee::{backend::CandleInferenceBackend, setup_worker_with_backend, HttpServer};
use std::net::SocketAddr;

/// CLI arguments for ROCm worker daemon
#[derive(Parser, Debug)]
#[command(name = "llorch-rocm-candled")]
#[command(about = "AMD ROCm GPU Candle-based multi-model worker daemon")]
struct Args {
    /// Worker ID (UUID) - assigned by pool-managerd
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF or SafeTensors format)
    #[arg(long)]
    model: String,

    /// Model reference (e.g., "hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    #[arg(long)]
    model_ref: String,

    /// HTTP server port - assigned by pool-managerd
    #[arg(long)]
    port: u16,

    /// Hive URL - where to send heartbeats
    #[arg(long)]
    hive_url: String,

    /// ROCm device ID (default: 0)
    #[arg(long, default_value = "0")]
    rocm_device: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Initialize tracing (JSON format for structured logging)
    tracing_subscriber::fmt().with_target(false).json().init();

    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        rocm_device = args.rocm_device,
        backend = "rocm",
        "Starting llorch-rocm-candled"
    );

    // Load model to ROCm GPU
    tracing::info!(model = %args.model, rocm_device = args.rocm_device, "Loading model to ROCm GPU...");
    let mut backend = CandleInferenceBackend::load(&args.model, args.rocm_device)?;
    tracing::info!("Model loaded successfully on ROCm GPU {}", args.rocm_device);

    // GPU Warmup
    backend.warmup()?;
    tracing::info!("ROCm GPU warmup complete - ready for inference");

    // Start heartbeat task
    tracing::info!("Starting heartbeat task");

    let worker_info = worker_contract::WorkerInfo {
        id: args.worker_id.clone(),
        model_id: args.model_ref.clone(),
        device: format!("rocm:{}", args.rocm_device),
        port: args.port,
        status: worker_contract::WorkerStatus::Ready,
        implementation: "llm-worker-rbee-rocm".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let _heartbeat_handle =
        llm_worker_rbee::heartbeat::start_heartbeat_task(worker_info, args.hive_url.clone());
    tracing::info!("Heartbeat task started (30s interval)");

    // Start HTTP server
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));

    let expected_token = std::env::var("LLORCH_API_TOKEN").unwrap_or_else(|_| {
        tracing::info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
        String::new()
    });

    if !expected_token.is_empty() {
        tracing::info!("✅ API token loaded (authentication enabled)");
    }

    let router = setup_worker_with_backend(backend, expected_token);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!("llorch-rocm-candled ready on port {} (ROCm GPU {})", args.port, args.rocm_device);

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
```

### Cargo.toml Binary Definition

**Need to add to Cargo.toml (after line 264):**
```toml
[[bin]]
name = "llm-worker-rbee-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

---

## 5. error.rs - Error Types

**File:** `/home/vince/Projects/rbee/bin/30_llm_worker_rbee/src/error.rs`

### Current Implementation (Lines 23-25)

```rust
#[error("CUDA error: {0}")]
#[cfg(feature = "cuda")]
CudaError(String),
```

### ❌ MISSING: ROCm Error

**Need to add:**
```rust
#[error("ROCm error: {0}")]
#[cfg(feature = "rocm")]
RocmError(String),
```

---

## Implementation Checklist

### Phase 1: Candle ROCm Support (Prerequisite)

- [ ] Verify `/home/vince/Projects/rbee/deps/candle` has ROCm support
- [ ] Check `candle-core` exposes `Device::new_rocm(gpu_id)`
- [ ] Check `candle-core/Cargo.toml` has `rocm` feature
- [ ] Check `candle-nn/Cargo.toml` has `rocm` feature
- [ ] Check `candle-kernels` has ROCm kernel support

### Phase 2: Feature Definition

- [ ] Add `rocm` feature to `Cargo.toml` (line ~240)
- [ ] Add ROCm binary definition to `Cargo.toml` (after line 264)

### Phase 3: Device Initialization

- [ ] Add `init_rocm_device()` to `src/device.rs`
- [ ] Add ROCm test to `src/device.rs`

### Phase 4: Inference Backend

- [ ] Add `#[cfg(feature = "rocm")]` load() to `src/backend/inference.rs`
- [ ] Add ROCm VRAM tracking to `vram_usage()`

### Phase 5: Binary Entry Point

- [ ] Create `src/bin/rocm.rs` (copy from `cuda.rs`, modify for ROCm)
- [ ] Update CLI args: `--rocm-device` instead of `--cuda-device`
- [ ] Update device string: `format!("rocm:{}", args.rocm_device)`
- [ ] Update binary name: `llm-worker-rbee-rocm`

### Phase 6: Error Handling

- [ ] Add `RocmError` variant to `src/error.rs`

### Phase 7: Testing

- [ ] Test compilation: `cargo check --features rocm`
- [ ] Test build: `cargo build --bin llm-worker-rbee-rocm --features rocm`
- [ ] Test device init: `cargo test --features rocm test_rocm_device_init`
- [ ] Test model loading on ROCm GPU
- [ ] Test inference on ROCm GPU

---

## Pattern Consistency

All backends follow the same pattern:

| Backend | Feature | Device Init | Load Signature | Binary | CLI Arg |
|---------|---------|-------------|----------------|--------|---------|
| CPU | `cpu` | `init_cpu_device()` | `load(model_path)` | `llm-worker-rbee-cpu` | N/A |
| CUDA | `cuda` | `init_cuda_device(gpu_id)` | `load(model_path, gpu_id)` | `llm-worker-rbee-cuda` | `--cuda-device` |
| Metal | `metal` | `init_metal_device(gpu_id)` | `load(model_path, gpu_id)` | `llm-worker-rbee-metal` | `--metal-device` |
| **ROCm** | `rocm` | `init_rocm_device(gpu_id)` | `load(model_path, gpu_id)` | `llm-worker-rbee-rocm` | `--rocm-device` |

**ROCm should follow the exact same pattern as CUDA and Metal.**

---

## Next Steps

1. **Verify Candle ROCm support** - Check `/home/vince/Projects/rbee/deps/candle` has all ROCm APIs
2. **Add feature definition** - Update `Cargo.toml`
3. **Implement device init** - Add to `device.rs`
4. **Implement model loading** - Add to `backend/inference.rs`
5. **Create binary** - Add `src/bin/rocm.rs`
6. **Test compilation** - `cargo check --features rocm`

---

**Status:** 🔴 **ROCm support needs to be added to match CPU/CUDA/Metal pattern**
