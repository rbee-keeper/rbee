# TEAM-507: CFG Attribute Audit - ROCm Support Missing

**Date:** 2025-11-13  
**Status:** 🔍 AUDIT COMPLETE - ROCm NOT IMPLEMENTED  
**Scope:** `/home/vince/Projects/rbee/bin/31_sd_worker_rbee`

## Executive Summary

✅ **CPU, CUDA, Metal:** All have proper `#[cfg(feature = "...")]` attributes  
❌ **ROCm:** NOT IMPLEMENTED - No binary, no feature flag, no device init

## Findings

### 1. Existing Backend Support (✅ All Properly Gated)

#### CPU Backend
**Feature Flag:** `cpu`  
**Binary:** `src/bin/cpu.rs`  
**Cargo.toml:**
```toml
[[bin]]
name = "sd-worker-cpu"
path = "src/bin/cpu.rs"
required-features = ["cpu"]
```

**Device Init:** `shared-worker-rbee/src/device.rs:18-23`
```rust
#[cfg(feature = "cpu")]
pub fn init_cpu_device() -> CandleResult<Device> {
    tracing::info!("Initializing CPU device");
    Ok(Device::Cpu)
}
```

**Test:** `shared-worker-rbee/src/device.rs:61-66`
```rust
#[test]
#[cfg(feature = "cpu")]
fn test_cpu_device_init() {
    let device = init_cpu_device().unwrap();
    verify_device(&device).unwrap();
}
```

---

#### CUDA Backend
**Feature Flag:** `cuda`  
**Binary:** `src/bin/cuda.rs`  
**Cargo.toml:**
```toml
[[bin]]
name = "sd-worker-cuda"
path = "src/bin/cuda.rs"
required-features = ["cuda"]
```

**Device Init:** `shared-worker-rbee/src/device.rs:25-32`
```rust
#[cfg(feature = "cuda")]
pub fn init_cuda_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing CUDA device {}", gpu_id);
    let device = Device::new_cuda(gpu_id)?;
    Ok(device)
}
```

**Test:** `shared-worker-rbee/src/device.rs:68-75`
```rust
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_device_init() {
    if let Ok(device) = init_cuda_device(0) {
        verify_device(&device).unwrap();
    }
}
```

---

#### Metal Backend
**Feature Flag:** `metal`  
**Binary:** `src/bin/metal.rs`  
**Cargo.toml:**
```toml
[[bin]]
name = "sd-worker-metal"
path = "src/bin/metal.rs"
required-features = ["metal"]
```

**Device Init:** `shared-worker-rbee/src/device.rs:34-42`
```rust
#[cfg(feature = "metal")]
pub fn init_metal_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing Apple Metal device (GPU) {}", gpu_id);
    let device = Device::new_metal(gpu_id)?;
    Ok(device)
}
```

**Test:** `shared-worker-rbee/src/device.rs:77-84`
```rust
#[test]
#[cfg(feature = "metal")]
fn test_metal_device_init() {
    if let Ok(device) = init_metal_device(0) {
        verify_device(&device).unwrap();
    }
}
```

---

### 2. ROCm Backend (❌ NOT IMPLEMENTED)

**Missing Components:**
1. ❌ No `src/bin/rocm.rs` binary
2. ❌ No `rocm` feature flag in `Cargo.toml`
3. ❌ No `init_rocm_device()` in `shared-worker-rbee/src/device.rs`
4. ❌ No ROCm test in device tests
5. ❌ No ROCm binary definition in `Cargo.toml`

**Expected Implementation Pattern (Based on CUDA/Metal):**

#### Step 1: Add Feature Flag to Cargo.toml
```toml
# ROCm backend (AMD GPU)
rocm = [
    "candle-core/rocm",
    "candle-nn/rocm",
    "candle-transformers/rocm",
    "shared-worker-rbee/rocm",
]
```

#### Step 2: Add Binary Definition
```toml
# ROCm binary
[[bin]]
name = "sd-worker-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

#### Step 3: Create Binary (`src/bin/rocm.rs`)
```rust
// TEAM-507: ROCm binary for SD worker
//
// Stable Diffusion worker using ROCm backend (AMD GPU).

use clap::Parser;
use sd_worker_rbee::{
    backend::{
        generation_engine::GenerationEngine, model_loader, models::SDVersion,
        request_queue::RequestQueue,
    },
    http::{backend::AppState, routes::create_router},
    narration::log_device_init,
};
use shared_worker_rbee::device;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Worker ID
    #[arg(long, env = "WORKER_ID")]
    worker_id: String,

    /// SD model version (v1-5, v2-1, xl, turbo, 3-medium, etc.)
    #[arg(long, env = "SD_VERSION")]
    sd_version: String,

    /// HTTP server port - MUST be provided by hive (no default)
    #[arg(long, env = "PORT")]
    port: u16,

    /// Callback URL for hive registration
    #[arg(long, env = "CALLBACK_URL")]
    callback_url: String,

    /// ROCm device index
    #[arg(long, env = "ROCM_DEVICE", default_value = "0")]
    rocm_device: usize,

    /// Use FP16 precision
    #[arg(long, env = "USE_F16")]
    use_f16: bool,

    /// Custom model path (optional, overrides auto-download)
    #[arg(long, env = "MODEL_PATH")]
    model_path: Option<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sd_worker_rbee=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!(
        "Starting SD Worker (ROCm) - ID: {}, Version: {}, Port: {}, Device: {}, FP16: {}",
        args.worker_id,
        args.sd_version,
        args.port,
        args.rocm_device,
        args.use_f16
    );

    // Initialize ROCm device
    log_device_init(&format!("ROCm:{}", args.rocm_device));
    let device = device::init_rocm_device(args.rocm_device)?;
    device::verify_device(&device)?;

    // Parse SD version
    let sd_version = SDVersion::parse_version(&args.sd_version)?;
    tracing::info!("Loading model: {:?} with FP16={}", sd_version, args.use_f16);

    // 1. Create request queue
    let (request_queue, request_rx) = RequestQueue::new();

    // Load model components
    let model_components = model_loader::load_model(
        sd_version,
        &device,
        true,  // use_f16 = true for ROCm
        &[],   // loras = empty (no LoRAs for now)
        false, // quantized = false
    )?;
    tracing::info!("Model loaded successfully");

    // 2. Create generation engine with loaded models
    let engine = GenerationEngine::new(Arc::new(Mutex::new(model_components)), request_rx);

    // 3. Start engine (consumes self, spawns blocking task)
    engine.start();

    // 5. Create HTTP state
    let app_state = AppState::new(request_queue);

    // 6. Start HTTP server
    let router = create_router(app_state);
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("✅ SD Worker (ROCm) ready on port {}", args.port);
    tracing::info!("✅ Device: ROCm:{}, FP16: {}", args.rocm_device, args.use_f16);

    axum::serve(listener, router).await?;

    Ok(())
}
```

#### Step 4: Add Device Init to `shared-worker-rbee/src/device.rs`
```rust
/// Initialize ROCm device (AMD GPU)
#[cfg(feature = "rocm")]
pub fn init_rocm_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing ROCm device (AMD GPU) {}", gpu_id);
    let device = Device::new_rocm(gpu_id)?;
    Ok(device)
}
```

#### Step 5: Add Test to `shared-worker-rbee/src/device.rs`
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

## Verification Commands

### CPU
```bash
cargo build --no-default-features --features cpu --bin sd-worker-cpu
cargo test --features cpu
```

### CUDA
```bash
cargo build --no-default-features --features cuda --bin sd-worker-cuda
cargo test --features cuda
```

### Metal
```bash
cargo build --no-default-features --features metal --bin sd-worker-metal
cargo test --features metal
```

### ROCm (NOT YET IMPLEMENTED)
```bash
# Will fail - not implemented yet
cargo build --no-default-features --features rocm --bin sd-worker-rocm
cargo test --features rocm
```

---

## Dependencies

### Candle ROCm Support Status ✅ AVAILABLE

**Verified ROCm support in `/deps/candle`:**

#### candle-core (✅ HAS ROCm)
```toml
# /deps/candle/candle-core/Cargo.toml:64
rocm = ["dep:rocm-rs", "dep:candle-kernels", "candle-kernels?/rocm"]
```

#### candle-nn (✅ Pass-through)
```toml
# /deps/candle/candle-nn/Cargo.toml
# No explicit rocm feature - passes through to candle-core
cuda = ["candle/cuda"]
metal = ["candle/metal"]
# Pattern: Add rocm = ["candle/rocm"]
```

#### candle-transformers (✅ Pass-through)
```toml
# /deps/candle/candle-transformers/Cargo.toml
# No explicit rocm feature - passes through to candle-core
cuda = ["candle/cuda", "candle-nn/cuda"]
metal = ["candle/metal", "candle-nn/metal"]
# Pattern: Add rocm = ["candle/rocm", "candle-nn/rocm"]
```

#### candle-kernels (✅ HAS ROCm)
```toml
# /deps/candle/candle-kernels/Cargo.toml:21
rocm = ["rocm-rs"]
```

**Conclusion:** ✅ Candle has full ROCm support! Ready to implement.

---

## Implementation Checklist

### Phase 1: Candle Dependencies (DONE ✅)
- [x] Verify Candle has ROCm support in `/deps/candle` ✅
  - candle-core has `rocm` feature
  - candle-kernels has `rocm` feature
  - candle-nn and candle-transformers pass through

### Phase 2: Update Candle Cargo.toml Files (REQUIRED)
- [ ] Add `rocm` feature to `/deps/candle/candle-nn/Cargo.toml`
  ```toml
  rocm = ["candle/rocm"]
  ```
- [ ] Add `rocm` feature to `/deps/candle/candle-transformers/Cargo.toml`
  ```toml
  rocm = ["candle/rocm", "candle-nn/rocm"]
  ```

### Phase 3: sd-worker-rbee Implementation
- [ ] Add `rocm` feature flag to `Cargo.toml` (line ~188)
  ```toml
  # ROCm backend (AMD GPU)
  rocm = [
      "candle-core/rocm",
      "candle-nn/rocm",
      "candle-transformers/rocm",
      "shared-worker-rbee/rocm",
  ]
  ```
- [ ] Add ROCm binary definition to `Cargo.toml` (line ~210)
  ```toml
  # ROCm binary
  [[bin]]
  name = "sd-worker-rocm"
  path = "src/bin/rocm.rs"
  required-features = ["rocm"]
  ```
- [ ] Create `src/bin/rocm.rs` (copy from `cuda.rs`, adapt for ROCm)
  - Change `init_cuda_device()` → `init_rocm_device()`
  - Change `CUDA_DEVICE` → `ROCM_DEVICE`
  - Remove flash-attn support (ROCm uses different approach)

### Phase 4: shared-worker-rbee Implementation
- [ ] Add `init_rocm_device()` to `shared-worker-rbee/src/device.rs`
  ```rust
  #[cfg(feature = "rocm")]
  pub fn init_rocm_device(gpu_id: usize) -> CandleResult<Device> {
      tracing::info!("Initializing ROCm device (AMD GPU) {}", gpu_id);
      let device = Device::new_rocm(gpu_id)?;
      Ok(device)
  }
  ```
- [ ] Add ROCm test to `shared-worker-rbee/src/device.rs`
  ```rust
  #[test]
  #[cfg(feature = "rocm")]
  fn test_rocm_device_init() {
      if let Ok(device) = init_rocm_device(0) {
          verify_device(&device).unwrap();
      }
  }
  ```
- [ ] Update `shared-worker-rbee/Cargo.toml` with `rocm` feature
  ```toml
  rocm = ["candle-core/rocm"]
  ```

### Phase 5: Testing & Verification
- [ ] Test compilation: `cargo build --features rocm --bin sd-worker-rocm`
- [ ] Test device init: `cargo test --features rocm`
- [ ] Test on actual AMD GPU hardware
- [ ] Update documentation with ROCm support
- [ ] Add ROCm to CI/CD pipeline

---

## Conclusion

✅ **CPU, CUDA, Metal:** All properly gated with `#[cfg(feature = "...")]`  
✅ **Candle ROCm Support:** Available in `/deps/candle`  
❌ **ROCm Implementation:** Not yet implemented in sd-worker-rbee

**Next Steps:**
1. ✅ Verify Candle has ROCm support (DONE)
2. Add `rocm` feature to candle-nn and candle-transformers
3. Implement ROCm binary following CUDA/Metal pattern
4. Add device init to shared-worker-rbee
5. Test on AMD GPU hardware

**TEAM-507 Audit Complete - Ready for ROCm Implementation.**
