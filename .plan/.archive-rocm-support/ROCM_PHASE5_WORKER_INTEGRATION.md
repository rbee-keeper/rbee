# Phase 5: Worker Integration

**Duration:** Week 7 (5-7 days)  
**Team:** TEAM-488  
**Status:** 📋 READY TO START

---

## Goal

Enable ROCm in LLM and SD workers, allowing end-to-end inference on AMD GPUs.

**Success Criteria:**
- ✅ LLM worker runs on AMD GPU
- ✅ SD worker runs on AMD GPU
- ✅ Backend selection works
- ✅ Performance acceptable
- ✅ No crashes or errors

---

## Prerequisites

### Phase 4 Complete
- ✅ Flash Attention integrated
- ✅ All tests passing

---

## Day 34-35: LLM Worker Integration

### Task 5.1: Add ROCm Feature

**IMPORTANT:** rbee enables Candle's rocm feature, which enables rocm-rs

**Dependency chain:**
```
rbee worker (bin/30_llm_worker_rbee)
    ↓ enables feature "rocm"
Candle (deps/candle/candle-core)
    ↓ enables dependency "rocm-rs"
rocm-rs (deps/rocm-rs)
    ↓ FFI bindings
ROCm libraries (system)
```

**File:** `bin/30_llm_worker_rbee/Cargo.toml`

```toml
[dependencies]
# Use local Candle fork
candle-core = { path = "../../deps/candle/candle-core" }
candle-nn = { path = "../../deps/candle/candle-nn" }
candle-transformers = { path = "../../deps/candle/candle-transformers" }

[features]
# ... existing features ...
rocm = [
    "candle-core/rocm",      # ← Enable Candle's rocm feature
    "candle-nn/rocm",
    "candle-transformers/rocm",
    "candle-flash-attn/rocm",
]

[[bin]]
name = "llm-worker-rbee-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

**What happens:**
1. User runs: `cargo build --bin llm-worker-rbee-rocm --features rocm`
2. rbee enables `candle-core/rocm` feature
3. Candle enables `rocm-rs` dependency
4. rocm-rs provides HIP/rocBLAS/MIOpen bindings

**Checklist:**
- [ ] Added rocm feature (enables Candle's rocm)
- [ ] Added rocm binary
- [ ] Dependencies correct
- [ ] Understand dependency chain

---

### Task 5.2: Create ROCm Binary

**File:** `bin/30_llm_worker_rbee/src/bin/rocm.rs`

```rust
// src/bin/rocm.rs
// Created by: TEAM-488 (Phase 5)
// LLM worker with ROCm backend

use llm_worker_rbee::*;
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Verify ROCm is available
    if !Device::is_rocm_available() {
        anyhow::bail!("ROCm not available on this system");
    }

    let device_count = Device::rocm_device_count()?;
    tracing::info!("Found {} ROCm device(s)", device_count);

    // Parse CLI args
    let args = cli::Args::parse();

    // Create ROCm device
    let device = Device::new_rocm(args.gpu_id)?;
    tracing::info!("Using ROCm device {}", args.gpu_id);

    // Run worker
    run_worker(device, args)?;

    Ok(())
}
```

**Checklist:**
- [ ] Created rocm.rs
- [ ] Device initialization
- [ ] Error handling
- [ ] Compiles

---

### Task 5.3: Update Backend Selection

**File:** `bin/30_llm_worker_rbee/src/backend/mod.rs`

```rust
// src/backend/mod.rs
// Updated by: TEAM-488 (Phase 5)

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Rocm,  // ✅ TEAM-488: Added ROCm
}

impl Backend {
    pub fn device(&self, gpu_id: usize) -> Result<Device> {
        match self {
            Backend::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Device::new_cuda(gpu_id),
            #[cfg(feature = "metal")]
            Backend::Metal => Device::new_metal(gpu_id),
            #[cfg(feature = "rocm")]
            Backend::Rocm => Device::new_rocm(gpu_id),  // ✅ TEAM-488
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!("Backend not available")),
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Backend::Cpu),
            "cuda" => Ok(Backend::Cuda),
            "metal" => Ok(Backend::Metal),
            "rocm" => Ok(Backend::Rocm),  // ✅ TEAM-488
            _ => Err(anyhow::anyhow!("Unknown backend: {}", s)),
        }
    }
}
```

**Checklist:**
- [ ] Added Rocm to Backend enum
- [ ] Implemented device() for ROCm
- [ ] Updated from_str()
- [ ] Compiles

---

### Task 5.4: Test LLM Worker

```bash
cd /home/vince/Projects/rbee

# Build LLM worker with ROCm
cargo build --release --bin llm-worker-rbee-rocm --features rocm

# Test with small model
./target/release/llm-worker-rbee-rocm \
    --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --port 8080 \
    --gpu-id 0

# In another terminal, test inference
curl -X POST http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello, how are you?",
        "max_tokens": 50
    }'
```

**Checklist:**
- [ ] Worker starts successfully
- [ ] Model loads
- [ ] Inference works
- [ ] Output is correct
- [ ] No crashes

---

## Day 36-37: SD Worker Integration

### Task 5.5: Add ROCm Feature to SD Worker

**File:** `bin/31_sd_worker_rbee/Cargo.toml`

First, update to use local Candle fork:

```toml
[dependencies]
# OLD: Using upstream
# candle-core = { git = "https://github.com/huggingface/candle.git" }

# NEW: Using local fork (TEAM-488: Phase 5)
candle-core = { path = "../../deps/candle/candle-core" }
candle-nn = { path = "../../deps/candle/candle-nn" }
candle-transformers = { path = "../../deps/candle/candle-transformers" }

[features]
# ... existing features ...
rocm = [
    "candle-core/rocm",
    "candle-nn/rocm",
    "candle-transformers/rocm",
    "shared-worker-rbee/rocm",
]

[[bin]]
name = "sd-worker-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

**Checklist:**
- [ ] Updated to local Candle fork
- [ ] Added rocm feature
- [ ] Added rocm binary

---

### Task 5.6: Create SD ROCm Binary

**File:** `bin/31_sd_worker_rbee/src/bin/rocm.rs`

```rust
// src/bin/rocm.rs
// Created by: TEAM-488 (Phase 5)
// SD worker with ROCm backend

use sd_worker_rbee::*;
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Verify ROCm is available
    if !Device::is_rocm_available() {
        anyhow::bail!("ROCm not available on this system");
    }

    let device_count = Device::rocm_device_count()?;
    tracing::info!("Found {} ROCm device(s)", device_count);

    // Parse CLI args
    let args = cli::Args::parse();

    // Create ROCm device
    let device = Device::new_rocm(args.gpu_id)?;
    tracing::info!("Using ROCm device {}", args.gpu_id);

    // Run worker
    run_worker(device, args)?;

    Ok(())
}
```

**Checklist:**
- [ ] Created rocm.rs
- [ ] Device initialization
- [ ] Compiles

---

### Task 5.7: Test SD Worker

```bash
cd /home/vince/Projects/rbee

# Build SD worker with ROCm
cargo build --release --bin sd-worker-rocm --features rocm

# Test with Stable Diffusion
./target/release/sd-worker-rocm \
    --model-id "runwayml/stable-diffusion-v1-5" \
    --port 8081 \
    --gpu-id 0

# In another terminal, test image generation
curl -X POST http://localhost:8081/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A beautiful sunset over mountains",
        "n": 1,
        "size": "512x512"
    }' \
    --output image.png
```

**Checklist:**
- [ ] Worker starts successfully
- [ ] Model loads
- [ ] Image generation works
- [ ] Output is correct
- [ ] No crashes

---

## Day 38: Update Shared Worker

### Task 5.8: Add ROCm to Shared Worker

**File:** `bin/32_shared_worker_rbee/Cargo.toml`

```toml
[features]
# ... existing features ...
rocm = ["candle-core/rocm"]
```

**File:** `bin/32_shared_worker_rbee/src/lib.rs`

```rust
// Updated by: TEAM-488 (Phase 5)

#[cfg(feature = "rocm")]
pub fn is_rocm_available() -> bool {
    candle_core::Device::is_rocm_available()
}

#[cfg(feature = "rocm")]
pub fn rocm_device_count() -> Result<usize> {
    candle_core::Device::rocm_device_count()
}
```

**Checklist:**
- [ ] Added rocm feature
- [ ] Added helper functions
- [ ] Compiles

---

## Day 39: Documentation and Scripts

### Task 5.9: Create Deployment Scripts

**File:** `bin/30_llm_worker_rbee/scripts/run_rocm.sh`

```bash
#!/bin/bash
# Created by: TEAM-488 (Phase 5)
# Run LLM worker with ROCm

set -e

# Check ROCm
if ! command -v rocm-smi &> /dev/null; then
    echo "Error: ROCm not found"
    exit 1
fi

# Show GPU info
echo "=== ROCm Devices ==="
rocm-smi
echo ""

# Build
echo "=== Building LLM Worker (ROCm) ==="
cargo build --release --bin llm-worker-rbee-rocm --features rocm

# Run
echo "=== Starting LLM Worker ==="
./target/release/llm-worker-rbee-rocm \
    --model-id "${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --port "${PORT:-8080}" \
    --gpu-id "${GPU_ID:-0}"
```

**File:** `bin/31_sd_worker_rbee/scripts/run_rocm.sh`

```bash
#!/bin/bash
# Created by: TEAM-488 (Phase 5)
# Run SD worker with ROCm

set -e

# Check ROCm
if ! command -v rocm-smi &> /dev/null; then
    echo "Error: ROCm not found"
    exit 1
fi

# Show GPU info
echo "=== ROCm Devices ==="
rocm-smi
echo ""

# Build
echo "=== Building SD Worker (ROCm) ==="
cargo build --release --bin sd-worker-rocm --features rocm

# Run
echo "=== Starting SD Worker ==="
./target/release/sd-worker-rocm \
    --model-id "${MODEL_ID:-runwayml/stable-diffusion-v1-5}" \
    --port "${PORT:-8081}" \
    --gpu-id "${GPU_ID:-0}"
```

```bash
chmod +x bin/*/scripts/run_rocm.sh
```

**Checklist:**
- [ ] Created run scripts
- [ ] Scripts are executable
- [ ] Scripts work

---

### Task 5.10: Update Documentation

**File:** `bin/30_llm_worker_rbee/README.md`

Add ROCm section:

```markdown
## ROCm Support (AMD GPUs)

### Prerequisites

- AMD GPU (MI200, MI300, or RDNA)
- ROCm 6.0+

### Building

```bash
cargo build --release --bin llm-worker-rbee-rocm --features rocm
```

### Running

```bash
./target/release/llm-worker-rbee-rocm \
    --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --port 8080 \
    --gpu-id 0
```

### Performance

- Flash Attention enabled by default
- 2-4x faster than CPU
- 50-75% lower memory usage vs standard attention
```

**File:** `bin/31_sd_worker_rbee/README.md`

Add similar ROCm section.

**Checklist:**
- [ ] Updated LLM worker README
- [ ] Updated SD worker README
- [ ] Examples included

---

## Day 40: End-to-End Testing

### Task 5.11: Comprehensive E2E Tests

**File:** `tests/e2e/rocm_workers.rs`

```rust
// tests/e2e/rocm_workers.rs
// Created by: TEAM-488 (Phase 5)
// End-to-end tests for ROCm workers

#[cfg(feature = "rocm")]
mod rocm_e2e_tests {
    use std::process::{Command, Stdio};
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_llm_worker_e2e() {
        if !candle_core::Device::is_rocm_available() {
            return;
        }

        // Start LLM worker
        let mut worker = Command::new("./target/release/llm-worker-rbee-rocm")
            .args(&[
                "--model-id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--port", "8080",
                "--gpu-id", "0",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start worker");

        // Wait for startup
        sleep(Duration::from_secs(30)).await;

        // Test inference
        let client = reqwest::Client::new();
        let response = client
            .post("http://localhost:8080/v1/completions")
            .json(&serde_json::json!({
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
            }))
            .send()
            .await
            .expect("Failed to send request");

        assert!(response.status().is_success());

        let result: serde_json::Value = response.json().await.unwrap();
        assert!(result["choices"][0]["text"].is_string());

        // Cleanup
        worker.kill().expect("Failed to kill worker");
    }

    #[tokio::test]
    async fn test_sd_worker_e2e() {
        if !candle_core::Device::is_rocm_available() {
            return;
        }

        // Start SD worker
        let mut worker = Command::new("./target/release/sd-worker-rocm")
            .args(&[
                "--model-id", "runwayml/stable-diffusion-v1-5",
                "--port", "8081",
                "--gpu-id", "0",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start worker");

        // Wait for startup
        sleep(Duration::from_secs(60)).await;

        // Test image generation
        let client = reqwest::Client::new();
        let response = client
            .post("http://localhost:8081/v1/images/generations")
            .json(&serde_json::json!({
                "prompt": "A beautiful sunset",
                "n": 1,
                "size": "512x512",
            }))
            .send()
            .await
            .expect("Failed to send request");

        assert!(response.status().is_success());

        // Cleanup
        worker.kill().expect("Failed to kill worker");
    }
}
```

**Run:**
```bash
cargo test --features rocm rocm_e2e_tests
```

**Checklist:**
- [ ] E2E tests created
- [ ] LLM worker test passes
- [ ] SD worker test passes
- [ ] No crashes

---

## Commit and Push

```bash
cd /home/vince/Projects/rbee

git add bin/30_llm_worker_rbee/
git add bin/31_sd_worker_rbee/
git add bin/32_shared_worker_rbee/
git add tests/e2e/

git commit -m "TEAM-488: Phase 5 - Worker integration complete

Integrated ROCm support into LLM and SD workers:

LLM Worker:
- Added rocm feature and binary
- Updated backend selection
- Created deployment scripts
- E2E tests passing

SD Worker:
- Switched to local Candle fork
- Added rocm feature and binary
- Created deployment scripts
- E2E tests passing

Both workers:
- Run successfully on AMD GPUs
- Flash Attention enabled
- Performance acceptable
- No crashes

Ready for Phase 6 (testing & optimization)."

git push
```

---

## Success Criteria Review

At the end of Phase 5, you should have:

- ✅ LLM worker runs on AMD GPU
- ✅ SD worker runs on AMD GPU
- ✅ Backend selection works
- ✅ Performance acceptable
- ✅ No crashes or errors
- ✅ E2E tests passing

---

## Next Phase

**Phase 6: Testing & Optimization**

Document: `ROCM_PHASE6_TESTING_OPTIMIZATION.md`

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📋 PHASE 5 GUIDE
