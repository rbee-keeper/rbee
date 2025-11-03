# shared-worker-rbee

**Shared utilities for all worker types (LLM, SD, etc.)**

Created by: **TEAM-XXX**

---

## Overview

`shared-worker-rbee` provides common infrastructure shared between all worker types:
- `llm-worker-rbee` (LLM inference)
- `sd-worker-rbee` (Stable Diffusion image generation)
- Future workers (audio, video, etc.)

## What's Shared

### Device Management
- Multi-backend device initialization (CPU/CUDA/Metal)
- Device verification and smoke tests
- Feature-gated compilation for different backends

### Heartbeat System
- Worker-to-queen heartbeat protocol
- Periodic heartbeat task spawning
- Worker health status reporting

### Worker Patterns
- Common worker initialization
- Worker info structures
- Lifecycle management patterns

## Architecture

```
┌─────────────────────────────────────────┐
│      shared-worker-rbee (this crate)     │
├─────────────────────────────────────────┤
│  Device Management                      │
│  ├─ init_cpu_device()                   │
│  ├─ init_cuda_device(gpu_id)            │
│  ├─ init_metal_device(gpu_id)           │
│  └─ verify_device(device)               │
├─────────────────────────────────────────┤
│  Heartbeat System                       │
│  ├─ send_heartbeat_to_queen()           │
│  └─ start_heartbeat_task()              │
├─────────────────────────────────────────┤
│  Worker Patterns                        │
│  ├─ WorkerInfo                          │
│  └─ Common initialization helpers       │
└─────────────────────────────────────────┘
         ▲                    ▲
         │                    │
    ┌────┴────┐          ┌────┴────┐
    │   LLM   │          │   SD    │
    │ Worker  │          │ Worker  │
    └─────────┘          └─────────┘
```

## Usage

### Device Management

```rust
use shared_worker_rbee::device;

// CPU
let device = device::init_cpu_device()?;
device::verify_device(&device)?;

// CUDA
#[cfg(feature = "cuda")]
let device = device::init_cuda_device(0)?;

// Metal
#[cfg(feature = "metal")]
let device = device::init_metal_device(0)?;
```

### Heartbeat

```rust
use shared_worker_rbee::heartbeat;
use worker_contract::WorkerInfo;

let worker_info = WorkerInfo {
    worker_id: "llm-worker-1".to_string(),
    worker_type: "llm".to_string(),
    // ... other fields
};

// Start periodic heartbeat task
let heartbeat_handle = heartbeat::start_heartbeat_task(
    worker_info,
    "http://localhost:8500".to_string(),
);

// Heartbeat runs every 30s in background
```

## Features

### Backend Selection (Mutually Exclusive)

```bash
# CPU only
cargo build --features cpu

# CUDA (NVIDIA GPU)
cargo build --features cuda

# CUDA with cuDNN
cargo build --features cudnn

# Metal (Apple Silicon)
cargo build --features metal
```

## Dependencies

### Core
- `candle-core` - Device management (feature-gated)
- `observability-narration-core` - Logging
- `worker-contract` - Worker types

### Runtime
- `tokio` - Async runtime for heartbeat
- `anyhow`, `thiserror` - Error handling

## Extracted From

This crate consolidates code that was duplicated between:
- `bin/30_llm_worker_rbee/src/device.rs` → `src/device.rs`
- `bin/30_llm_worker_rbee/src/heartbeat.rs` → `src/heartbeat.rs`

## Benefits

✅ **Single source of truth** - Device and heartbeat logic in one place  
✅ **No duplication** - LLM and SD workers share the same code  
✅ **Consistent behavior** - All workers use the same patterns  
✅ **Easier maintenance** - Fix bugs once, all workers benefit  
✅ **Smaller binaries** - Shared code reduces binary size  

## Integration

### LLM Worker

```toml
# bin/30_llm_worker_rbee/Cargo.toml
[dependencies]
shared-worker-rbee = { path = "../32_shared_worker_rbee" }
```

```rust
// bin/30_llm_worker_rbee/src/bin/cpu.rs
use shared_worker_rbee::device;

let device = device::init_cpu_device()?;
```

### SD Worker

```toml
# bin/31_sd_worker_rbee/Cargo.toml
[dependencies]
shared-worker-rbee = { path = "../32_shared_worker_rbee" }
```

```rust
// bin/31_sd_worker_rbee/src/bin/cuda.rs
use shared_worker_rbee::device;

let device = device::init_cuda_device(args.cuda_device)?;
```

## Testing

```bash
# Run all tests (CPU)
cargo test --features cpu

# Run CUDA tests (requires GPU)
cargo test --features cuda

# Run Metal tests (requires macOS)
cargo test --features metal
```

## License

GPL-3.0-or-later

---

**Status**: 🚧 In Development  
**Version**: 0.1.0  
**Last Updated**: 2025-11-03
