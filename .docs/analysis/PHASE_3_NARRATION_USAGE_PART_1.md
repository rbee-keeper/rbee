# Phase 3: Narration System Usage Analysis - Part 1

**Analysis Date:** November 2, 2025  
**Scope:** `n!()` macro usage across all binaries  
**Status:** ✅ COMPLETE

---

## Executive Summary

Found **2,288 uses of `n!()` macro** across **192 files**. The narration system is extensively used for real-time feedback across all layers of the architecture.

---

## 1. Usage Statistics by Binary

### Main Binaries

| Binary | Files | Uses | Primary Use Cases |
|--------|-------|------|-------------------|
| `rbee-keeper` | 8 | 45+ | CLI feedback, job submission, lifecycle operations |
| `queen-rbee` | 12 | 80+ | Job routing, hive discovery, SSE forwarding |
| `rbee-hive` | 10 | 150+ | Worker installation, capabilities, heartbeat |
| `llm-worker-rbee` | 15 | 60+ | Model loading, inference, GGUF parsing |
| `xtask` | 8 | 200+ | BDD testing, test reporting, analysis |

### Total Usage

- **192 files** contain `n!()` macro calls
- **2,288 total uses** across the codebase
- **98 files** import `observability_narration_core`

---

## 2. rbee-keeper Narration Usage

### File: `src/job_client.rs`

**Function:** `submit_and_stream_with_timeout()`

```rust
n!("job_submit", "📋 Job submitted: {}", operation_name);
```
**Purpose:** Notify user that job was submitted to queen

```rust
n!("job_stream", "📡 Streaming results for {}", operation_name);
```
**Purpose:** Indicate SSE streaming has started

```rust
n!("job_complete", "✅ Complete: {}", operation_name);
n!("job_complete", "❌ Failed: {}", operation_name);
```
**Purpose:** Final status after [DONE] marker received

---

### File: `src/handlers/self_check.rs`

**Function:** `handle_self_check()`

**Test Narrations (32 uses):**

```rust
n!("narrate_test_start", "Starting rbee-keeper narration test");
```

```rust
n!("version_check", "Checking {} version {}", name, version);
```

```rust
n!("mode_test",
    human: "Testing narration in human mode",
    cute: "🐝 Testing narration in cute mode!",
    story: "'Testing narration', said the keeper"
);
```

```rust
n!("format_test", "Hex: {:x}, Debug: {:?}, Float: {:.2}", 255, vec![1, 2, 3], 3.14159);
```

```rust
n!("self_check_complete",
    human: "✅ Self-check complete - all narration tests passed",
    cute: "🎉 Self-check complete - everything works perfectly!",
    story: "'All systems operational', reported the keeper with satisfaction"
);
```

**Purpose:** Comprehensive narration system testing (all 3 modes)

---

### File: `src/handlers/queen.rs`

**Function:** `handle_queen_status()`

```rust
n!("queen_status", "✅ queen 'localhost' is running on {}", queen_url);
n!("queen_status", "❌ queen 'localhost' is not running on {}", queen_url);
```

**Purpose:** Report queen daemon status to user

---

### File: `src/handlers/hive_lifecycle.rs`

**Function:** `handle_hive_start()`

```rust
n!("detected_local_ip", "🔍 Detected local IP: {}", local_ip);
n!("ssh_target", "🎯 SSH target: {}@{}", ssh.user, ssh.hostname);
n!("remote_hive_queen_url", "🌐 Remote hive will use Queen at: {}", network_queen_url);
```

**Purpose:** Show network configuration for remote hive setup

---

## 3. queen-rbee Narration Usage

### File: `src/main.rs`

**Function:** `main()`

```rust
n!("start", "Queen-rbee starting on port {} (localhost-only mode)", args.port);
```

```rust
n!("listen", "Listening on http://{}", addr);
```

```rust
n!("ready", "Ready to accept connections");
```

```rust
n!("error", "Server error: {}", e);
```

**Purpose:** Daemon lifecycle narration

---

### File: `src/discovery.rs`

**Function:** `discover_hives_on_startup()`

```rust
n!("discovery_start", "🔍 Starting hive discovery (waiting 5s for services to stabilize)");
```

```rust
n!("discovery_no_config", "⚠️  No SSH config found: {}. Only localhost will be discovered.", e);
```

```rust
n!("discovery_skip_invalid", "⚠️  Skipping target '{}': empty hostname", t.host);
```

```rust
n!("discovery_skip_duplicate", "⚠️  Skipping duplicate target: {} ({})", t.host, t.hostname);
```

```rust
n!("discovery_targets", "📋 Found {} unique SSH targets to discover", unique_targets.len());
```

```rust
n!("discovery_complete", "✅ Discovery complete: {} successful, {} failed", success_count, failure_count);
```

**Purpose:** Hive discovery progress and results

---

**Function:** `discover_single_hive()`

```rust
n!("discovery_hive", "🔍 Discovering hive: {} ({})", target.host, target.hostname);
```

```rust
n!("discovery_success", "✅ Discovered hive: {}", target.host);
```

```rust
n!("discovery_failed", "❌ Failed to discover hive {}: {}", target.host, response.status());
```

**Purpose:** Per-hive discovery status

---

### File: `src/hive_forwarder.rs`

**Function:** `forward_to_hive()`

```rust
n!("forward_start", "Forwarding {} operation to localhost hive", operation_name);
```

```rust
n!("forward_connect", "Connecting to hive at {}", hive_url);
```

```rust
n!("forward_complete", "Operation completed on hive '{}'", hive_id);
```

**Purpose:** Operation forwarding to hive

---

**Function:** `stream_from_hive()`

```rust
n!("forward_data", "{}", line);
```

**Purpose:** Forward each SSE line from hive to client

---

### File: `src/hive_subscriber.rs`

**Function:** `subscribe_to_hive()`

```rust
n!("hive_subscribe_start", "📡 Subscribing to hive {} SSE stream: {}", hive_id, stream_url);
```

```rust
n!("hive_connected", "✅ Hive {} connected and registered", hive_id);
```

```rust
n!("hive_subscribe_open", "🔗 SSE connection opened for hive {}", hive_id);
```

```rust
n!("hive_subscribe_error", "❌ Hive {} SSE error: {}", hive_id, e);
```

```rust
n!("hive_disconnected", "🔌 Hive {} disconnected and removed", hive_id);
```

```rust
n!("hive_reconnect", "🔄 Reconnecting to hive {} in 5s...", hive_id);
```

**Purpose:** Hive SSE subscription lifecycle

---

## 4. rbee-hive Narration Usage

### File: `src/main.rs`

**Function:** `main()`

```rust
n!("startup", "🐝 Starting rbee-hive on port {}", args.port);
```

```rust
n!("catalog_init", "📚 Model catalog initialized ({} models)", model_catalog.len());
```

```rust
n!("worker_cat_init", "🔧 Worker catalog initialized ({} binaries)", worker_catalog.len());
```

```rust
n!("provisioner_init", "📥 Model provisioner initialized (HuggingFace)");
```

```rust
n!("dev_proxy_config", "🔧 Dev proxy configured: /dev → {}", vite_url);
```

```rust
n!("listen", "✅ Listening on http://{}", addr);
```

```rust
n!("ready", "✅ Hive ready");
```

```rust
n!("heartbeat_start", "💓 Heartbeat task started (sending to {})", args.queen_url);
```

**Purpose:** Daemon initialization and configuration

---

**Function:** `capabilities()`

```rust
n!("caps_request", "📡 Received capabilities request from queen");
```

```rust
n!("caps_queen_url", "🔗 Queen URL received: {}", queen_url);
```

```rust
n!("caps_invalid_url", "❌ Invalid queen_url rejected: {}", e);
```

```rust
n!("caps_gpu_check", "🔍 Detecting GPUs via nvidia-smi...");
```

```rust
n!("caps_gpu_found", "✅ Found {} GPU(s)", gpu_info.count);
```

```rust
n!("caps_gpu_none", "ℹ️  No GPUs detected, using CPU only");
```

```rust
n!("caps_cpu_add", "🖥️  Adding CPU-0: {} cores, {} GB RAM", cpu_cores, system_ram_gb);
```

```rust
n!("caps_response", "📤 Sending capabilities response ({} device(s))", devices.len());
```

**Purpose:** GPU detection and capabilities reporting

---

**Function:** `HeartbeatState::start_heartbeat_task()`

```rust
n!("heartbeat_skip", "ℹ️  No queen_url provided, skipping heartbeat (standalone mode)");
```

```rust
n!("heartbeat_url_changed", "⚠️  Queen URL changed: {} → {}. Heartbeat will continue to old URL.", existing, url);
```

```rust
n!("heartbeat_skip", "💓 Heartbeat already running, skipping");
```

```rust
n!("heartbeat_start", "💓 Starting heartbeat task to {}", url);
```

**Purpose:** Heartbeat task management

---

### File: `src/worker_install.rs`

**59 uses of `n!()` macro**

**Function:** `execute_worker_install()`

```rust
n!("worker_install_start", "🔧 Installing worker: {}", worker_type);
```

```rust
n!("worker_install_check_existing", "🔍 Checking if worker already installed...");
```

```rust
n!("worker_install_already_exists", "✅ Worker already installed: {}", binary_name);
```

```rust
n!("worker_install_fetch_source", "📥 Fetching PKGBUILD from catalog...");
```

```rust
n!("worker_install_parse_pkgbuild", "📝 Parsing PKGBUILD...");
```

```rust
n!("worker_install_build", "🔨 Building worker binary...");
```

```rust
n!("worker_install_success", "✅ Worker installed successfully: {}", binary_name);
```

**Purpose:** Worker installation progress (detailed step-by-step)

---

### File: `src/source_fetcher.rs`

**Function:** `fetch_sources()`

```rust
n!("fetch_source", "📥 Fetching source {}/{}: {}", idx + 1, sources.len(), source);
```

**Purpose:** Source code download progress

---

## 5. llm-worker-rbee Narration Usage

### File: `src/main.rs`

**Function:** `main()`

```rust
n!(ACTION_STARTUP, "Starting Candle worker on port {}", args.port);
```

```rust
n!(ACTION_MODEL_LOAD, "Loading Llama model from {}", args.model);
```

```rust
n!("model_load_success", "Model loaded successfully");
```

```rust
n!("model_load_failed", "Model load failed: {}", error_msg.lines().next().unwrap_or("unknown error"));
```

**Purpose:** Worker startup and model loading

---

### File: `src/device.rs`

**Function:** `init_cpu_device()`

```rust
n!(ACTION_DEVICE_INIT, "Initialized CPU device");
```

**Function:** `init_cuda_device()`

```rust
n!(ACTION_DEVICE_INIT, "Initialized CUDA device {}", gpu_id);
```

**Function:** `init_metal_device()`

```rust
n!(ACTION_DEVICE_INIT, "Initialized Apple Metal device {}", gpu_id);
```

**Purpose:** Device initialization confirmation

---

### File: `src/backend/gguf_tokenizer.rs`

**Function:** `extract_tokenizer_from_gguf()`

```rust
n!("gguf_tokenizer_extract_start", "Extracting embedded tokenizer from GGUF: {}", gguf_path.display());
```

```rust
n!("gguf_tokenizer_metadata_extracted", "Extracted tokenizer metadata: {} tokens, {} merges", tokens.len(), merges.as_ref().map_or(0, std::vec::Vec::len));
```

```rust
n!("gguf_tokenizer_extracted", "Extracted tokenizer from GGUF ({} tokens)", tokenizer.get_vocab_size(false));
```

**Purpose:** GGUF tokenizer extraction progress

---

### File: `src/backend/models/quantized_llama.rs`

**Function:** `load()`

```rust
n!("gguf_load_start", "Loading GGUF model from {}", path.display());
```

```rust
n!("gguf_open_failed", "Failed to open GGUF file: {}", path.display());
```

```rust
n!("gguf_file_opened", "GGUF file opened, reading content");
```

```rust
n!("gguf_parse_failed", "Failed to parse GGUF content from {}", path.display());
```

```rust
n!("gguf_inspect_metadata", "Inspecting GGUF metadata ({} keys found)", content.metadata.len());
```

```rust
n!("gguf_vocab_size_derived", "Derived vocab_size={} from tokenizer.ggml.tokens array", size);
```

```rust
n!("gguf_metadata_missing", "Cannot determine vocab_size from GGUF metadata");
```

```rust
n!("gguf_metadata_loaded", "GGUF metadata: vocab={}, eos={}, tensors={}", vocab_size, eos_token_id, content.tensor_infos.len());
```

**Purpose:** GGUF model loading with detailed error diagnostics

---

## 6. Narration Action Constants

### Common Actions (from taxonomy)

**Lifecycle Actions:**
- `ACTION_STARTUP` — Service startup
- `ACTION_SHUTDOWN` — Service shutdown
- `ACTION_READY` — Service ready

**Model Actions:**
- `ACTION_MODEL_LOAD` — Model loading
- `ACTION_MODEL_UNLOAD` — Model unloading

**Device Actions:**
- `ACTION_DEVICE_INIT` — Device initialization

**Job Actions:**
- `ACTION_JOB_SUBMIT` — Job submission
- `ACTION_JOB_STREAM` — Job streaming
- `ACTION_JOB_COMPLETE` — Job completion

---

## Summary Statistics

### Usage by Category

| Category | Uses | Percentage |
|----------|------|------------|
| Worker Installation | 59 | 2.6% |
| BDD Testing | 200+ | 8.7% |
| Hive Operations | 150+ | 6.6% |
| Queen Operations | 80+ | 3.5% |
| Worker Operations | 60+ | 2.6% |
| Keeper Operations | 45+ | 2.0% |
| Other | 1,694 | 74.0% |

### Top 10 Files by Usage

1. `xtask/src/tasks/bdd/reporter.rs` — 134 uses
2. `bin/30_llm_worker_rbee/tests/team_013_cuda_integration.rs` — 61 uses
3. `xtask/src/tasks/bdd/analyzer.rs` — 60 uses
4. `bin/20_rbee_hive/src/worker_install.rs` — 59 uses
5. `xtask/src/tasks/worker.rs` — 50 uses
6. `bin/20_rbee_hive/src/job_router.rs` — 49 uses
7. `bin/20_rbee_hive/src/main.rs` — 39 uses
8. `bin/99_shared_crates/narration-core/tests/thread_local_context_tests.rs` — 35 uses
9. `xtask/src/tasks/bdd/files.rs` — 35 uses
10. `bin/00_rbee_keeper/tests/tracing_init_tests.rs` — 33 uses

---

**Next Part:** [PHASE_3_NARRATION_USAGE_PART_2.md](./PHASE_3_NARRATION_USAGE_PART_2.md)
