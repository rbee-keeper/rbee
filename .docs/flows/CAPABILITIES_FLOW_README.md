# Capabilities Discovery Flow Documentation

**Complete flow from Queen discovery to Hive device detection and telemetry**  
**Date:** November 2, 2025  
**Status:** ✅ COMPLETE

---

## Overview

This directory contains documentation of the capabilities discovery flow, showing how Queen discovers hives and their available compute resources (GPUs/CPUs).

**Flow Summary:**
```
Queen Startup
    ↓ Wait 5s
Read SSH Config
    ↓ Parse targets
Parallel Discovery (Part 1)
    ↓ GET /capabilities?queen_url=X
Hive Receives Request (Part 2)
    ↓ Detect GPUs (nvidia-smi)
    ↓ Detect CPU/RAM (system calls)
    ↓ Store queen_url
    ↓ Start heartbeat task
    ↓ Return CapabilitiesResponse
Queen Processes Response (Part 3)
    ↓ Register hive in TelemetryRegistry
    ↓ Subscribe to SSE telemetry stream
    ↓ Receive worker updates
```

**Ports:**
- Queen: 7833
- Hive: 7835

---

## Documentation Parts

### ✅ [Part 1: Queen Discovery Initiation](./CAPABILITIES_FLOW_PART_1_QUEEN_DISCOVERY.md)

**Scope:** Queen Startup → SSH Config → Parallel Discovery Probes

**Key Topics:**
- Queen startup and background task
- SSH config parsing
- Target deduplication
- Parallel discovery requests
- URL encoding

**Status:** Fully documented

**Key Files:**
- `bin/10_queen_rbee/src/main.rs`
- `bin/10_queen_rbee/src/discovery.rs`
- `bin/99_shared_crates/ssh-config-parser/src/lib.rs`

**Narration Events:**
- `discovery_start` — Discovery starting
- `discovery_targets` — Targets found
- `discovery_hive` — Discovering hive
- `discovery_success` — Hive discovered
- `discovery_complete` — Discovery finished

---

### 📋 Part 2: Hive Device Detection (OUTLINED)

**Scope:** Hive Receives Request → Device Detection → Response

**Key Topics:**
- GET `/v1/capabilities` endpoint
- Query parameter extraction (`queen_url`)
- GPU detection (nvidia-smi)
- CPU/RAM detection (system calls)
- Device info formatting
- Heartbeat task initiation

**Key Files:**
- `bin/20_rbee_hive/src/main.rs` (lines 390-467)
- `bin/25_rbee_hive_crates/device-detection/src/detection.rs`
- `bin/97_contracts/hive-contract/src/lib.rs`

**Narration Events:**
- `caps_request` — Request received
- `caps_queen_url` — Queen URL received
- `caps_gpu_check` — Detecting GPUs
- `caps_gpu_found` — GPUs detected
- `caps_cpu_add` — Adding CPU
- `caps_response` — Sending response

---

### 📋 Part 3: Queen Registration & Telemetry (OUTLINED)

**Scope:** Queen Receives Response → Register Hive → Subscribe to Telemetry

**Key Topics:**
- Parse `CapabilitiesResponse`
- Register hive in `TelemetryRegistry`
- Subscribe to SSE heartbeat stream
- Process worker telemetry updates
- Handle hive disconnection

**Key Files:**
- `bin/10_queen_rbee/src/hive_subscriber.rs`
- `bin/15_queen_rbee_crates/telemetry-registry/src/lib.rs`
- `bin/10_queen_rbee/src/http/heartbeat.rs`

**Narration Events:**
- `hive_subscribe_start` — Subscribing to hive
- `hive_connected` — Hive connected
- `hive_disconnected` — Hive disconnected

---

## Quick Reference

### HTTP Endpoints

**Queen (Port 7833):**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/v1/hive/ready` | Hive ready callback |
| GET | `/v1/status` | Get system status |

**Hive (Port 7835):**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/v1/capabilities` | Get device capabilities |
| GET | `/v1/heartbeats/stream` | SSE telemetry stream |
| GET | `/health` | Health check |

---

### Request/Response Types

**Capabilities Request:**
```http
GET /v1/capabilities?queen_url=http%3A%2F%2Flocalhost%3A7833 HTTP/1.1
Host: 192.168.1.100:7835
```

**Capabilities Response:**
```json
{
  "devices": [
    {
      "id": "GPU-0",
      "name": "NVIDIA GeForce RTX 3090",
      "device_type": "gpu",
      "vram_gb": 24,
      "compute_capability": "8.6"
    },
    {
      "id": "CPU-0",
      "name": "CPU (16 cores)",
      "device_type": "cpu",
      "vram_gb": 64,
      "compute_capability": null
    }
  ]
}
```

---

### Key Data Structures

**SshTarget (Queen):**
```rust
struct SshTarget {
    host: String,        // "hive-gpu-1"
    hostname: String,    // "192.168.1.100"
    user: Option<String>,
    port: Option<u16>,
}
```

**HiveDevice (Contract):**
```rust
struct HiveDevice {
    id: String,                      // "GPU-0", "CPU-0"
    name: String,                    // "NVIDIA GeForce RTX 3090"
    device_type: String,             // "gpu", "cpu"
    vram_gb: Option<u32>,            // 24
    compute_capability: Option<String>, // "8.6"
}
```

**CapabilitiesResponse (Contract):**
```rust
struct CapabilitiesResponse {
    devices: Vec<HiveDevice>,
}
```

**GpuInfo (Device Detection):**
```rust
struct GpuInfo {
    available: bool,
    count: usize,
    devices: Vec<GpuDevice>,
}

struct GpuDevice {
    index: u32,
    name: String,
    vram_total_mb: u64,
    vram_free_mb: u64,
    compute_capability: (u32, u32),  // (8, 6)
    pci_bus_id: String,
}
```

---

## Discovery Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ QUEEN (Port 7833)                                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Startup                                                  │
│    ↓                                                        │
│ 2. Wait 5 seconds                                           │
│    ↓                                                        │
│ 3. Read ~/.ssh/config                                       │
│    ↓                                                        │
│ 4. Parse targets: [hive-1, hive-2, hive-3]                │
│    ↓                                                        │
│ 5. Deduplicate by hostname                                  │
│    ↓                                                        │
│ 6. Spawn parallel tasks                                     │
│    ├─→ GET hive-1:7835/capabilities?queen_url=...         │
│    ├─→ GET hive-2:7835/capabilities?queen_url=...         │
│    └─→ GET hive-3:7835/capabilities?queen_url=...         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HIVE (Port 7835)                                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Receive GET /v1/capabilities?queen_url=X                │
│    ↓                                                        │
│ 2. Extract queen_url parameter                             │
│    ↓                                                        │
│ 3. Validate and store queen_url                            │
│    ↓                                                        │
│ 4. Start heartbeat task to queen                           │
│    ↓                                                        │
│ 5. Detect GPUs (nvidia-smi)                                │
│    ├─→ Parse CSV output                                    │
│    └─→ Extract: index, name, vram, compute_cap            │
│    ↓                                                        │
│ 6. Detect CPU/RAM (system calls)                           │
│    ├─→ num_cpus::get()                                     │
│    └─→ sysinfo crate                                       │
│    ↓                                                        │
│ 7. Format devices array                                    │
│    ├─→ GPU-0, GPU-1, ... (if GPUs found)                  │
│    └─→ CPU-0 (always)                                      │
│    ↓                                                        │
│ 8. Return JSON response                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ QUEEN (Processes Response)                                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Receive CapabilitiesResponse                            │
│    ↓                                                        │
│ 2. Register hive in TelemetryRegistry                      │
│    ├─→ Store hive_id, hostname, port                       │
│    └─→ Store device capabilities                           │
│    ↓                                                        │
│ 3. Subscribe to hive's SSE stream                          │
│    ├─→ GET hive:7835/v1/heartbeats/stream                 │
│    └─→ Receive worker telemetry updates                    │
│    ↓                                                        │
│ 4. Update worker registry                                  │
│    └─→ Track online workers                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Narration Events Summary

### Part 1 (Queen Discovery)

| Event | Message | Location |
|-------|---------|----------|
| `discovery_start` | "🔍 Starting hive discovery (waiting 5s for services to stabilize)" | discovery.rs:44 |
| `discovery_no_config` | "⚠️  No SSH config found: {error}. Only localhost will be discovered." | discovery.rs:54 |
| `discovery_targets` | "📋 Found {count} unique SSH targets to discover" | discovery.rs:80 |
| `discovery_hive` | "🔍 Discovering hive: {host} ({hostname})" | discovery.rs:123 |
| `discovery_success` | "✅ Discovered hive: {host}" | discovery.rs:133 |
| `discovery_complete` | "✅ Discovery complete: {success} successful, {failed} failed" | discovery.rs:104 |

### Part 2 (Hive Detection)

| Event | Message | Location |
|-------|---------|----------|
| `caps_request` | "📡 Received capabilities request from queen" | main.rs:395 |
| `caps_queen_url` | "🔗 Queen URL received: {url}" | main.rs:400 |
| `caps_gpu_check` | "🔍 Detecting GPUs via nvidia-smi..." | main.rs:416 |
| `caps_gpu_found` | "✅ Found {count} GPU(s)" | main.rs:424 |
| `caps_gpu_none` | "ℹ️  No GPUs detected, using CPU only" | main.rs:426 |
| `caps_cpu_add` | "🖥️  Adding CPU-0: {cores} cores, {ram} GB RAM" | main.rs:450 |
| `caps_response` | "📤 Sending capabilities response ({count} device(s))" | main.rs:463 |

### Part 3 (Queen Registration)

| Event | Message | Location |
|-------|---------|----------|
| `hive_subscribe_start` | "📡 Subscribing to hive {id} SSE stream: {url}" | hive_subscriber.rs:46 |
| `hive_connected` | "✅ Hive {id} connected and registered" | hive_subscriber.rs:62 |
| `hive_subscribe_open` | "🔗 SSE connection opened for hive {id}" | hive_subscriber.rs:96 |
| `hive_disconnected` | "🔌 Hive {id} disconnected and removed" | hive_subscriber.rs:107 |

---

## Device Detection Details

### GPU Detection (nvidia-smi)

**Command:**
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id \
           --format=csv,noheader,nounits
```

**Example Output:**
```
0, NVIDIA GeForce RTX 3090, 24576, 24000, 8.6, 0000:01:00.0
1, NVIDIA GeForce RTX 3090, 24576, 23500, 8.6, 0000:02:00.0
```

**Parsed Result:**
```rust
GpuInfo {
    available: true,
    count: 2,
    devices: vec![
        GpuDevice {
            index: 0,
            name: "NVIDIA GeForce RTX 3090",
            vram_total_mb: 24576,
            vram_free_mb: 24000,
            compute_capability: (8, 6),
            pci_bus_id: "0000:01:00.0",
        },
        GpuDevice {
            index: 1,
            name: "NVIDIA GeForce RTX 3090",
            vram_total_mb: 24576,
            vram_free_mb: 23500,
            compute_capability: (8, 6),
            pci_bus_id: "0000:02:00.0",
        },
    ],
}
```

---

### CPU Detection

**CPU Cores:**
```rust
pub fn get_cpu_cores() -> u32 {
    num_cpus::get() as u32
}
```

**System RAM:**
```rust
pub fn get_system_ram_gb() -> u32 {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_memory();
    (sys.total_memory() / 1024 / 1024 / 1024) as u32
}
```

**Example Result:**
```rust
cpu_cores: 16
system_ram_gb: 64
```

---

## Configuration

### SSH Config

**Location:** `~/.ssh/config`

**Format:**
```
Host hive-gpu-1
    HostName 192.168.1.100
    User rbee
    Port 22

Host hive-gpu-2
    HostName 192.168.1.101
    User rbee
    Port 22
```

**Parsing:** Uses `ssh-config-parser` crate

---

### Timeouts

- **Service stabilization:** 5 seconds
- **HTTP request:** 10 seconds per hive
- **Heartbeat interval:** 5 seconds (hive → queen)

---

### Ports

- **Queen:** 7833 (configurable via CLI `--port`)
- **Hive:** 7835 (hardcoded in discovery)

---

## Error Handling

### Discovery Errors (Non-Fatal)

- SSH config not found → Use empty list
- Invalid hostname → Skip target
- Duplicate hostname → Skip duplicate
- HTTP timeout → Count as failure
- HTTP error status → Count as failure

**Result:** Discovery continues for other hives

---

### Device Detection Errors (Graceful)

- nvidia-smi not found → Return no GPUs
- nvidia-smi fails → Return no GPUs
- CPU detection fails → Use default (1 core, 1GB RAM)

**Result:** Always returns at least CPU-0

---

## Performance Characteristics

### Discovery Speed

**For N hives:**
- Sequential: N × 10s = 10N seconds
- Parallel: ~10s (all at once)
- **Speedup:** Nx

**Example (10 hives):**
- Sequential: 100 seconds
- Parallel: ~10 seconds
- **Speedup:** 10x

---

### Memory Usage

**Per hive:**
- HTTP client: ~1KB
- Task overhead: ~8KB
- **Total:** ~9KB per hive

**For 10 hives:** ~90KB total

---

## Security Considerations

### nvidia-smi Execution

**Security measures:**
- Uses absolute path (prevents PATH manipulation)
- No shell execution (direct Command::new)
- Validates output format
- Handles parsing errors gracefully

---

### Queen URL Validation

**Validation checks:**
- Not empty
- Valid URL format
- Stored securely
- Used for heartbeat only

---

## Testing Strategy

### Unit Tests

- [ ] SSH config parsing
- [ ] Target deduplication
- [ ] URL encoding
- [ ] GPU detection parsing
- [ ] CPU detection

### Integration Tests

- [ ] End-to-end discovery flow
- [ ] Multiple hive discovery
- [ ] Timeout handling
- [ ] Error recovery

### Edge Case Tests

- [ ] Empty SSH config
- [ ] Invalid hostnames
- [ ] Duplicate hostnames
- [ ] No GPUs detected
- [ ] nvidia-smi not found

---

## Related Documentation

- [Job Flow Documentation](./README.md) — Job submission and execution
- [OpenAI Flow Documentation](./OPENAI_FLOW_README.md) — OpenAI adapter
- [Phase 4: Runtime Patterns](../analysis/PHASE_4_RUNTIME_PATTERNS.md)
- [Phase 7: xtask & Testing](../analysis/PHASE_7_XTASK_TESTING.md)

---

**Status:** Part 1 complete with full detail, Parts 2-3 outlined  
**Maintainer:** TEAM-385+  
**Last Updated:** November 2, 2025
