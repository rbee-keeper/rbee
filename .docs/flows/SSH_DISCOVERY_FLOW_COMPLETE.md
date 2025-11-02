# SSH Discovery Flow: Complete Roundtrip

**Flow:** Queen SSH Config → Parallel Discovery → Hive Health Check → Summary  
**Date:** November 2, 2025  
**Status:** ✅ COMPLETE

---

## Overview

This document traces the complete flow from when Queen reads SSH targets, sends parallel health checks to hives, validates their responses, and returns a discovery summary.

**Pattern:** Pull-based discovery with parallel execution

**Ports:**
- Queen: 7833
- Hive: 7835

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ QUEEN STARTUP                                               │
├─────────────────────────────────────────────────────────────┤
│ main() spawns background task                               │
│    ↓                                                        │
│ discover_hives_on_startup()                                 │
│    ↓                                                        │
│ Wait 5 seconds for services                                 │
│    ↓                                                        │
│ Read ~/.ssh/config                                          │
│    ↓                                                        │
│ parse_ssh_config()                                          │
│    ├─→ Extract Host entries                                │
│    ├─→ Parse HostName, User, Port                          │
│    └─→ Return Vec<SshTarget>                               │
│    ↓                                                        │
│ Deduplicate by hostname                                     │
│    ├─→ Skip empty hostnames                                │
│    ├─→ Skip duplicate hostnames                            │
│    └─→ Return unique targets                               │
│    ↓                                                        │
│ Spawn parallel tasks                                        │
│    ├─→ Task 1: discover_single_hive(target1)              │
│    ├─→ Task 2: discover_single_hive(target2)              │
│    └─→ Task 3: discover_single_hive(target3)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ PARALLEL DISCOVERY (Per Target)                            │
├─────────────────────────────────────────────────────────────┤
│ discover_single_hive(target, queen_url)                    │
│    ↓                                                        │
│ Construct URL:                                              │
│    http://{hostname}:7835/capabilities?queen_url={encoded} │
│    ↓                                                        │
│ Send GET request (10s timeout)                             │
│    ↓                                                        │
│ Check response status                                       │
│    ├─→ Success (200) → Log success                         │
│    └─→ Error → Log failure, return error                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HIVE HEALTH CHECK                                           │
├─────────────────────────────────────────────────────────────┤
│ GET /v1/capabilities handler                               │
│    ↓                                                        │
│ Extract queen_url parameter                                 │
│    ↓                                                        │
│ Validate and store queen_url                               │
│    ↓                                                        │
│ Start heartbeat task                                        │
│    ↓                                                        │
│ Detect GPUs (nvidia-smi)                                   │
│    ├─→ Parse CSV output                                    │
│    └─→ Create GPU devices                                  │
│    ↓                                                        │
│ Detect CPU/RAM (system calls)                              │
│    ├─→ Get CPU cores                                       │
│    └─→ Get system RAM                                      │
│    ↓                                                        │
│ Build CapabilitiesResponse                                  │
│    └─→ { devices: [GPU-0, GPU-1, CPU-0] }                 │
│    ↓                                                        │
│ Return JSON (200 OK)                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ QUEEN PROCESSES RESULTS                                     │
├─────────────────────────────────────────────────────────────┤
│ Await all parallel tasks                                    │
│    ├─→ Task 1: Ok(Ok(())) → success_count++               │
│    ├─→ Task 2: Ok(Err(_)) → failure_count++               │
│    └─→ Task 3: Err(_) → failure_count++                   │
│    ↓                                                        │
│ Emit discovery_complete event                               │
│    "✅ Discovery complete: X successful, Y failed"         │
│    ↓                                                        │
│ Return Ok(()) or Err(...)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Queen Startup

**File:** `bin/10_queen_rbee/src/main.rs`

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // ... server initialization ...
    
    // TEAM-365: Start hive discovery in background
    let queen_url = format!("http://localhost:{}", args.port);
    tokio::spawn(async move {
        if let Err(e) = discovery::discover_hives_on_startup(&queen_url).await {
            n!("discovery_error", "❌ Hive discovery failed: {}", e);
        }
    });
    
    // ... start HTTP server ...
}
```

**Location:** Lines 86-91  
**Narration:** `discovery_error` (if discovery fails)

---

### Step 2: Discovery Initialization

**File:** `bin/10_queen_rbee/src/discovery.rs`

```rust
pub async fn discover_hives_on_startup(queen_url: &str) -> Result<()> {
    // Validate queen_url
    if queen_url.is_empty() {
        n!("discovery_invalid_url", "❌ Cannot start discovery: empty queen_url");
        anyhow::bail!("Cannot start discovery with empty queen_url");
    }
    
    if let Err(e) = url::Url::parse(queen_url) {
        n!("discovery_invalid_url", "❌ Cannot start discovery: invalid queen_url '{}': {}", queen_url, e);
        anyhow::bail!("Invalid queen_url '{}': {}", queen_url, e);
    }
    
    n!("discovery_start", "🔍 Starting hive discovery (waiting 5s for services to stabilize)");
    
    // Wait for services to stabilize
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Continue to SSH config parsing...
}
```

**Location:** Lines 32-47  
**Narration:**
- `discovery_invalid_url` — Invalid queen URL
- `discovery_start` — Discovery starting

---

### Step 3: SSH Config Parsing

**File:** `bin/10_queen_rbee/src/discovery.rs`

```rust
// Read SSH config
let ssh_config_path = get_default_ssh_config_path();
let targets = match parse_ssh_config(&ssh_config_path) {
    Ok(targets) => targets,
    Err(e) => {
        n!("discovery_no_config", "⚠️  No SSH config found: {}. Only localhost will be discovered.", e);
        vec![]
    }
};
```

**Location:** Lines 49-57  
**Narration:** `discovery_no_config` (if config not found)

---

**SSH Config Parser:**

**File:** `bin/99_shared_crates/ssh-config-parser/src/lib.rs`

```rust
/// Parse SSH config file and extract targets
///
/// # Format
/// ```
/// Host hive-gpu-1
///     HostName 192.168.1.100
///     User rbee
///     Port 22
/// ```
pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>> {
    let content = std::fs::read_to_string(path)?;
    let mut targets = Vec::new();
    let mut current_host: Option<String> = None;
    let mut current_hostname: Option<String> = None;
    let mut current_user: Option<String> = None;
    let mut current_port: Option<u16> = None;
    
    for line in content.lines() {
        let line = line.trim();
        
        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        // Parse key-value pairs
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        
        match parts[0].to_lowercase().as_str() {
            "host" => {
                // Save previous host if exists
                if let (Some(host), Some(hostname)) = (current_host.take(), current_hostname.take()) {
                    targets.push(SshTarget {
                        host,
                        hostname,
                        user: current_user.take(),
                        port: current_port.take(),
                    });
                }
                
                // Start new host
                current_host = Some(parts[1].to_string());
            }
            "hostname" => {
                current_hostname = Some(parts[1].to_string());
            }
            "user" => {
                current_user = Some(parts[1].to_string());
            }
            "port" => {
                current_port = parts[1].parse().ok();
            }
            _ => {}
        }
    }
    
    // Save last host
    if let (Some(host), Some(hostname)) = (current_host, current_hostname) {
        targets.push(SshTarget {
            host,
            hostname,
            user: current_user,
            port: current_port,
        });
    }
    
    Ok(targets)
}
```

**Location:** Lines 64-120  
**Function:** `parse_ssh_config()`

**SshTarget Type:**
```rust
#[derive(Debug, Clone)]
pub struct SshTarget {
    pub host: String,        // "hive-gpu-1"
    pub hostname: String,    // "192.168.1.100"
    pub user: Option<String>,
    pub port: Option<u16>,
}
```

---

### Step 4: Target Deduplication

**File:** `bin/10_queen_rbee/src/discovery.rs`

```rust
// Deduplicate targets by hostname
let mut seen = HashSet::new();
let unique_targets: Vec<_> = targets
    .into_iter()
    .filter(|t| {
        // Skip invalid hostnames
        if t.hostname.is_empty() {
            n!("discovery_skip_invalid", "⚠️  Skipping target '{}': empty hostname", t.host);
            return false;
        }
        
        // Skip duplicates
        if !seen.insert(t.hostname.clone()) {
            n!("discovery_skip_duplicate", "⚠️  Skipping duplicate target: {} ({})", t.host, t.hostname);
            return false;
        }
        
        true
    })
    .collect();

n!("discovery_targets", "📋 Found {} unique SSH targets to discover", unique_targets.len());
```

**Location:** Lines 59-80  
**Narration:**
- `discovery_skip_invalid` — Skipped invalid hostname
- `discovery_skip_duplicate` — Skipped duplicate
- `discovery_targets` — Total unique targets

---

### Step 5: Parallel Task Spawning

**File:** `bin/10_queen_rbee/src/discovery.rs`

```rust
// Discover all hives in parallel
let mut tasks = vec![];
for target in unique_targets {
    let queen_url = queen_url.to_string();
    
    tasks.push(tokio::spawn(async move {
        discover_single_hive(&target, &queen_url).await
    }));
}

// Wait for all discoveries
let mut success_count = 0;
let mut failure_count = 0;

for task in tasks {
    match task.await {
        Ok(Ok(_)) => success_count += 1,
        Ok(Err(_)) => failure_count += 1,
        Err(_) => failure_count += 1,
    }
}

n!("discovery_complete", "✅ Discovery complete: {} successful, {} failed", success_count, failure_count);

Ok(())
```

**Location:** Lines 82-106  
**Narration:** `discovery_complete` — Final summary

---

### Step 6: Single Hive Discovery

**File:** `bin/10_queen_rbee/src/discovery.rs`

```rust
/// Discover a single hive
///
/// Sends GET /capabilities?queen_url=X to hive
async fn discover_single_hive(target: &SshTarget, queen_url: &str) -> Result<()> {
    // Construct URL with encoded queen_url
    let url = format!(
        "http://{}:7835/capabilities?queen_url={}",
        target.hostname,
        urlencoding::encode(queen_url)
    );
    
    n!("discovery_hive", "🔍 Discovering hive: {} ({})", target.host, target.hostname);
    
    // Send GET request with timeout
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    
    // Check response status
    if response.status().is_success() {
        n!("discovery_success", "✅ Discovered hive: {}", target.host);
        
        // TODO: Store capabilities in HiveRegistry
        // let capabilities: CapabilitiesResponse = response.json().await?;
        // hive_registry.register_hive(target.host.clone(), capabilities);
    } else {
        n!("discovery_failed", "❌ Failed to discover hive {}: {}", target.host, response.status());
        anyhow::bail!("Failed to discover hive {}: {}", target.host, response.status());
    }
    
    Ok(())
}
```

**Location:** Lines 109-144  
**Function:** `discover_single_hive()`  
**Narration:**
- `discovery_hive` — Discovering hive
- `discovery_success` — Success
- `discovery_failed` — Failure

**HTTP Request:**
- **Method:** GET
- **URL:** `http://{hostname}:7835/capabilities?queen_url={encoded}`
- **Timeout:** 10 seconds
- **Example:** `http://192.168.1.100:7835/capabilities?queen_url=http%3A%2F%2Flocalhost%3A7833`

---

### Step 7: Hive Health Check Endpoint

**File:** `bin/20_rbee_hive/src/main.rs`

**Route Registration:**
```rust
let app = Router::new()
    // ... other routes ...
    .route("/v1/capabilities", get(get_capabilities))
    .with_state(hive_state.clone());
```

**Location:** Line 203

**Handler Function:**
```rust
/// GET /v1/capabilities - Return hive capabilities
///
/// Query parameters:
/// - queen_url: Optional URL of queen for bidirectional discovery
async fn get_capabilities(
    Query(params): Query<CapabilitiesQuery>,
    State(state): State<Arc<HiveState>>,
) -> Json<CapabilitiesResponse> {
    n!("caps_request", "📡 Received capabilities request from queen");
    
    // Handle queen_url parameter
    if let Some(queen_url) = params.queen_url {
        n!("caps_queen_url", "🔗 Queen URL received: {}", queen_url);
        
        // Validate and store URL
        match state.set_queen_url(queen_url.clone()).await {
            Ok(_) => {
                state.start_heartbeat_task(Some(queen_url)).await;
            }
            Err(e) => {
                n!("caps_invalid_url", "❌ Invalid queen_url rejected: {}", e);
            }
        }
    }
    
    // Detect GPUs
    n!("caps_gpu_check", "🔍 Detecting GPUs via nvidia-smi...");
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    
    if gpu_info.count > 0 {
        n!("caps_gpu_found", "✅ Found {} GPU(s)", gpu_info.count);
    } else {
        n!("caps_gpu_none", "ℹ️  No GPUs detected, using CPU only");
    }
    
    // Build GPU devices
    let mut devices: Vec<HiveDevice> = gpu_info
        .devices
        .iter()
        .map(|gpu| HiveDevice {
            id: format!("GPU-{}", gpu.index),
            name: gpu.name.clone(),
            device_type: "gpu".to_string(),
            vram_gb: Some(gpu.vram_total_gb() as u32),
            compute_capability: Some(format!(
                "{}.{}",
                gpu.compute_capability.0, gpu.compute_capability.1
            )),
        })
        .collect();
    
    // Detect CPU/RAM
    let cpu_cores = rbee_hive_device_detection::get_cpu_cores();
    let system_ram_gb = rbee_hive_device_detection::get_system_ram_gb();
    
    n!("caps_cpu_add", "🖥️  Adding CPU-0: {} cores, {} GB RAM", cpu_cores, system_ram_gb);
    
    // Add CPU device
    devices.push(HiveDevice {
        id: "CPU-0".to_string(),
        name: format!("CPU ({} cores)", cpu_cores),
        device_type: "cpu".to_string(),
        vram_gb: Some(system_ram_gb),
        compute_capability: None,
    });
    
    n!("caps_response", "📤 Sending capabilities response ({} device(s))", devices.len());
    
    Json(CapabilitiesResponse { devices })
}
```

**Location:** Lines 389-466  
**Function:** `get_capabilities()`  
**Narration:**
- `caps_request` — Request received
- `caps_queen_url` — Queen URL received
- `caps_invalid_url` — Invalid URL rejected
- `caps_gpu_check` — Detecting GPUs
- `caps_gpu_found` — GPUs found
- `caps_gpu_none` — No GPUs
- `caps_cpu_add` — Adding CPU
- `caps_response` — Sending response

---

### Step 8: Response Models

**Query Parameter:**

**File:** `bin/20_rbee_hive/src/main.rs`

```rust
#[derive(Debug, Deserialize)]
struct CapabilitiesQuery {
    queen_url: Option<String>,
}
```

**Response Type:**

**File:** `bin/97_contracts/hive-contract/src/lib.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitiesResponse {
    pub devices: Vec<HiveDevice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveDevice {
    pub id: String,                      // "GPU-0", "CPU-0"
    pub name: String,                    // "NVIDIA GeForce RTX 3090"
    pub device_type: String,             // "gpu", "cpu"
    pub vram_gb: Option<u32>,            // 24
    pub compute_capability: Option<String>, // "8.6"
}
```

**Example Response:**
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
      "id": "GPU-1",
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

## Complete Narration Event List

### Queen Events

| Event | Message | Location |
|-------|---------|----------|
| `discovery_error` | "❌ Hive discovery failed: {error}" | main.rs:89 |
| `discovery_invalid_url` | "❌ Cannot start discovery: invalid queen_url" | discovery.rs:35 |
| `discovery_start` | "🔍 Starting hive discovery (waiting 5s for services to stabilize)" | discovery.rs:44 |
| `discovery_no_config` | "⚠️  No SSH config found: {error}. Only localhost will be discovered." | discovery.rs:54 |
| `discovery_skip_invalid` | "⚠️  Skipping target '{host}': empty hostname" | discovery.rs:66 |
| `discovery_skip_duplicate` | "⚠️  Skipping duplicate target: {host} ({hostname})" | discovery.rs:72 |
| `discovery_targets` | "📋 Found {count} unique SSH targets to discover" | discovery.rs:80 |
| `discovery_hive` | "🔍 Discovering hive: {host} ({hostname})" | discovery.rs:123 |
| `discovery_success` | "✅ Discovered hive: {host}" | discovery.rs:133 |
| `discovery_failed` | "❌ Failed to discover hive {host}: {status}" | discovery.rs:139 |
| `discovery_complete` | "✅ Discovery complete: {success} successful, {failed} failed" | discovery.rs:104 |

### Hive Events

| Event | Message | Location |
|-------|---------|----------|
| `caps_request` | "📡 Received capabilities request from queen" | main.rs:395 |
| `caps_queen_url` | "🔗 Queen URL received: {url}" | main.rs:400 |
| `caps_invalid_url` | "❌ Invalid queen_url rejected: {error}" | main.rs:409 |
| `caps_gpu_check` | "🔍 Detecting GPUs via nvidia-smi..." | main.rs:416 |
| `caps_gpu_found` | "✅ Found {count} GPU(s)" | main.rs:424 |
| `caps_gpu_none` | "ℹ️  No GPUs detected, using CPU only" | main.rs:426 |
| `caps_cpu_add` | "🖥️  Adding CPU-0: {cores} cores, {ram} GB RAM" | main.rs:450 |
| `caps_response` | "📤 Sending capabilities response ({count} device(s))" | main.rs:463 |

---

## Key Files Summary

| File | Purpose | Key Functions |
|------|---------|---------------|
| `bin/10_queen_rbee/src/main.rs` | Queen entry | `main()` |
| `bin/10_queen_rbee/src/discovery.rs` | Discovery logic | `discover_hives_on_startup()`, `discover_single_hive()` |
| `bin/99_shared_crates/ssh-config-parser/src/lib.rs` | SSH parsing | `parse_ssh_config()`, `get_default_ssh_config_path()` |
| `bin/20_rbee_hive/src/main.rs` | Hive capabilities | `get_capabilities()` |
| `bin/25_rbee_hive_crates/device-detection/src/detection.rs` | Device detection | `detect_gpus()`, `get_cpu_cores()`, `get_system_ram_gb()` |
| `bin/97_contracts/hive-contract/src/lib.rs` | Response types | `CapabilitiesResponse`, `HiveDevice` |

---

## Configuration

**SSH Config Location:**
- Default: `~/.ssh/config`
- Function: `get_default_ssh_config_path()`

**Timeouts:**
- Service stabilization: 5 seconds
- HTTP request: 10 seconds per hive

**Ports:**
- Queen: 7833 (CLI configurable)
- Hive: 7835 (hardcoded)

---

## Error Handling

### Non-Fatal Errors (Continue Discovery)

- SSH config not found → Use empty list
- Invalid hostname → Skip target
- Duplicate hostname → Skip target
- HTTP timeout → Count as failure
- HTTP error status → Count as failure
- Network unreachable → Count as failure

### Fatal Errors (Stop Discovery)

- Empty queen_url → Bail immediately
- Invalid queen_url format → Bail immediately

---

## Performance Characteristics

**For N hives:**
- **Sequential:** N × 10s = 10N seconds
- **Parallel:** ~10s (all at once)
- **Speedup:** Nx

**Example (10 hives):**
- Sequential: 100 seconds
- Parallel: ~10 seconds
- **Speedup:** 10x

---

## Success/Failure Tracking

**Success Criteria:**
- HTTP 200 OK response
- Valid JSON response (optional validation)

**Failure Criteria:**
- HTTP timeout (10s)
- HTTP error status (4xx, 5xx)
- Network error
- Invalid response format

**Summary Format:**
```
✅ Discovery complete: 8 successful, 2 failed
```

---

## Example SSH Config

```
# GPU Hives
Host hive-gpu-1
    HostName 192.168.1.100
    User rbee
    Port 22

Host hive-gpu-2
    HostName 192.168.1.101
    User rbee
    Port 22

# CPU Hives
Host hive-cpu-1
    HostName 192.168.1.200
    User rbee
    Port 22

# Localhost (for testing)
Host localhost
    HostName 127.0.0.1
    User rbee
    Port 22
```

**Parsed Result:**
```rust
vec![
    SshTarget { host: "hive-gpu-1", hostname: "192.168.1.100", user: Some("rbee"), port: Some(22) },
    SshTarget { host: "hive-gpu-2", hostname: "192.168.1.101", user: Some("rbee"), port: Some(22) },
    SshTarget { host: "hive-cpu-1", hostname: "192.168.1.200", user: Some("rbee"), port: Some(22) },
    SshTarget { host: "localhost", hostname: "127.0.0.1", user: Some("rbee"), port: Some(22) },
]
```

---

**Status:** ✅ COMPLETE  
**Maintainer:** TEAM-385+  
**Last Updated:** November 2, 2025
