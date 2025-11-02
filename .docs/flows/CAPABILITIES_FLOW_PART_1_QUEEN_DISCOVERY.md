# Capabilities Flow Part 1: Queen Discovery Initiation

**Flow:** Queen Startup → SSH Config → Parallel Discovery Probes  
**Date:** November 2, 2025  
**Status:** ✅ COMPLETE

---

## Overview

This document traces the flow from when `queen-rbee` starts up to when it sends parallel capabilities discovery requests to all configured hives.

**Discovery Pattern:** Pull-based (Queen initiates)

**Ports:**
- Queen: 7833
- Hive: 7835

---

## Step 1: Queen Startup

### File: `bin/10_queen_rbee/src/main.rs`

**Main Function:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // ... initialization ...
    
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
**Function:** `main()`  
**Purpose:** Spawn discovery task in background

**Key Details:**
- Discovery runs in background (doesn't block server startup)
- Queen URL constructed from CLI port argument
- Default: `http://localhost:7833`

---

## Step 2: Discovery Initialization

### File: `bin/10_queen_rbee/src/discovery.rs`

**Discovery Function:**
```rust
/// Discover all hives on Queen startup
///
/// TEAM-365: Pull-based discovery (Scenario 1 from HEARTBEAT_ARCHITECTURE.md)
///
/// # Flow
/// 1. Wait 5 seconds for services to stabilize
/// 2. Read SSH config
/// 3. Send parallel GET /capabilities?queen_url=X to all hives
/// 4. Store capabilities in HiveRegistry (TODO: when implemented)
pub async fn discover_hives_on_startup(queen_url: &str) -> Result<()> {
    // Step 2a: Validate queen_url
    if queen_url.is_empty() {
        n!("discovery_invalid_url", "❌ Cannot start discovery: empty queen_url");
        anyhow::bail!("Cannot start discovery with empty queen_url");
    }
    
    if let Err(e) = url::Url::parse(queen_url) {
        n!("discovery_invalid_url", "❌ Cannot start discovery: invalid queen_url '{}': {}", queen_url, e);
        anyhow::bail!("Invalid queen_url '{}': {}", queen_url, e);
    }
    
    // Step 2b: Emit start event
    n!("discovery_start", "🔍 Starting hive discovery (waiting 5s for services to stabilize)");
    
    // Step 2c: Wait for services to stabilize
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // ... continued ...
}
```

**Location:** Lines 32-47  
**Function:** `discover_hives_on_startup()`  
**Purpose:** Initialize discovery process

**Narration Events:**
- `discovery_invalid_url` — Invalid queen URL
- `discovery_start` — Discovery starting

**Why 5-Second Wait:**
- Allows hives to finish startup
- Prevents connection failures
- Gives time for network stabilization

---

## Step 3: SSH Config Parsing

### File: `bin/10_queen_rbee/src/discovery.rs`

**SSH Config Reading:**
```rust
// Step 3a: Get default SSH config path
let ssh_config_path = get_default_ssh_config_path();

// Step 3b: Parse SSH config
let targets = match parse_ssh_config(&ssh_config_path) {
    Ok(targets) => targets,
    Err(e) => {
        n!("discovery_no_config", "⚠️  No SSH config found: {}. Only localhost will be discovered.", e);
        vec![]
    }
};
```

**Location:** Lines 49-57  
**Purpose:** Read hive targets from SSH config

**Narration Events:**
- `discovery_no_config` — No SSH config found (not an error)

**SSH Config Format:**
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

**Parsed Result:**
```rust
vec![
    SshTarget {
        host: "hive-gpu-1",
        hostname: "192.168.1.100",
        user: Some("rbee"),
        port: Some(22),
    },
    SshTarget {
        host: "hive-gpu-2",
        hostname: "192.168.1.101",
        user: Some("rbee"),
        port: Some(22),
    },
]
```

---

## Step 4: Target Deduplication

### File: `bin/10_queen_rbee/src/discovery.rs`

**Deduplication Logic:**
```rust
// Step 4a: Create set for tracking seen hostnames
let mut seen = HashSet::new();

// Step 4b: Filter targets
let unique_targets: Vec<_> = targets
    .into_iter()
    .filter(|t| {
        // EDGE CASE #8: Skip invalid hostnames
        if t.hostname.is_empty() {
            n!("discovery_skip_invalid", "⚠️  Skipping target '{}': empty hostname", t.host);
            return false;
        }
        
        // EDGE CASE #7: Skip duplicates
        if !seen.insert(t.hostname.clone()) {
            n!("discovery_skip_duplicate", "⚠️  Skipping duplicate target: {} ({})", t.host, t.hostname);
            return false;
        }
        
        true
    })
    .collect();

// Step 4c: Emit target count
n!("discovery_targets", "📋 Found {} unique SSH targets to discover", unique_targets.len());
```

**Location:** Lines 59-80  
**Purpose:** Deduplicate targets by hostname

**Narration Events:**
- `discovery_skip_invalid` — Skipped invalid hostname
- `discovery_skip_duplicate` — Skipped duplicate hostname
- `discovery_targets` — Total unique targets found

**Why Deduplicate:**
- Multiple SSH aliases may point to same host
- Prevents duplicate discovery requests
- Reduces network overhead

---

## Step 5: Parallel Discovery

### File: `bin/10_queen_rbee/src/discovery.rs`

**Parallel Task Spawning:**
```rust
// Step 5a: Create task list
let mut tasks = vec![];

// Step 5b: Spawn task for each target
for target in unique_targets {
    let queen_url = queen_url.to_string();
    
    tasks.push(tokio::spawn(async move {
        discover_single_hive(&target, &queen_url).await
    }));
}

// Step 5c: Wait for all tasks
let mut success_count = 0;
let mut failure_count = 0;

for task in tasks {
    match task.await {
        Ok(Ok(_)) => success_count += 1,
        Ok(Err(_)) => failure_count += 1,
        Err(_) => failure_count += 1,
    }
}

// Step 5d: Emit completion event
n!("discovery_complete", "✅ Discovery complete: {} successful, {} failed", success_count, failure_count);
```

**Location:** Lines 82-106  
**Purpose:** Discover all hives in parallel

**Narration Events:**
- `discovery_complete` — Discovery finished with counts

**Why Parallel:**
- ✅ Faster discovery (all hives at once)
- ✅ Non-blocking (independent failures)
- ✅ Timeout isolation (one slow hive doesn't block others)

---

## Step 6: Single Hive Discovery

### File: `bin/10_queen_rbee/src/discovery.rs`

**Discovery Request:**
```rust
/// Discover a single hive
///
/// TEAM-365: Send GET /capabilities?queen_url=X to hive
async fn discover_single_hive(target: &SshTarget, queen_url: &str) -> Result<()> {
    // Step 6a: Construct URL
    let url = format!(
        "http://{}:7835/capabilities?queen_url={}",
        target.hostname,
        urlencoding::encode(queen_url)
    );
    
    // Step 6b: Emit discovery event
    n!("discovery_hive", "🔍 Discovering hive: {} ({})", target.host, target.hostname);
    
    // Step 6c: Send GET request
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    
    // Step 6d: Check response
    if response.status().is_success() {
        n!("discovery_success", "✅ Discovered hive: {}", target.host);
        
        // TODO: TEAM-365: Store capabilities in HiveRegistry
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
**Purpose:** Send capabilities request to one hive

**Narration Events:**
- `discovery_hive` — Discovering hive
- `discovery_success` — Hive discovered successfully
- `discovery_failed` — Hive discovery failed

**Request Details:**
- **Method:** GET
- **URL:** `http://{hostname}:7835/capabilities?queen_url={encoded_url}`
- **Timeout:** 10 seconds
- **Example:** `http://192.168.1.100:7835/capabilities?queen_url=http%3A%2F%2Flocalhost%3A7833`

---

## Step 7: URL Encoding

### Query Parameter Encoding

**Queen URL Encoding:**
```rust
urlencoding::encode(queen_url)
```

**Example:**
```
Input:  "http://localhost:7833"
Output: "http%3A%2F%2Flocalhost%3A7833"
```

**Why Encode:**
- Queen URL contains special characters (`:`, `/`)
- Must be URL-safe for query parameter
- Prevents parsing errors

---

## Data Flow Summary

```
Queen Startup
    ↓
main() [main.rs:86]
    ↓ spawn discovery task
    ↓
discover_hives_on_startup() [discovery.rs:32]
    ↓ validate queen_url
    ↓ n!("discovery_start")
    ↓ sleep 5 seconds
    ↓
get_default_ssh_config_path() [ssh-config-parser]
    ↓ ~/.ssh/config
    ↓
parse_ssh_config() [ssh-config-parser]
    ↓ parse SSH config file
    ↓ extract targets
    ↓
Deduplicate targets [discovery.rs:59]
    ↓ n!("discovery_skip_invalid")
    ↓ n!("discovery_skip_duplicate")
    ↓ n!("discovery_targets")
    ↓
Spawn parallel tasks [discovery.rs:82]
    ↓ tokio::spawn for each target
    ↓
discover_single_hive() [discovery.rs:116]
    ↓ construct URL
    ↓ n!("discovery_hive")
    ↓ GET http://{hostname}:7835/capabilities?queen_url={encoded}
    ↓ timeout 10s
    ↓
Hive Responds (Part 2)
    ↓ (continued in next part)
```

---

## Narration Events (Part 1)

| Event | Action | Message | Location |
|-------|--------|---------|----------|
| `discovery_invalid_url` | Validation | "❌ Cannot start discovery: invalid queen_url" | discovery.rs:35 |
| `discovery_start` | Start | "🔍 Starting hive discovery (waiting 5s for services to stabilize)" | discovery.rs:44 |
| `discovery_no_config` | No config | "⚠️  No SSH config found: {error}. Only localhost will be discovered." | discovery.rs:54 |
| `discovery_skip_invalid` | Skip | "⚠️  Skipping target '{host}': empty hostname" | discovery.rs:66 |
| `discovery_skip_duplicate` | Skip | "⚠️  Skipping duplicate target: {host} ({hostname})" | discovery.rs:72 |
| `discovery_targets` | Count | "📋 Found {count} unique SSH targets to discover" | discovery.rs:80 |
| `discovery_hive` | Probe | "🔍 Discovering hive: {host} ({hostname})" | discovery.rs:123 |
| `discovery_success` | Success | "✅ Discovered hive: {host}" | discovery.rs:133 |
| `discovery_failed` | Failure | "❌ Failed to discover hive {host}: {status}" | discovery.rs:139 |
| `discovery_complete` | Complete | "✅ Discovery complete: {success} successful, {failed} failed" | discovery.rs:104 |
| `discovery_error` | Error | "❌ Hive discovery failed: {error}" | main.rs:89 |

---

## Key Files Referenced

| File | Purpose | Key Functions |
|------|---------|---------------|
| `bin/10_queen_rbee/src/main.rs` | Queen entry point | `main()` |
| `bin/10_queen_rbee/src/discovery.rs` | Discovery logic | `discover_hives_on_startup()`, `discover_single_hive()` |
| `bin/99_shared_crates/ssh-config-parser/src/lib.rs` | SSH config parsing | `parse_ssh_config()`, `get_default_ssh_config_path()` |

---

## Configuration

**SSH Config Location:**
- Default: `~/.ssh/config`
- Can be overridden via environment variable

**Timeouts:**
- Service stabilization: 5 seconds
- HTTP request: 10 seconds per hive

**Ports:**
- Queen: 7833 (configurable via CLI)
- Hive: 7835 (hardcoded in discovery)

---

## Error Handling

### Possible Errors

**Validation Errors:**
- Empty queen_url
- Invalid queen_url format

**Discovery Errors:**
- SSH config not found (not fatal)
- Invalid hostname (skipped)
- Duplicate hostname (skipped)
- HTTP timeout (10s)
- HTTP error status
- Network unreachable

**Error Recovery:**
- Individual hive failures don't stop discovery
- Parallel tasks are isolated
- Counts reported at end

---

## Edge Cases

### EDGE CASE #6: Empty queen_url
**Problem:** Empty string passed as queen_url  
**Solution:** Validate before starting discovery  
**Result:** Bail early with error

### EDGE CASE #7: Duplicate Hostnames
**Problem:** Multiple SSH aliases point to same host  
**Solution:** Deduplicate by hostname  
**Result:** Only one request per physical hive

### EDGE CASE #8: Invalid Hostnames
**Problem:** SSH config has empty hostname  
**Solution:** Skip during filtering  
**Result:** No request sent to invalid target

---

## Performance Characteristics

### Discovery Speed

**Sequential (hypothetical):**
- 10 hives × 10s timeout = 100s total

**Parallel (actual):**
- 10 hives × 10s timeout = ~10s total (all at once)

**Speedup:** 10x for 10 hives

### Memory Usage

**Per hive:**
- HTTP client: ~1KB
- Task overhead: ~8KB
- Total: ~9KB per hive

**For 10 hives:** ~90KB total

---

**Next:** [CAPABILITIES_FLOW_PART_2_HIVE_DETECTION.md](./CAPABILITIES_FLOW_PART_2_HIVE_DETECTION.md) — Hive receives request, detects devices
