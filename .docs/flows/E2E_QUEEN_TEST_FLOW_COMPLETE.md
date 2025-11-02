# E2E Queen Lifecycle Test Flow: Complete Roundtrip

**Flow:** cargo xtask → Test Harness → Start → Health Poll → Stop → Report  
**Date:** November 2, 2025  
**Status:** ✅ COMPLETE

---

## Overview

This document traces the complete E2E test flow from when a developer runs `cargo xtask e2e:queen` to when the test result is printed to the console.

**Command:**
```bash
cargo xtask e2e:queen
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ DEVELOPER TERMINAL                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Run: cargo xtask e2e:queen                              │
│    ↓                                                        │
│ 2. Cargo builds xtask binary                               │
│    ↓                                                        │
│ 3. xtask CLI parses command                                │
│    └─→ Match Cmd::E2eQueen                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ XTASK TEST HARNESS                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. test_queen_lifecycle() starts                           │
│    ↓                                                        │
│ 2. Print: "🚀 E2E Test: Queen Lifecycle"                   │
│    ↓                                                        │
│ 3. Spawn: rbee-keeper queen start                          │
│    ├─→ Command::new("target/debug/rbee-keeper")           │
│    ├─→ .args(["queen", "start"])                           │
│    ├─→ .spawn() (shows live output)                        │
│    └─→ .wait() (blocks until complete)                     │
│    ↓                                                        │
│ 4. Check exit code                                          │
│    └─→ if !status.success() { bail!() }                    │
│    ↓                                                        │
│ 5. Spawn: rbee-keeper queen stop                           │
│    ├─→ Command::new("target/debug/rbee-keeper")           │
│    ├─→ .args(["queen", "stop"])                            │
│    ├─→ .spawn() (shows live output)                        │
│    └─→ .wait() (blocks until complete)                     │
│    ↓                                                        │
│ 6. Check exit code                                          │
│    └─→ if !status.success() { bail!() }                    │
│    ↓                                                        │
│ 7. Print: "✅ E2E Test PASSED: Queen Lifecycle"            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ RBEE-KEEPER (Queen Start)                                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Parse CLI: queen start                                  │
│    ↓                                                        │
│ 2. handle_queen_lifecycle(QueenAction::Start)              │
│    ↓                                                        │
│ 3. Build HttpDaemonConfig                                   │
│    ├─→ daemon_name: "queen-rbee"                           │
│    ├─→ health_url: "http://localhost:7833/health"         │
│    └─→ args: ["--port", "7833"]                            │
│    ↓                                                        │
│ 4. Call lifecycle_local::start_daemon()                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LIFECYCLE-LOCAL (Start Daemon)                             │
├─────────────────────────────────────────────────────────────┤
│ 1. Find binary                                              │
│    ├─→ Try: target/debug/queen-rbee                        │
│    ├─→ Try: target/release/queen-rbee                      │
│    └─→ Found: target/debug/queen-rbee                      │
│    ↓                                                        │
│ 2. Start daemon process                                     │
│    ├─→ Command: target/debug/queen-rbee --port 7833       │
│    ├─→ Stdio: Stdio::null() (no output capture)           │
│    ├─→ Spawn process                                       │
│    └─→ Capture PID: 12345                                  │
│    ↓                                                        │
│ 3. Emit narration: "✅ Daemon started with PID: 12345"     │
│    ↓                                                        │
│ 4. Poll health endpoint                                     │
│    └─→ health_poll::poll_health()                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HEALTH-POLL (Exponential Backoff)                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Create HTTP client (5s timeout)                         │
│    ↓                                                        │
│ 2. Attempt 1: GET /health                                  │
│    ├─→ Emit: "⏳ Health check attempt 1/30"                │
│    ├─→ Result: Connection refused                          │
│    └─→ Emit: "⏳ Connection failed - retrying..."          │
│    ↓                                                        │
│ 3. Wait 200ms                                               │
│    ↓                                                        │
│ 4. Attempt 2: GET /health                                  │
│    ├─→ Emit: "⏳ Health check attempt 2/30"                │
│    ├─→ Result: Connection refused                          │
│    └─→ Emit: "⏳ Connection failed - retrying..."          │
│    ↓                                                        │
│ 5. Wait 300ms (200 × 1.5)                                  │
│    ↓                                                        │
│ 6. Attempt 3: GET /health                                  │
│    ├─→ Emit: "⏳ Health check attempt 3/30"                │
│    ├─→ Result: 200 OK ✅                                   │
│    └─→ Emit: "✅ Health check passed"                      │
│    ↓                                                        │
│ 7. Return Ok(())                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LIFECYCLE-LOCAL (Complete Start)                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Emit: "✅ Daemon is healthy and responding"             │
│    ↓                                                        │
│ 2. Emit: "🎉 queen-rbee started successfully (PID: 12345)" │
│    ↓                                                        │
│ 3. Return Ok(PID)                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ RBEE-KEEPER (Queen Stop)                                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Parse CLI: queen stop                                   │
│    ↓                                                        │
│ 2. handle_queen_lifecycle(QueenAction::Stop)               │
│    ↓                                                        │
│ 3. Build StopConfig                                         │
│    ├─→ daemon_name: "queen-rbee"                           │
│    ├─→ shutdown_url: "http://localhost:7833/v1/shutdown"  │
│    └─→ health_url: "http://localhost:7833/health"         │
│    ↓                                                        │
│ 4. Call lifecycle_local::stop_daemon()                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ LIFECYCLE-LOCAL (Stop Daemon)                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Try graceful shutdown                                    │
│    ├─→ POST /v1/shutdown                                   │
│    ├─→ Result: 200 OK                                      │
│    └─→ Emit: "✅ Graceful shutdown successful"             │
│    ↓                                                        │
│ 2. Poll health until down                                   │
│    ├─→ GET /health                                         │
│    ├─→ Result: Connection refused ✅                       │
│    └─→ Emit: "✅ Daemon stopped"                           │
│    ↓                                                        │
│ 3. Return Ok(())                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ DEVELOPER TERMINAL (Final Output)                          │
├─────────────────────────────────────────────────────────────┤
│ ✅ E2E Test PASSED: Queen Lifecycle                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Developer Runs Test

**Command:**
```bash
cargo xtask e2e:queen
```

**What Happens:**
1. Cargo builds `xtask` binary
2. Runs `xtask` with `e2e:queen` argument
3. xtask CLI parses command

---

### Step 2: xtask CLI Dispatch

**File:** `xtask/src/main.rs`

```rust
#[derive(Parser)]
enum Cmd {
    // ... other commands
    
    /// E2E test: Queen lifecycle (start + stop)
    #[command(name = "e2e:queen")]
    E2eQueen,
    
    // ... other commands
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.cmd {
        Cmd::E2eQueen => {
            tokio::runtime::Runtime::new()?
                .block_on(e2e::test_queen_lifecycle())?
        }
        // ... other commands
    }
    
    Ok(())
}
```

**Location:** Line 173  
**Purpose:** Route command to test function

---

### Step 3: Test Harness Entry

**File:** `xtask/src/e2e/queen_lifecycle.rs`

```rust
/// Queen lifecycle E2E test
///
/// Tests:
/// - rbee queen start
/// - rbee queen stop
///
/// TEAM-162: Tests rely ONLY on CLI stdout/stderr.
/// No internal product functions. Pure black-box testing.
///
/// TEAM-164: E2E tests MUST show live narration output.
/// Using .output() hides all narration until command completes.
/// Using .spawn() + .wait() shows narration in real-time.
pub async fn test_queen_lifecycle() -> Result<()> {
    println!("🚀 E2E Test: Queen Lifecycle\n");
    
    // Step 1: rbee queen start
    println!("📝 Running: rbee queen start\n");
    
    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper")
        .args(["queen", "start"])
        .spawn()?;
    
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee queen start failed with exit code: {:?}", status.code());
    }
    
    println!();
    
    // Step 2: rbee queen stop
    println!("📝 Running: rbee queen stop\n");
    
    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper")
        .args(["queen", "stop"])
        .spawn()?;
    
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("rbee queen stop failed with exit code: {:?}", status.code());
    }
    
    println!();
    
    println!("✅ E2E Test PASSED: Queen Lifecycle");
    Ok(())
}
```

**Location:** Lines 20-51  
**Purpose:** Execute test sequence

**Key Design Decisions:**
- **Black-box testing** — No internal product functions
- **Live output** — `.spawn()` + `.wait()` shows narration in real-time
- **Exit code checking** — Fail fast on non-zero exit

---

### Step 4: rbee-keeper Queen Start

**File:** `bin/00_rbee_keeper/src/handlers/queen.rs`

```rust
pub async fn handle_queen_lifecycle(action: QueenAction) -> Result<()> {
    match action {
        QueenAction::Start => {
            let base_url = "http://localhost:7833";
            let health_url = format!("{}/health", base_url);
            let args = vec!["--port".to_string(), "7833".to_string()];
            
            let daemon_config = lifecycle_local::HttpDaemonConfig::new("queen-rbee", &health_url)
                .with_args(args);
            
            let config = lifecycle_local::StartConfig {
                daemon_config,
                job_id: None,
            };
            
            let _pid = lifecycle_local::start_daemon(config).await?;
            Ok(())
        }
        // ... other actions
    }
}
```

**Location:** Lines 53-61  
**Purpose:** Delegate to lifecycle-local

---

### Step 5: lifecycle-local Start Daemon

**File:** `bin/96_lifecycle/lifecycle-local/src/start.rs`

```rust
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    let daemon_config = start_config.daemon_config;
    let daemon_name = &daemon_config.daemon_name;
    
    // Step 1: Find binary
    n!("finding_binary", "🔍 Finding binary: {}", daemon_name);
    
    let binary_path = find_binary(daemon_name)?;
    
    n!("found_binary", "✅ Found binary: {}", binary_path);
    
    // Step 2: Start daemon process
    n!("starting", "🚀 Starting daemon: {}", daemon_name);
    
    let mut cmd = Command::new(&binary_path);
    cmd.args(&daemon_config.args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    
    let child = cmd.spawn()
        .context(format!("Failed to spawn daemon: {}", daemon_name))?;
    
    let pid = child.id();
    
    n!("started", "✅ Daemon started with PID: {}", pid);
    
    // Step 3: Poll health endpoint
    n!("health_check", "🏥 Polling health endpoint: {}", daemon_config.health_url);
    
    health_poll::poll_health(
        &daemon_config.health_url,
        30,  // max_attempts
        200, // initial_delay_ms
        1.5, // backoff_multiplier
    )
    .await
    .context("Daemon started but failed health check")?;
    
    n!("healthy", "✅ Daemon is healthy and responding");
    n!("start_complete", "🎉 {} started successfully (PID: {})", daemon_name, pid);
    
    Ok(pid)
}
```

**Location:** Lines 100-209  
**Purpose:** Start daemon and verify health

**Narration Events:**
- `finding_binary` — Finding binary
- `found_binary` — Binary found
- `starting` — Starting daemon
- `started` — Daemon started with PID
- `health_check` — Polling health endpoint
- `healthy` — Daemon is healthy
- `start_complete` — Start complete

---

### Step 6: health-poll Exponential Backoff

**File:** `bin/96_lifecycle/health-poll/src/lib.rs`

```rust
/// Poll a health endpoint until it responds successfully
///
/// Uses exponential backoff for retries
pub async fn poll_health(
    url: &str,
    max_attempts: usize,
    initial_delay_ms: u64,
    backoff_multiplier: f64,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("Failed to create HTTP client")?;
    
    let mut delay_ms = initial_delay_ms;
    
    for attempt in 1..=max_attempts {
        // Wait before attempt (except first)
        if attempt > 1 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            delay_ms = (delay_ms as f64 * backoff_multiplier) as u64;
        }
        
        // Emit narration
        n!("health_attempt", "⏳ Health check attempt {}/{}", attempt, max_attempts);
        
        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => {
                n!("health_success", "✅ Health check passed");
                return Ok(());
            }
            Ok(response) => {
                n!("health_retry", "⏳ HTTP {} - retrying...", response.status());
            }
            Err(_) => {
                n!("health_retry", "⏳ Connection failed - retrying...");
            }
        }
    }
    
    anyhow::bail!("Health check failed after {} attempts: {}", max_attempts, url)
}
```

**Location:** Lines 40-96  
**Purpose:** Poll health with exponential backoff

**Backoff Schedule:**
```
Attempt 1: Wait 0ms
Attempt 2: Wait 200ms
Attempt 3: Wait 300ms (200 × 1.5)
Attempt 4: Wait 450ms (300 × 1.5)
Attempt 5: Wait 675ms (450 × 1.5)
...
Max: 30 attempts
```

**Narration Events:**
- `health_attempt` — Health check attempt N/M
- `health_success` — Health check passed
- `health_retry` — Retrying after failure

---

### Step 7: rbee-keeper Queen Stop

**File:** `bin/00_rbee_keeper/src/handlers/queen.rs`

```rust
QueenAction::Stop => {
    let shutdown_url = "http://localhost:7833/v1/shutdown";
    let health_url = "http://localhost:7833/health";
    
    let config = lifecycle_local::StopConfig {
        daemon_name: "queen-rbee".to_string(),
        shutdown_url: shutdown_url.to_string(),
        health_url: health_url.to_string(),
        job_id: None,
    };
    
    lifecycle_local::stop_daemon(config).await
}
```

**Location:** Lines 64-74  
**Purpose:** Delegate to lifecycle-local

---

### Step 8: lifecycle-local Stop Daemon

**File:** `bin/96_lifecycle/lifecycle-local/src/stop.rs`

```rust
pub async fn stop_daemon(stop_config: StopConfig) -> Result<()> {
    let daemon_name = &stop_config.daemon_name;
    let shutdown_url = &stop_config.shutdown_url;
    let health_url = &stop_config.health_url;
    
    // Step 1: Try graceful shutdown
    n!("stopping", "🛑 Stopping daemon: {}", daemon_name);
    
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    
    match client.post(shutdown_url).send().await {
        Ok(response) if response.status().is_success() => {
            n!("shutdown_sent", "✅ Graceful shutdown successful");
        }
        Ok(response) => {
            n!("shutdown_failed", "⚠️  Shutdown endpoint returned: {}", response.status());
        }
        Err(e) => {
            n!("shutdown_error", "⚠️  Shutdown request failed: {}", e);
        }
    }
    
    // Step 2: Poll health until down
    n!("waiting_stop", "⏳ Waiting for daemon to stop...");
    
    for attempt in 1..=30 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        match client.get(health_url).send().await {
            Err(_) => {
                // Connection refused = daemon stopped
                n!("stopped", "✅ Daemon stopped");
                return Ok(());
            }
            Ok(_) => {
                // Still running
                if attempt % 5 == 0 {
                    n!("still_running", "⏳ Daemon still running (attempt {}/30)", attempt);
                }
            }
        }
    }
    
    anyhow::bail!("Daemon did not stop after 30 attempts")
}
```

**Location:** Lines 50-150  
**Purpose:** Stop daemon gracefully

**Narration Events:**
- `stopping` — Stopping daemon
- `shutdown_sent` — Graceful shutdown successful
- `waiting_stop` — Waiting for daemon to stop
- `stopped` — Daemon stopped

---

### Step 9: Test Result Reporting

**File:** `xtask/src/e2e/queen_lifecycle.rs`

```rust
println!("✅ E2E Test PASSED: Queen Lifecycle");
Ok(())
```

**Location:** Line 49  
**Purpose:** Report test success

**Console Output:**
```
🚀 E2E Test: Queen Lifecycle

📝 Running: rbee queen start

🔍 Finding binary: queen-rbee
✅ Found binary: target/debug/queen-rbee
🚀 Starting daemon: queen-rbee
✅ Daemon started with PID: 12345
🏥 Polling health endpoint: http://localhost:7833/health
⏳ Health check attempt 1/30
⏳ Connection failed - retrying...
⏳ Health check attempt 2/30
⏳ Connection failed - retrying...
⏳ Health check attempt 3/30
✅ Health check passed
✅ Daemon is healthy and responding
🎉 queen-rbee started successfully (PID: 12345)

📝 Running: rbee queen stop

🛑 Stopping daemon: queen-rbee
✅ Graceful shutdown successful
⏳ Waiting for daemon to stop...
✅ Daemon stopped

✅ E2E Test PASSED: Queen Lifecycle
```

---

## Key Files Summary

| File | Purpose | Key Functions |
|------|---------|---------------|
| `xtask/src/main.rs` | CLI dispatch | Command routing |
| `xtask/src/e2e/queen_lifecycle.rs` | Test harness | `test_queen_lifecycle()` |
| `bin/00_rbee_keeper/src/handlers/queen.rs` | Queen lifecycle | `handle_queen_lifecycle()` |
| `bin/96_lifecycle/lifecycle-local/src/start.rs` | Start daemon | `start_daemon()` |
| `bin/96_lifecycle/lifecycle-local/src/stop.rs` | Stop daemon | `stop_daemon()` |
| `bin/96_lifecycle/health-poll/src/lib.rs` | Health polling | `poll_health()` |

---

## Narration Events Summary

### Start Events

| Event | Message | Location |
|-------|---------|----------|
| `finding_binary` | "🔍 Finding binary: {daemon}" | start.rs:105 |
| `found_binary` | "✅ Found binary: {path}" | start.rs:109 |
| `starting` | "🚀 Starting daemon: {daemon}" | start.rs:113 |
| `started` | "✅ Daemon started with PID: {pid}" | start.rs:192 |
| `health_check` | "🏥 Polling health endpoint: {url}" | start.rs:195 |
| `health_attempt` | "⏳ Health check attempt {n}/{max}" | health-poll.rs:61 |
| `health_retry` | "⏳ Connection failed - retrying..." | health-poll.rs:80 |
| `health_success` | "✅ Health check passed" | health-poll.rs:66 |
| `healthy` | "✅ Daemon is healthy and responding" | start.rs:207 |
| `start_complete` | "🎉 {daemon} started successfully (PID: {pid})" | start.rs:208 |

### Stop Events

| Event | Message | Location |
|-------|---------|----------|
| `stopping` | "🛑 Stopping daemon: {daemon}" | stop.rs:55 |
| `shutdown_sent` | "✅ Graceful shutdown successful" | stop.rs:65 |
| `waiting_stop` | "⏳ Waiting for daemon to stop..." | stop.rs:75 |
| `stopped` | "✅ Daemon stopped" | stop.rs:85 |

---

## Performance Characteristics

### Typical Timing

- **Binary resolution:** <10ms
- **Process spawn:** ~50-100ms
- **Health check (3 attempts):** ~500-800ms
- **Total start:** ~600-1000ms
- **Graceful shutdown:** ~200-500ms
- **Total test:** ~1-2 seconds

---

## Testing

### Run Test

```bash
# Build binaries first
cargo build --bin rbee-keeper --bin queen-rbee

# Run E2E test
cargo xtask e2e:queen
```

### Expected Output

```
🚀 E2E Test: Queen Lifecycle

📝 Running: rbee queen start

✅ Daemon started with PID: 12345
✅ Health check passed
🎉 queen-rbee started successfully (PID: 12345)

📝 Running: rbee queen stop

✅ Graceful shutdown successful
✅ Daemon stopped

✅ E2E Test PASSED: Queen Lifecycle
```

---

**Status:** ✅ COMPLETE  
**Total Documentation:** ~1,000 lines  
**All components documented with exact file paths, narration events, and timing characteristics**
