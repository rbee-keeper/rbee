# Remote Hive Start Flow Documentation

**Complete roundtrip: Keeper → SSH → Hive → Health Check → Success**  
**Date:** November 2, 2025  
**Status:** ✅ Part 1 Complete, Parts 2-3 Outlined

---

## Overview

This directory contains documentation of the remote hive start flow, showing how rbee-keeper starts a hive daemon on a remote machine via SSH.

**Flow Summary:**
```
User Command
    ↓
Keeper CLI (Part 1)
    ↓ Parse command
    ↓ Resolve SSH config
    ↓ Detect local IP
    ↓ Build daemon config
    ↓
SSH Execution (Part 2)
    ↓ Find binary on remote
    ↓ Start daemon with nohup
    ↓ Capture PID
    ↓
Health Check (Part 2)
    ↓ Poll HTTP endpoint
    ↓ Exponential backoff
    ↓ Verify daemon running
    ↓
Success Response (Part 3)
    ↓ Return PID
    ↓ Print success message
```

---

## Documentation Parts

### ✅ [Part 1: Keeper Dispatch](./REMOTE_HIVE_START_PART_1_KEEPER_DISPATCH.md)

**Scope:** CLI Command → Handler → Lifecycle Crate Selection

**Key Topics:**
- CLI command parsing
- Hive lifecycle action enum
- Conditional dispatch (localhost vs remote)
- SSH config resolution
- Local IP detection
- Network-accessible queen_url
- Daemon configuration

**Status:** Fully documented

**Key Files:**
- `bin/00_rbee_keeper/src/main.rs`
- `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs`
- `bin/00_rbee_keeper/src/ssh_resolver.rs`

**Narration Events:**
- `detected_local_ip` — Local IP detected
- `ssh_target` — SSH target identified
- `remote_hive_queen_url` — Queen URL for remote
- `vite_dev_server` — Vite dev server URL

---

### 📋 Part 2: SSH Execution & Health Check (OUTLINED)

**Scope:** SSH Commands → Daemon Start → Health Polling

**Key Topics:**
- Find binary on remote machine
- Start daemon with nohup
- Capture PID from stdout
- HTTP health check polling
- Exponential backoff
- Timeout handling

**Key Files:**
- `bin/96_lifecycle/lifecycle-ssh/src/start.rs`
- `bin/96_lifecycle/lifecycle-ssh/src/utils/ssh.rs`
- `bin/99_shared_crates/health-poll/src/lib.rs`

**SSH Commands:**
1. **Find Binary:**
   ```bash
   (test -x target/debug/rbee-hive && echo target/debug/rbee-hive) || \
   (test -x target/release/rbee-hive && echo target/release/rbee-hive) || \
   (test -x ~/.local/bin/rbee-hive && echo ~/.local/bin/rbee-hive) || \
   which rbee-hive 2>/dev/null || \
   echo 'NOT_FOUND'
   ```

2. **Start Daemon:**
   ```bash
   nohup /path/to/rbee-hive --port 7835 --queen-url http://192.168.1.50:7833 --hive-id remote-gpu-1 > /dev/null 2>&1 & echo $!
   ```

**Health Check:**
```rust
poll_health(
    "http://192.168.1.100:7835/health",
    30,    // max attempts
    200,   // initial delay (ms)
    1.5    // backoff multiplier
).await
```

**Narration Events:**
- `ssh_find_binary` — Finding binary
- `ssh_binary_found` — Binary found
- `ssh_start_daemon` — Starting daemon
- `ssh_daemon_started` — Daemon started with PID
- `health_poll_start` — Starting health check
- `health_poll_attempt` — Health check attempt
- `health_poll_success` — Health check passed

---

### 📋 Part 3: Error Handling & Response (OUTLINED)

**Scope:** Error Scenarios → Success/Failure Response

**Key Topics:**
- SSH connection failures
- Binary not found errors
- Daemon start failures
- Health check timeouts
- Success response
- Error propagation

**Error Scenarios:**

1. **SSH Connection Failed:**
   ```
   ❌ SSH connection failed: Connection refused
   Exit code: 1
   ```

2. **Binary Not Found:**
   ```
   ❌ Binary 'rbee-hive' not found on remote machine
   Exit code: 1
   ```

3. **Daemon Start Failed:**
   ```
   ❌ Failed to start daemon: Permission denied
   Exit code: 1
   ```

4. **Health Check Timeout:**
   ```
   ❌ Daemon started but failed health check after 30 attempts
   Exit code: 1
   ```

5. **Success:**
   ```
   ✅ Hive started successfully on remote-gpu-1 (PID: 12345)
   Exit code: 0
   ```

**Narration Events:**
- `ssh_error` — SSH connection error
- `binary_not_found` — Binary not found
- `daemon_start_error` — Daemon start failed
- `health_check_timeout` — Health check timeout
- `hive_start_success` — Hive started successfully

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ KEEPER (Local Machine)                                      │
├─────────────────────────────────────────────────────────────┤
│ 1. User runs: rbee-keeper hive start --host remote-gpu-1   │
│    ↓                                                        │
│ 2. Parse CLI command                                        │
│    ├─→ alias: "remote-gpu-1"                               │
│    └─→ port: 7835                                          │
│    ↓                                                        │
│ 3. Resolve SSH config                                       │
│    ├─→ Read ~/.ssh/config                                  │
│    ├─→ Find "remote-gpu-1" entry                           │
│    └─→ Extract: hostname, user, port                       │
│    ↓                                                        │
│ 4. Detect local IP                                          │
│    ├─→ Query network interfaces                            │
│    └─→ Result: 192.168.1.50                                │
│    ↓                                                        │
│ 5. Build network-accessible queen_url                       │
│    ├─→ Input: http://localhost:7833                        │
│    └─→ Output: http://192.168.1.50:7833                    │
│    ↓                                                        │
│ 6. Build daemon config                                      │
│    ├─→ daemon_name: "rbee-hive"                            │
│    ├─→ health_url: "http://192.168.1.100:7835/health"     │
│    └─→ args: [--port, 7835, --queen-url, ...]             │
│    ↓                                                        │
│ 7. Call lifecycle_ssh::start_daemon()                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ SSH EXECUTION (Remote Machine: 192.168.1.100)              │
├─────────────────────────────────────────────────────────────┤
│ 1. SSH connect to remote-gpu-1                             │
│    ├─→ User: rbee                                          │
│    ├─→ Host: 192.168.1.100                                 │
│    └─→ Port: 22                                            │
│    ↓                                                        │
│ 2. Find binary on remote                                    │
│    ├─→ Try: target/debug/rbee-hive                         │
│    ├─→ Try: target/release/rbee-hive                       │
│    ├─→ Try: ~/.local/bin/rbee-hive                         │
│    ├─→ Try: which rbee-hive                                │
│    └─→ Result: /home/rbee/.local/bin/rbee-hive            │
│    ↓                                                        │
│ 3. Start daemon with nohup                                  │
│    ├─→ Command: nohup /path/to/rbee-hive [args] & echo $! │
│    ├─→ Redirect: > /dev/null 2>&1                          │
│    └─→ Capture PID: 12345                                  │
│    ↓                                                        │
│ 4. Daemon initializes                                       │
│    ├─→ Parse CLI args                                      │
│    ├─→ Initialize Tokio runtime                            │
│    ├─→ Start HTTP server on port 7835                      │
│    ├─→ Register with queen (http://192.168.1.50:7833)     │
│    └─→ Start heartbeat task                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HEALTH CHECK (From Keeper)                                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Poll health endpoint                                     │
│    ├─→ URL: http://192.168.1.100:7835/health              │
│    ├─→ Max attempts: 30                                    │
│    ├─→ Initial delay: 200ms                                │
│    └─→ Backoff: 1.5x per attempt                           │
│    ↓                                                        │
│ 2. Attempt 1: Connection refused (daemon not ready)        │
│    ├─→ Wait 200ms                                          │
│    └─→ Retry                                               │
│    ↓                                                        │
│ 3. Attempt 2: Connection refused                           │
│    ├─→ Wait 300ms (200 × 1.5)                              │
│    └─→ Retry                                               │
│    ↓                                                        │
│ 4. Attempt 3: 200 OK ✅                                     │
│    └─→ Health check passed!                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ SUCCESS RESPONSE                                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Return PID to keeper                                     │
│    └─→ PID: 12345                                          │
│    ↓                                                        │
│ 2. Print success message                                    │
│    └─→ "✅ Hive started successfully on remote-gpu-1"      │
│    ↓                                                        │
│ 3. Exit code: 0                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

### 1. Network-Accessible Queen URL

**Problem:**
```
Keeper: http://localhost:7833  ← Works on keeper's machine
Remote Hive: http://localhost:7833  ← WRONG! Points to hive's localhost
```

**Solution:**
```
Detect local IP: 192.168.1.50
Use: http://192.168.1.50:7833  ← Accessible from remote hive
```

**Why This Matters:**
- Remote hive needs to call back to queen
- Queen discovery sends requests to hive
- Hive registers with queen
- Bidirectional communication required

---

### 2. Two SSH Commands Only

**Design Goal:** Minimize SSH overhead

**Commands:**
1. Find binary (one SSH call)
2. Start daemon (one SSH call)

**Health Check:** HTTP only (no SSH)

**Benefits:**
- ✅ Faster execution
- ✅ Less SSH overhead
- ✅ Simpler error handling
- ✅ HTTP health check more reliable

---

### 3. Exponential Backoff

**Health Check Strategy:**
```
Attempt 1: Wait 200ms
Attempt 2: Wait 300ms (200 × 1.5)
Attempt 3: Wait 450ms (300 × 1.5)
Attempt 4: Wait 675ms (450 × 1.5)
...
Max: 30 attempts
```

**Why Exponential:**
- ✅ Fast initial checks (daemon might be ready quickly)
- ✅ Longer waits later (daemon might take time to initialize)
- ✅ Prevents overwhelming the remote machine
- ✅ Total timeout: ~30-60 seconds

---

### 4. Conditional Dispatch

**Localhost vs Remote:**
```rust
if alias == "localhost" {
    lifecycle_local::start_daemon()  // No SSH
} else {
    lifecycle_ssh::start_daemon()    // SSH-based
}
```

**Benefits:**
- ✅ Optimal performance for localhost
- ✅ SSH only when needed
- ✅ Same API for both paths
- ✅ Easy to test locally

---

## SSH Config Format

**Example ~/.ssh/config:**
```
# GPU Hives
Host remote-gpu-1
    HostName 192.168.1.100
    User rbee
    Port 22

Host remote-gpu-2
    HostName 192.168.1.101
    User rbee
    Port 22

# CPU Hives
Host remote-cpu-1
    HostName 192.168.1.200
    User rbee
    Port 22
```

**Required Fields:**
- `Host` — Alias used in CLI
- `HostName` — IP address or hostname

**Optional Fields:**
- `User` — SSH user (default: "root")
- `Port` — SSH port (default: 22)

---

## Daemon Arguments

**Generated Args:**
```bash
rbee-hive \
  --port 7835 \
  --queen-url http://192.168.1.50:7833 \
  --hive-id remote-gpu-1
```

**Why These Args:**
- `--port` — HTTP server port
- `--queen-url` — Network-accessible queen URL
- `--hive-id` — Unique identifier for this hive

---

## Error Handling Summary

### SSH Errors

| Error | Message | Exit Code |
|-------|---------|-----------|
| Connection refused | "❌ SSH connection failed: Connection refused" | 1 |
| Authentication failed | "❌ SSH authentication failed" | 1 |
| Host unreachable | "❌ SSH host unreachable" | 1 |

### Binary Errors

| Error | Message | Exit Code |
|-------|---------|-----------|
| Not found | "❌ Binary 'rbee-hive' not found on remote machine" | 1 |
| Not executable | "❌ Binary found but not executable" | 1 |

### Daemon Errors

| Error | Message | Exit Code |
|-------|---------|-----------|
| Start failed | "❌ Failed to start daemon: {error}" | 1 |
| Permission denied | "❌ Permission denied" | 1 |

### Health Check Errors

| Error | Message | Exit Code |
|-------|---------|-----------|
| Timeout | "❌ Daemon started but failed health check after 30 attempts" | 1 |
| Connection refused | "❌ Health check failed: Connection refused" | 1 |

### Success

| Status | Message | Exit Code |
|--------|---------|-----------|
| Success | "✅ Hive started successfully on {alias} (PID: {pid})" | 0 |

---

## Performance Characteristics

### Typical Latency

- **SSH connection:** ~100-500ms
- **Find binary:** ~50-200ms
- **Start daemon:** ~100-300ms
- **Health check:** ~1-5 seconds (depends on daemon init)
- **Total:** ~2-6 seconds

### Optimization

**Fast Path (daemon ready quickly):**
- 3 attempts × 200ms = ~600ms health check
- Total: ~2 seconds

**Slow Path (daemon takes time):**
- 10 attempts × exponential backoff = ~5 seconds health check
- Total: ~6 seconds

---

## Testing Strategy

### Unit Tests

- [ ] SSH config resolution
- [ ] Local IP detection
- [ ] Daemon config building
- [ ] Network queen_url construction

### Integration Tests

- [ ] Localhost start (no SSH)
- [ ] Remote start (with SSH)
- [ ] Health check polling
- [ ] Error scenarios

### E2E Tests

- [ ] Full roundtrip (keeper → SSH → hive → health → success)
- [ ] Multiple concurrent starts
- [ ] Failure recovery

---

## Related Documentation

- [Job Flow Documentation](./README.md) — Job submission and execution
- [Capabilities Discovery](./CAPABILITIES_FLOW_README.md) — Hive discovery
- [SSH Discovery Flow](./SSH_DISCOVERY_FLOW_COMPLETE.md) — Queen → Hive discovery

---

**Status:** Part 1 complete with full detail, Parts 2-3 outlined  
**Maintainer:** TEAM-385+  
**Last Updated:** November 2, 2025
