# Port Violations Report

**TEAM-XXX**: Comprehensive audit of hardcoded worker ports that violate dynamic allocation

**Date**: 2025-11-08  
**Status**: 🔴 CRITICAL - Multiple violations found

---

## ✅ What We Fixed

### 1. PORT_CONFIGURATION.md (Canonical Source)
- ✅ Updated to reflect dynamic port allocation (8080+)
- ✅ Removed fixed port assignments for workers
- ✅ Added critical warning about dynamic allocation
- ✅ Documented PortAssigner component

### 2. Created PortAssigner Component
- ✅ Location: `/bin/25_rbee_hive_crates/port-assigner/`
- ✅ Thread-safe port allocation starting from 8080
- ✅ Sequential assignment with wraparound
- ✅ Port release and reuse functionality
- ✅ Comprehensive test coverage

---

## 🔴 VIOLATIONS FOUND - Must Fix

### Critical: Worker Binaries with Hardcoded Default Ports

#### 1. SD Worker - Port 8081 Hardcoded
**Files:**
- `/bin/31_sd_worker_rbee/src/bin/cpu.rs:31`
- `/bin/31_sd_worker_rbee/src/bin/cuda.rs:31`
- `/bin/31_sd_worker_rbee/src/bin/metal.rs:31`

```rust
/// HTTP server port
#[arg(long, env = "PORT", default_value = "8081")]  // ❌ WRONG
port: u16,
```

**Fix Required:**
```rust
/// HTTP server port - MUST be provided by hive (no default)
#[arg(long, env = "PORT")]  // ✅ CORRECT - No default
port: u16,
```

**Impact:** HIGH - Workers will use wrong port if hive doesn't provide one

---

### Medium: Documentation Examples with Hardcoded Ports

#### 2. Worker HTTP Server Documentation
**Files:**
- `/bin/31_sd_worker_rbee/src/http/server.rs:47,60,89`
- `/bin/30_llm_worker_rbee/src/http/server.rs` (similar)

```rust
/// # Example
/// ```
/// let addr: SocketAddr = "0.0.0.0:8080".parse()?;  // ❌ Misleading
/// ```
```

**Fix Required:**
```rust
/// # Example
/// ```
/// // Port assigned dynamically by hive
/// let port = 8080; // Example only - actual port from hive
/// let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
/// ```
```

**Impact:** MEDIUM - Misleading documentation

---

#### 3. Health Check Documentation
**Files:**
- `/bin/31_sd_worker_rbee/src/http/health.rs:45,52`
- `/bin/31_sd_worker_rbee/src/http/ready.rs:57,64`

```rust
/// # Example
/// ```bash
/// curl http://localhost:8080/health  // ❌ Hardcoded port
/// ```
```

**Fix Required:**
```rust
/// # Example
/// ```bash
/// # Port is dynamically assigned by hive
/// curl http://localhost:${WORKER_PORT}/health
/// ```
```

**Impact:** MEDIUM - Misleading documentation

---

### Low: Test Files with Example Ports

#### 4. Monitor Tests
**Files:**
- `/bin/25_rbee_hive_crates/monitor/tests/telemetry_collection_tests.rs:18-21`
- `/bin/25_rbee_hive_crates/monitor/tests/process_monitor_tests.rs:390,396`

```rust
let workers = vec![
    ("llm", "8080"),  // ❌ Hardcoded for tests
    ("llm", "8081"),
];
```

**Fix Required:**
```rust
// These are test fixtures - ports don't matter
// In production, hive assigns ports dynamically
let workers = vec![
    ("llm", "9001"),  // Test port (not real)
    ("llm", "9002"),  // Test port (not real)
];
```

**Impact:** LOW - Test-only, but should use different range to avoid confusion

---

#### 5. Keeper Config Tests
**Files:**
- `/bin/97_contracts/keeper-config-contract/src/config.rs:85-86`

```rust
#[test]
fn test_queen_url() {
    let config = KeeperConfig { queen_port: 8080 };  // ❌ Wrong port
    assert_eq!(config.queen_url(), "http://localhost:8080");
}
```

**Fix Required:**
```rust
#[test]
fn test_queen_url() {
    let config = KeeperConfig { queen_port: 7833 };  // ✅ Correct queen port
    assert_eq!(config.queen_url(), "http://localhost:7833");
}
```

**Impact:** LOW - Test uses wrong port number

---

### Info: Frontend Shared Config

#### 6. Frontend shared-config Package
**Files:**
- `/frontend/packages/shared-config/src/ports.ts`
- `/frontend/packages/shared-config/src/ports.test.ts`

**Current State:**
```typescript
worker: {
  llm: { dev: 7837, prod: 8080, backend: 8080 },  // ❌ Fixed backend port
  sd: { dev: 5174, prod: 8081, backend: 8081 },   // ❌ Fixed backend port
  comfy: { dev: 7838, prod: 8188, backend: 8188 }, // ❌ Fixed backend port
  vllm: { dev: 7839, prod: 8000, backend: 8000 },  // ❌ Fixed backend port
}
```

**Fix Required:**
- Remove `prod` and `backend` ports for workers
- Keep only `dev` ports (for local development)
- Workers in production use dynamic ports assigned by hive
- Frontend should query hive for actual worker URLs

**Impact:** MEDIUM - Frontend assumes fixed worker ports

---

## 📋 Summary

### Violations by Severity

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 HIGH | 3 | Worker binaries with hardcoded default ports |
| 🟡 MEDIUM | 5 | Misleading documentation examples |
| 🟢 LOW | 3 | Test files with example ports |
| ℹ️ INFO | 2 | Frontend config needs update |

### Total Files Affected: 13

---

## 🔧 Action Plan

### Priority 1: Fix Worker Binaries (HIGH)
1. Remove `default_value = "8081"` from SD worker CLIs
2. Remove `default_value = "8080"` from LLM worker CLIs
3. Ensure hive ALWAYS provides port when spawning workers

### Priority 2: Update Documentation (MEDIUM)
1. Fix HTTP server examples to show dynamic ports
2. Fix health check examples to use variable ports
3. Add comments explaining dynamic allocation

### Priority 3: Fix Tests (LOW)
1. Update test fixtures to use different port range (9000+)
2. Fix keeper config test to use correct queen port (7833)
3. Add comments explaining test vs production ports

### Priority 4: Update Frontend (MEDIUM)
1. Remove hardcoded worker backend ports from shared-config
2. Update frontend to query hive for worker URLs
3. Keep dev ports for local development only

---

## 🎯 Next Steps

1. **Integrate PortAssigner into rbee-hive**
   - Add `port-assigner` to hive's Cargo.toml
   - Use PortAssigner when spawning workers
   - Track assigned ports in worker registry

2. **Remove Default Ports from Workers**
   - Update all worker CLIs to require port argument
   - Remove default_value from port arguments

3. **Update Documentation**
   - Fix all code examples
   - Add dynamic port allocation guide

4. **Update Frontend**
   - Remove hardcoded worker ports
   - Query hive for actual worker URLs

---

## ✅ Verification Checklist

- [ ] PortAssigner component created and tested
- [ ] PORT_CONFIGURATION.md updated
- [ ] Worker binaries require port argument (no defaults)
- [ ] Hive uses PortAssigner to assign ports
- [ ] Documentation updated with dynamic port examples
- [ ] Test files use different port range
- [ ] Frontend queries hive for worker URLs
- [ ] All hardcoded 8080, 8081, 8188, 8000 removed from worker code

---

**CRITICAL:** Workers MUST NOT have default ports. The hive is responsible for port assignment using the PortAssigner component.
