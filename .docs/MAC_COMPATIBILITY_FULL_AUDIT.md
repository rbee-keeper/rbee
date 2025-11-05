# Mac Compatibility - Complete Audit

**Branch**: `mac-compat`  
**Date**: 2025-11-05  
**Team**: TEAM-XXX

## Executive Summary

Comprehensive audit of the entire `bin/` codebase for mac compatibility issues. This document catalogs ALL platform-specific code and provides mac compatibility status.

---

## ✅ Already Fixed (Previous Session)

### 1. Build System
- ✅ WASM build order (`marketplace-node/package.json`)
- ✅ wasm-bindgen race condition (`scripts/build-all.sh`)
- ✅ Darwin statvfs u32/u64 mismatch (`audit-logging/src/writer.rs`)
- ✅ xtask binary path resolution (`xtask/src/tasks/rbee.rs`)
- ✅ Tauri deep-link plugin (temporarily disabled)

---

## ✅ Already Cross-Platform (No Changes Needed)

### 1. SSH Config Editor (`bin/00_rbee_keeper/src/tauri_commands.rs`)
**Status**: ✅ **WORKING**

```rust
// Lines 400-420 - Already has mac support
#[cfg(target_os = "linux")]
let status = Command::new("xdg-open")...

#[cfg(target_os = "macos")]
let status = Command::new("open")...  // ✅ CORRECT

#[cfg(target_os = "windows")]
let status = Command::new("notepad.exe")...
```

### 2. Platform Module (`bin/00_rbee_keeper/src/platform/`)
**Status**: ✅ **WORKING**

Has separate implementations:
- `linux.rs` - Uses `/proc/{pid}`, `kill` command
- `macos.rs` - Uses `ps`, `kill` command, `which` ✅
- `windows.rs` - Uses `tasklist`, `taskkill`

**Mac Implementation** (`macos.rs`):
```rust
fn bin_dir() -> Result<PathBuf> {
    Ok(PathBuf::from("/usr/local/bin"))  // ✅ CORRECT for mac
}

fn is_running(pid: u32) -> bool {
    Command::new("ps").arg("-p").arg(pid.to_string())...  // ✅ CORRECT
}
```

---

## ⚠️ Linux-Only Features (Gracefully Degraded on Mac)

### 1. Process Monitoring (`bin/25_rbee_hive_crates/monitor/`)
**Status**: ⚠️ **LINUX-ONLY** (returns stubs on mac)

**Files**:
- `src/monitor.rs` - cgroup v2 operations
- `src/telemetry.rs` - Worker telemetry collection
- `tests/process_monitor_tests.rs` - All tests `#[cfg(target_os = "linux")]`

**Linux-Specific Code**:
```rust
#[cfg(target_os = "linux")]
async fn spawn_linux_cgroup(...) {
    // Creates /sys/fs/cgroup/rbee.slice/{group}/{instance}/
    let cgroup_path = format!("/sys/fs/cgroup/rbee.slice/{}/{}", ...);
    fs::create_dir_all(&cgroup_path)?;
    // Applies CPU/memory limits via cgroup
}

#[cfg(target_os = "linux")]
async fn collect_stats_linux(...) {
    // Reads from /sys/fs/cgroup/rbee.slice/{group}/{instance}/
    let procs = fs::read_to_string(format!("{}/cgroup.procs", cgroup_path))?;
    let cpu_stat = fs::read_to_string(format!("{}/cpu.stat", cgroup_path))?;
    let mem_current = fs::read_to_string(format!("{}/memory.current", cgroup_path))?;
}
```

**Mac Behavior**:
- `spawn()` - Falls back to plain `tokio::process::Command::spawn()`
- `collect_stats()` - Returns stub with zeros
- `enumerate_all()` - Returns empty `Vec<ProcessStats>`

**Impact**: No resource limiting or telemetry on mac. Workers run but aren't monitored.

---

### 2. GPU Detection (`bin/25_rbee_hive_crates/device-detection/`)
**Status**: ⚠️ **NVIDIA-ONLY** (returns empty on mac)

**Files**:
- `src/detection.rs` - nvidia-smi detection
- `src/backend.rs` - CUDA backend detection

**nvidia-smi Usage**:
```rust
fn detect_via_nvidia_smi() -> Result<GpuInfo> {
    let nvidia_smi_path = which::which("nvidia-smi")?;
    let output = Command::new(&nvidia_smi_path)
        .args(["--query-gpu=index,name,memory.total,..."])
        .output()?;
    parse_nvidia_smi_output(&stdout)
}
```

**Mac Behavior**:
- Returns `GpuError::NvidiaSmiNotFound`
- Falls back to empty GPU list
- CUDA backend unavailable

**Impact**: No GPU acceleration on mac. CPU-only mode.

**Future**: Could add Metal detection for Apple Silicon:
```rust
#[cfg(target_os = "macos")]
fn detect_via_metal() -> Result<GpuInfo> {
    // Use Metal Performance Shaders API
    // Query MTLDevice for GPU info
}
```

---

### 3. GPU Monitoring (`bin/25_rbee_hive_crates/monitor/src/monitor.rs`)
**Status**: ⚠️ **LINUX-ONLY** (returns zeros on mac)

**nvidia-smi Queries**:
```rust
#[cfg(target_os = "linux")]
fn query_nvidia_smi(pid: u32) -> Result<(f64, u64)> {
    Command::new("nvidia-smi")
        .args(&["--query-compute-apps=pid,used_memory,sm", ...])
        .output()?;
    // Returns (gpu_util%, vram_mb)
}

#[cfg(target_os = "linux")]
fn query_total_gpu_vram() -> Result<u64> {
    Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.total", ...])
        .output()?;
    // Returns total VRAM in MB
}
```

**Mac Behavior**: Returns `(0.0, 0)` for GPU stats

---

### 4. Process Info (`bin/25_rbee_hive_crates/monitor/src/monitor.rs`)
**Status**: ⚠️ **LINUX-ONLY** (not available on mac)

**`/proc` Filesystem Access**:
```rust
#[cfg(target_os = "linux")]
fn extract_model_from_cmdline(pid: u32) -> Result<Option<String>> {
    let cmdline_path = format!("/proc/{}/cmdline", pid);
    let cmdline = fs::read_to_string(&cmdline_path)?;
    // Parse --model argument
}

#[cfg(target_os = "linux")]
fn get_process_uptime(pid: u32) -> Result<u64> {
    let stat = fs::read_to_string(format!("/proc/{}/stat", pid))?;
    let uptime_str = fs::read_to_string("/proc/uptime")?;
    // Calculate process uptime
}
```

**Mac Alternative**: Could use `libproc` or `sysctl`:
```rust
#[cfg(target_os = "macos")]
fn get_process_uptime(pid: u32) -> Result<u64> {
    use std::process::Command;
    let output = Command::new("ps")
        .args(&["-p", &pid.to_string(), "-o", "etime="])
        .output()?;
    // Parse elapsed time
}
```

---

### 5. Platform Module - Linux Implementation
**Status**: ⚠️ **LINUX-ONLY** (mac has separate impl)

**File**: `bin/00_rbee_keeper/src/platform/linux.rs`

```rust
impl PlatformProcess for LinuxPlatform {
    fn is_running(pid: u32) -> bool {
        std::path::Path::new(&format!("/proc/{}", pid)).exists()  // ❌ Linux-only
    }
}
```

**Mac Has**: Separate `macos.rs` with `ps` command ✅

---

## 📝 Documentation References (No Code Changes)

### Archive/Docs Mentioning Linux Packages
**Status**: ℹ️ **DOCUMENTATION ONLY**

**Files** (all in `.archive/` or `docs/`):
- `bin/80-hono-worker-catalog/.archive/docs/WORKER_CATALOG_DESIGN.md` - AUR/pacman examples
- `bin/00_rbee_keeper/.archive/CROSS_PLATFORM.md` - Build deps for Arch/Ubuntu
- `bin/80-hono-worker-catalog/.archive/docs/VISION.md` - Package manager examples
- `bin/31_sd_worker_rbee/.windsurf/.archive/TEAM_401_PHASE_11_POLISH.md` - Docker apt-get
- `bin/30_llm_worker_rbee/.specs/TEAM_006_CRITICAL_REVIEW.md` - Profiling tools

**Impact**: None. These are documentation/archive files, not runtime code.

---

## 🔍 Test Files (Linux-Only Tests)

### Process Monitor Tests
**Status**: ⚠️ **TESTS DISABLED ON MAC**

**File**: `bin/25_rbee_hive_crates/monitor/tests/process_monitor_tests.rs`

All tests marked with:
```rust
#[tokio::test]
#[cfg(target_os = "linux")]
async fn test_spawn_creates_cgroup() { ... }
```

**Tests**:
- `test_spawn_creates_cgroup` - Verifies `/sys/fs/cgroup/` creation
- `test_spawn_applies_cpu_limit` - Checks `cpu.max` file
- `test_spawn_applies_memory_limit` - Checks `memory.max` file
- `test_collect_reads_cgroup_stats` - Reads cgroup stats
- `test_collect_queries_nvidia_smi` - GPU monitoring
- `test_enumerate_walks_cgroup_tree` - Walks `/sys/fs/cgroup/`

**Mac Behavior**: Tests are skipped (not compiled)

---

## 🛠️ Recommended Mac Implementations

### Priority 1: Process Monitoring (High Value)

**Current**: Stubs return zeros  
**Proposed**: Use macOS APIs

```rust
#[cfg(target_os = "macos")]
async fn collect_stats_macos(group: &str, instance: &str) -> Result<ProcessStats> {
    use std::process::Command;
    
    // Get PID from instance (e.g., port number)
    let pid = find_pid_by_port(instance)?;
    
    // Use ps for CPU/memory
    let output = Command::new("ps")
        .args(&["-p", &pid.to_string(), "-o", "pid,%cpu,%mem,vsz,rss,etime"])
        .output()?;
    
    // Parse output
    let stats = parse_ps_output(&output.stdout)?;
    
    Ok(ProcessStats {
        pid,
        cpu_percent: stats.cpu,
        memory_mb: stats.rss / 1024,
        uptime_secs: stats.uptime,
        // GPU stats remain 0 (no NVIDIA on mac)
        gpu_util_percent: 0.0,
        vram_mb: 0,
        total_vram_mb: 0,
        model_name: None,
    })
}
```

### Priority 2: GPU Detection for Apple Silicon

**Current**: Returns empty  
**Proposed**: Use Metal APIs

```rust
#[cfg(target_os = "macos")]
fn detect_via_metal() -> Result<GpuInfo> {
    // Use Metal framework
    // This would require metal-rs or direct FFI
    // Query MTLDevice for:
    // - Device name
    // - Recommended max working set size (VRAM equivalent)
    // - Feature set
    
    // For now, return stub indicating Metal availability
    Ok(GpuInfo {
        devices: vec![GpuDevice {
            index: 0,
            name: "Apple GPU (Metal)".to_string(),
            vram_total_mb: 0, // Metal uses unified memory
            vram_free_mb: 0,
            compute_capability: (0, 0),
            pci_bus_id: "N/A".to_string(),
        }],
        backend: Backend::Metal,
    })
}
```

### Priority 3: Resource Limiting

**Current**: No limits enforced  
**Proposed**: Use macOS Job Control

```rust
#[cfg(target_os = "macos")]
async fn spawn_with_limits_macos(
    config: MonitorConfig,
    binary_path: &str,
    args: Vec<String>,
) -> Result<u32> {
    // macOS doesn't have cgroups, but has alternatives:
    // 1. launchd plist with resource limits
    // 2. setrlimit() syscall
    // 3. Process groups with resource accounting
    
    use std::process::Command;
    
    let mut cmd = Command::new(binary_path);
    cmd.args(&args);
    
    // Could use setrlimit via libc:
    // unsafe {
    //     let rlim = libc::rlimit {
    //         rlim_cur: memory_limit,
    //         rlim_max: memory_limit,
    //     };
    //     libc::setrlimit(libc::RLIMIT_AS, &rlim);
    // }
    
    let child = cmd.spawn()?;
    Ok(child.id().unwrap())
}
```

---

## 📊 Summary Table

| Feature | Linux | macOS | Windows | Priority |
|---------|-------|-------|---------|----------|
| **Build System** | ✅ | ✅ | ❓ | - |
| **SSH Config Editor** | ✅ | ✅ | ✅ | - |
| **Platform Module** | ✅ | ✅ | ✅ | - |
| **Process Spawn** | ✅ | ✅ (no limits) | ✅ | Medium |
| **Process Monitoring** | ✅ (cgroup) | ⚠️ (stubs) | ⚠️ (stubs) | High |
| **Resource Limits** | ✅ (cgroup) | ❌ | ❌ | Medium |
| **GPU Detection** | ✅ (NVIDIA) | ❌ | ✅ (NVIDIA) | Low |
| **GPU Monitoring** | ✅ (nvidia-smi) | ❌ | ✅ (nvidia-smi) | Low |
| **Telemetry** | ✅ | ⚠️ (partial) | ⚠️ (partial) | High |

**Legend**:
- ✅ Fully working
- ⚠️ Partial/degraded (returns stubs)
- ❌ Not implemented
- ❓ Unknown/untested

---

## 🎯 Action Items

### Immediate (Already Done)
- [x] Fix build system issues
- [x] Fix Rust type mismatches
- [x] Fix xtask binary resolution
- [x] Disable problematic Tauri plugins

### Short Term (Recommended)
- [ ] Implement macOS process monitoring using `ps`
- [ ] Add process uptime calculation for mac
- [ ] Document mac limitations in user-facing docs

### Medium Term (Nice to Have)
- [ ] Implement resource limits using `setrlimit()`
- [ ] Add Metal GPU detection for Apple Silicon
- [ ] Create mac-specific tests

### Long Term (Future)
- [ ] Full Metal backend support for inference
- [ ] macOS-native monitoring dashboard
- [ ] Unified cross-platform monitoring API

---

## 🔧 Files Requiring Mac-Specific Code

### High Priority
1. `bin/25_rbee_hive_crates/monitor/src/monitor.rs`
   - Add `#[cfg(target_os = "macos")]` implementations
   - Use `ps` for process stats
   - Use `sysctl` for system info

2. `bin/25_rbee_hive_crates/monitor/src/telemetry.rs`
   - Add mac collection paths
   - Return actual data instead of stubs

### Medium Priority
3. `bin/25_rbee_hive_crates/device-detection/src/detection.rs`
   - Add Metal detection
   - Query unified memory on Apple Silicon

### Low Priority
4. Test files - Add mac-specific tests
5. Documentation - Update with mac limitations

---

## 📚 References

### macOS APIs for Process Monitoring
- `libproc` - Process information library
- `sysctl` - System information queries
- `ps` command - Process status
- `top` command - System monitoring

### macOS APIs for GPU
- Metal Performance Shaders (MPS)
- `MTLDevice` - GPU device queries
- Unified Memory Architecture

### Resource Limiting
- `setrlimit()` - POSIX resource limits
- `launchd` - Service management with limits
- Process groups - Resource accounting

---

**Last Updated**: 2025-11-05  
**Status**: ✅ Build working, ⚠️ Monitoring degraded, 📋 Roadmap defined
