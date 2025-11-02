# Remote Hive Start Part 1: Keeper Dispatch

**Flow:** CLI Command → Handler → Lifecycle Crate Selection  
**Date:** November 2, 2025  
**Status:** ✅ COMPLETE

---

## Overview

This document traces the flow from when a user types `rbee-keeper hive start --host remote-gpu-1` to when the appropriate lifecycle crate is selected for execution.

**Command Example:**
```bash
rbee-keeper hive start --host remote-gpu-1 --port 7835
```

---

## Step 1: CLI Command Parsing

### File: `bin/00_rbee_keeper/src/main.rs`

**Main Function:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Extract queen_url from config
    let config = Config::load()?;
    let queen_url = config.queen_url();  // Default: http://localhost:7833
    
    // Route command
    match cli.command {
        Commands::Hive { action } => {
            handle_hive_lifecycle(action, &queen_url).await
        }
        // ... other commands
    }
}
```

**Location:** Line 156  
**Purpose:** Route hive lifecycle commands to handler

---

## Step 2: Hive Lifecycle Action Enum

### File: `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs`

**Action Definition:**
```rust
#[derive(Subcommand)]
pub enum HiveLifecycleAction {
    /// Start rbee-hive
    Start {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
        
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    
    /// Stop rbee-hive
    Stop {
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    
    /// Check hive status
    Status {
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    
    // ... other actions (Install, Uninstall, Rebuild)
}
```

**Location:** Lines 22-69  
**Purpose:** Define CLI actions for hive lifecycle

---

## Step 3: Handler Entry Point

### File: `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs`

**Handler Function:**
```rust
pub async fn handle_hive_lifecycle(action: HiveLifecycleAction, queen_url: &str) -> Result<()> {
    match action {
        HiveLifecycleAction::Start { alias, port } => {
            let port = port.unwrap_or(7835);
            
            // TEAM-365: Conditional dispatch
            // localhost → lifecycle-local (no SSH)
            // remote → lifecycle-ssh (SSH-based)
            if alias == "localhost" {
                // Localhost path (Part 1a)
                handle_localhost_start(alias, port, queen_url).await
            } else {
                // Remote path (Part 1b)
                handle_remote_start(alias, port, queen_url).await
            }
        }
        // ... other actions
    }
}
```

**Location:** Lines 71-127  
**Function:** `handle_hive_lifecycle()`  
**Purpose:** Dispatch to localhost or remote handler

---

## Step 4a: Localhost Start (Reference)

**Localhost Flow:**
```rust
if alias == "localhost" {
    // Use lifecycle-local (no SSH)
    let base_url = format!("http://localhost:{}", port);
    let health_url = format!("{}/health", base_url);
    
    let args = vec![
        "--port".to_string(),
        port.to_string(),
        "--queen-url".to_string(),
        queen_url.to_string(),
        "--hive-id".to_string(),
        alias.clone(),
    ];
    
    let daemon_config = lifecycle_local::HttpDaemonConfig::new("rbee-hive", &health_url)
        .with_args(args);
    
    let config = lifecycle_local::StartConfig {
        daemon_config,
        job_id: None,
    };
    
    let _pid = lifecycle_local::start_daemon(config).await?;
}
```

**Location:** Lines 77-91  
**Purpose:** Start hive on localhost (no SSH)

**Key Details:**
- Uses `lifecycle-local` crate
- No SSH connection needed
- Direct process spawning
- Health check via localhost HTTP

---

## Step 4b: Remote Start (Main Flow)

**Remote Flow:**
```rust
else {
    // TEAM-365: Remote - use lifecycle-ssh
    
    // Step 4b.1: Resolve SSH config
    let ssh = resolve_ssh_config(&alias)?;
    
    // Step 4b.2: Construct URLs
    let base_url = format!("http://{}:{}", ssh.hostname, port);
    let health_url = format!("{}/health", base_url);
    
    // Step 4b.3: Get local IP for queen_url
    // TEAM-378: Remote hive needs Queen's network address, not localhost
    let local_ip = local_ip()
        .map_err(|e| anyhow::anyhow!("Failed to get local IP: {}", e))?;
    
    n!("detected_local_ip", "🔍 Detected local IP: {}", local_ip);
    n!("ssh_target", "🎯 SSH target: {}@{}", ssh.user, ssh.hostname);
    
    // Step 4b.4: Construct network-accessible queen_url
    let queen_port = queen_url.split(':').last()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(7833);
    let network_queen_url = format!("http://{}:{}", local_ip, queen_port);
    
    n!("remote_hive_queen_url", "🌐 Remote hive will use Queen at: {}", network_queen_url);
    n!("vite_dev_server", "🎨 Vite dev server will be at: http://{}:7836", local_ip);
    
    // Step 4b.5: Build daemon args
    let args = vec![
        "--port".to_string(),
        port.to_string(),
        "--queen-url".to_string(),
        network_queen_url,  // Network address, not localhost!
        "--hive-id".to_string(),
        alias.clone(),
    ];
    
    // Step 4b.6: Create daemon config
    let daemon_config = lifecycle_ssh::HttpDaemonConfig::new("rbee-hive", &health_url)
        .with_args(args);
    
    // Step 4b.7: Create start config
    let config = lifecycle_ssh::StartConfig {
        ssh_config: ssh,
        daemon_config,
        job_id: None,
    };
    
    // Step 4b.8: Start daemon via SSH
    let _pid = lifecycle_ssh::start_daemon(config).await?;
}
```

**Location:** Lines 92-125  
**Purpose:** Start hive on remote machine via SSH

**Narration Events:**
- `detected_local_ip` — Local IP detected
- `ssh_target` — SSH target identified
- `remote_hive_queen_url` — Queen URL for remote hive
- `vite_dev_server` — Vite dev server URL

---

## Step 5: SSH Config Resolution

### File: `bin/00_rbee_keeper/src/ssh_resolver.rs`

**Resolve SSH Config:**
```rust
/// Resolve SSH config for a host alias
///
/// TEAM-332: SSH config middleware - eliminates repeated SshConfig::localhost()
pub fn resolve_ssh_config(alias: &str) -> Result<SshConfig> {
    // Special case: localhost
    if alias == "localhost" {
        return Ok(SshConfig::localhost());
    }
    
    // Parse SSH config file
    let ssh_config_path = get_default_ssh_config_path();
    let targets = parse_ssh_config(&ssh_config_path)?;
    
    // Find matching target
    let target = targets
        .iter()
        .find(|t| t.host == alias)
        .ok_or_else(|| anyhow::anyhow!("SSH config entry '{}' not found", alias))?;
    
    // Convert to SshConfig
    Ok(SshConfig {
        hostname: target.hostname.clone(),
        user: target.user.clone().unwrap_or_else(|| "root".to_string()),
        port: target.port.unwrap_or(22),
    })
}
```

**Purpose:** Convert SSH alias to connection details

**Example:**
```
Input: "remote-gpu-1"

SSH Config:
Host remote-gpu-1
    HostName 192.168.1.100
    User rbee
    Port 22

Output: SshConfig {
    hostname: "192.168.1.100",
    user: "rbee",
    port: 22,
}
```

---

## Step 6: Local IP Detection

**Why Network Address Needed:**

**Problem:**
```
Keeper: http://localhost:7833  ← Works on keeper's machine
Remote Hive: http://localhost:7833  ← WRONG! Points to hive's localhost
```

**Solution:**
```rust
let local_ip = local_ip()?;  // e.g., "192.168.1.50"
let network_queen_url = format!("http://{}:{}", local_ip, queen_port);
// Result: "http://192.168.1.50:7833"  ← Accessible from remote hive!
```

**Why This Matters:**
- Remote hive needs to call back to queen
- Queen sends heartbeats to hive
- Hive registers with queen
- All communication requires network-accessible URL

---

## Step 7: Daemon Configuration

### HttpDaemonConfig

**Structure:**
```rust
pub struct HttpDaemonConfig {
    /// Daemon binary name (e.g., "rbee-hive")
    pub daemon_name: String,
    
    /// Health check URL (e.g., "http://192.168.1.100:7835/health")
    pub health_url: String,
    
    /// Command-line arguments
    pub args: Vec<String>,
}
```

**Example:**
```rust
let daemon_config = HttpDaemonConfig::new("rbee-hive", "http://192.168.1.100:7835/health")
    .with_args(vec![
        "--port".to_string(),
        "7835".to_string(),
        "--queen-url".to_string(),
        "http://192.168.1.50:7833".to_string(),
        "--hive-id".to_string(),
        "remote-gpu-1".to_string(),
    ]);
```

---

### StartConfig (SSH)

**Structure:**
```rust
pub struct StartConfig {
    /// SSH connection configuration
    pub ssh_config: SshConfig,
    
    /// Daemon configuration
    pub daemon_config: HttpDaemonConfig,
    
    /// Optional job_id for SSE narration routing
    pub job_id: Option<String>,
}
```

**Example:**
```rust
let config = StartConfig {
    ssh_config: SshConfig {
        hostname: "192.168.1.100".to_string(),
        user: "rbee".to_string(),
        port: 22,
    },
    daemon_config: daemon_config,
    job_id: None,  // No job_id for direct CLI commands
};
```

---

## Data Flow Summary

```
User Command
    ↓
rbee-keeper hive start --host remote-gpu-1 --port 7835
    ↓
main() [main.rs:156]
    ↓ parse CLI
    ↓ load config (queen_url)
    ↓
handle_hive_lifecycle() [hive_lifecycle.rs:71]
    ↓ match Start action
    ↓ extract alias and port
    ↓
Conditional Dispatch [hive_lifecycle.rs:76]
    ↓ if alias == "localhost" → lifecycle-local
    ↓ else → lifecycle-ssh
    ↓
resolve_ssh_config() [ssh_resolver.rs]
    ↓ parse ~/.ssh/config
    ↓ find "remote-gpu-1"
    ↓ return SshConfig { hostname, user, port }
    ↓
local_ip() [local-ip-address crate]
    ↓ detect local IP (e.g., 192.168.1.50)
    ↓
Construct network_queen_url
    ↓ http://192.168.1.50:7833
    ↓
Build daemon_config
    ↓ daemon_name: "rbee-hive"
    ↓ health_url: "http://192.168.1.100:7835/health"
    ↓ args: [--port, 7835, --queen-url, http://192.168.1.50:7833, ...]
    ↓
Build StartConfig
    ↓ ssh_config
    ↓ daemon_config
    ↓ job_id: None
    ↓
lifecycle_ssh::start_daemon() [lifecycle-ssh/start.rs]
    ↓ (continued in Part 2)
```

---

## Narration Events (Part 1)

| Event | Message | Location |
|-------|---------|----------|
| `detected_local_ip` | "🔍 Detected local IP: {ip}" | hive_lifecycle.rs:103 |
| `ssh_target` | "🎯 SSH target: {user}@{hostname}" | hive_lifecycle.rs:104 |
| `remote_hive_queen_url` | "🌐 Remote hive will use Queen at: {url}" | hive_lifecycle.rs:111 |
| `vite_dev_server` | "🎨 Vite dev server will be at: http://{ip}:7836" | hive_lifecycle.rs:112 |

---

## Key Files Referenced

| File | Purpose | Key Functions |
|------|---------|---------------|
| `bin/00_rbee_keeper/src/main.rs` | CLI entry | `main()` |
| `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs` | Lifecycle handler | `handle_hive_lifecycle()` |
| `bin/00_rbee_keeper/src/ssh_resolver.rs` | SSH config | `resolve_ssh_config()` |
| `bin/96_lifecycle/lifecycle-ssh/src/lib.rs` | SSH lifecycle | Types and exports |

---

## Configuration

**SSH Config:**
- Location: `~/.ssh/config`
- Format: Standard SSH config
- Required fields: Host, HostName
- Optional fields: User, Port

**Default Values:**
- Port: 7835 (hive)
- User: "root" (if not in SSH config)
- SSH Port: 22

---

## Error Handling

**Possible Errors:**
- SSH config not found
- SSH config entry not found
- Invalid SSH config format
- Local IP detection failed
- Invalid port number

**Error Messages:**
```
❌ SSH config entry 'remote-gpu-1' not found
❌ Failed to get local IP: Network unreachable
❌ Invalid port number
```

---

**Next:** [REMOTE_HIVE_START_PART_2_SSH_EXECUTION.md](./REMOTE_HIVE_START_PART_2_SSH_EXECUTION.md) — SSH daemon startup and health checking
