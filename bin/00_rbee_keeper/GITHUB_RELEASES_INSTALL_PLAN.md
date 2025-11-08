# GitHub Releases Installation Plan

**Created by:** TEAM-451

---

## 🎯 Current Situation

**rbee-keeper install flow:**
```
rbee hive install localhost
  ↓
lifecycle-local::install_daemon()
  ↓
resolve_binary_path()
  ↓
ONLY builds from source (cargo build)
```

**Problem:** ❌ No GitHub releases support!
- Always builds from source (slow!)
- No production binary downloads
- Users can't get fast installs

---

## ✅ Desired Behavior

### Production Mode (default)
```bash
rbee hive install localhost
# Downloads from GitHub releases (FAST!)
# https://github.com/rbee-keeper/rbee/releases/download/v0.1.0/rbee-hive-linux-x86_64-0.1.0.tar.gz
```

### Development Mode
```bash
rbee hive install localhost --dev
# Builds from source (development branch)
# cargo build --release --package rbee-hive
```

---

## 📋 Implementation Plan

### 1. Update `lifecycle-local/src/install.rs`

**Add GitHub release download logic:**

```rust
pub async fn install_daemon(install_config: InstallConfig) -> Result<()> {
    let daemon_name = &install_config.daemon_name;
    
    // Step 1: Determine binary source
    let binary_path = if install_config.use_github_release {
        // NEW: Download from GitHub releases
        download_from_github_release(daemon_name).await?
    } else {
        // EXISTING: Build from source
        resolve_binary_path(
            daemon_name,
            install_config.local_binary_path,
            install_config.job_id.clone(),
        ).await?
    };
    
    // ... rest of install logic (copy, chmod, verify)
}

async fn download_from_github_release(daemon_name: &str) -> Result<PathBuf> {
    // 1. Detect platform and architecture
    let platform = detect_platform(); // linux, macos
    let arch = detect_architecture(); // x86_64, arm64
    
    // 2. Get version from Cargo.toml
    let version = env!("CARGO_PKG_VERSION");
    
    // 3. Build download URL
    let url = format!(
        "https://github.com/rbee-keeper/rbee/releases/download/v{}/{}-{}-{}-{}.tar.gz",
        version, daemon_name, platform, arch, version
    );
    
    // 4. Download to temp directory
    let temp_dir = std::env::temp_dir();
    let tarball_path = temp_dir.join(format!("{}.tar.gz", daemon_name));
    
    n!("downloading", "📥 Downloading from GitHub releases...");
    n!("download_url", "🔗 URL: {}", url);
    
    download_file(&url, &tarball_path).await?;
    
    // 5. Extract tarball
    n!("extracting", "📦 Extracting...");
    extract_tarball(&tarball_path, &temp_dir)?;
    
    // 6. Return path to extracted binary
    Ok(temp_dir.join(daemon_name))
}
```

### 2. Update `InstallConfig` struct

```rust
pub struct InstallConfig {
    pub daemon_name: String,
    pub local_binary_path: Option<PathBuf>,
    pub job_id: Option<String>,
    pub force_reinstall: bool,
    pub use_github_release: bool,  // NEW: Default true
}
```

### 3. Update CLI arguments

**In `rbee-keeper/src/handlers/hive_lifecycle.rs`:**

```rust
HiveLifecycleAction::Install {
    /// Host alias (default: localhost, or use SSH config entry)
    #[arg(short = 'a', long = "host", default_value = "localhost")]
    alias: String,
    
    /// Binary type (release for production, dev/debug for development)
    #[arg(short = 'b', long = "binary")]
    binary: Option<String>,
    
    /// Use development build (build from source)
    #[arg(long = "dev")]
    dev: bool,  // NEW
}
```

**Update handler:**

```rust
HiveLifecycleAction::Install { alias, binary, dev } => {
    if alias == "localhost" {
        let config = lifecycle_local::InstallConfig {
            daemon_name: "rbee-hive".to_string(),
            local_binary_path: binary.clone().map(|b| b.into()),
            job_id: None,
            force_reinstall: false,
            use_github_release: !dev,  // NEW: Use GitHub unless --dev
        };
        lifecycle_local::install_daemon(config).await
    } else {
        // ... same for SSH
    }
}
```

### 4. Same for Queen

Update `rbee-keeper/src/handlers/queen.rs` with same logic.

---

## 🔧 Helper Functions Needed

### Platform Detection
```rust
fn detect_platform() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        panic!("Unsupported platform")
    }
}

fn detect_architecture() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        panic!("Unsupported architecture")
    }
}
```

### File Download
```rust
async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let response = reqwest::get(url).await?;
    
    if !response.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", response.status());
    }
    
    let bytes = response.bytes().await?;
    std::fs::write(dest, bytes)?;
    
    Ok(())
}
```

### Tarball Extraction
```rust
fn extract_tarball(tarball: &Path, dest_dir: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;
    
    let file = std::fs::File::open(tarball)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    
    archive.unpack(dest_dir)?;
    
    Ok(())
}
```

---

## 📊 User Experience

### Before (current)
```bash
$ rbee hive install localhost
🔨 Building rbee-hive from source...
⏱️  Time: 2-5 minutes
```

### After (with GitHub releases)
```bash
$ rbee hive install localhost
📥 Downloading from GitHub releases...
🔗 URL: https://github.com/rbee-keeper/rbee/releases/download/v0.1.0/rbee-hive-linux-x86_64-0.1.0.tar.gz
📦 Extracting...
✅ rbee-hive installed successfully
⏱️  Time: 5-10 seconds
```

### Development mode
```bash
$ rbee hive install localhost --dev
🔨 Building from source (development branch)...
⏱️  Time: 2-5 minutes
```

---

## 🎯 Benefits

**For Users:**
- ✅ **10-60x faster** installation (5 seconds vs 5 minutes)
- ✅ No Rust toolchain required for production
- ✅ Clear choice: fast (prod) vs latest (dev)

**For Developers:**
- ✅ Can still build from source with `--dev`
- ✅ Easy testing of latest changes

**For rbee-keeper:**
- ✅ Matches worker installation pattern
- ✅ Consistent UX across all components

---

## 📝 Dependencies

**Add to `lifecycle-local/Cargo.toml`:**
```toml
[dependencies]
reqwest = { version = "0.11", features = ["blocking"] }
flate2 = "1.0"
tar = "0.4"
```

---

## ✅ Checklist

- [ ] Add GitHub release download to `lifecycle-local`
- [ ] Add GitHub release download to `lifecycle-ssh`
- [ ] Update `InstallConfig` with `use_github_release` flag
- [ ] Update CLI with `--dev` flag
- [ ] Update hive install handler
- [ ] Update queen install handler
- [ ] Add platform/arch detection helpers
- [ ] Add download_file helper
- [ ] Add extract_tarball helper
- [ ] Update tests
- [ ] Update documentation

---

## 🚀 Next Steps

1. Implement GitHub release download in `lifecycle-local`
2. Update CLI arguments
3. Test with actual GitHub releases
4. Update documentation
5. Deploy!
