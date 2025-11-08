# Release xtask Implementation Plan

**Created by:** TEAM-451  
**Status:** Implementation Plan  
**Platform:** Cross-platform (Linux, macOS, Windows)

## 🎯 Goal

Create `cargo xtask release` - an **interactive** version manager that:
1. Prompts user for tier and bump type
2. Updates all relevant Cargo.toml and package.json files
3. Commits changes
4. Works on **macOS** (for your Mac CI)
5. Works on **Linux** (for GitHub Actions)

---

## 📋 Phase 1: Setup Dependencies

### 1.1 Update xtask/Cargo.toml

```toml
# xtask/Cargo.toml
[package]
name = "xtask"
version = "0.1.0"
edition = "2021"

[dependencies]
# Existing dependencies
# ... (keep existing)

# NEW: Release management dependencies
inquire = "0.7"           # Interactive prompts
toml_edit = "0.22"        # TOML parsing (preserves formatting)
serde_json = "1.0"        # JSON parsing
anyhow = "1.0"            # Error handling
thiserror = "1.0"         # Custom errors
colored = "2.1"           # Terminal colors
semver = "1.0"            # Semantic versioning
walkdir = "2.4"           # Directory traversal
regex = "1.10"            # Pattern matching

# Git operations (optional - can use Command instead)
# git2 = "0.18"

# CLI (already have clap)
clap = { version = "4", features = ["derive"] }
```

### 1.2 Install on macOS

```bash
# On your Mac
cd ~/Projects/rbee

# Update dependencies
cargo build --package xtask

# Verify it works
cargo xtask --help
```

---

## 📋 Phase 2: Define Tier Configurations

### 2.1 Create Tier Config Files

```bash
mkdir -p .version-tiers
```

### 2.2 Main Tier Config

```toml
# .version-tiers/main.toml
name = "main"
description = "User-facing binaries (synchronized)"

[rust]
# Main binaries
crates = [
    "bin/00_rbee_keeper",
    "bin/10_queen_rbee",
    "bin/20_rbee_hive",
]

# Shared crates (all get bumped together)
shared_crates = [
    "bin/15_queen_rbee_crates/*",
    "bin/25_rbee_hive_crates/*",
    "bin/96_lifecycle/*",
    "bin/97_contracts/*",
    "bin/98_security_crates/*",
    "bin/99_shared_crates/*",
    "contracts/*",
    "tools/*",
]

[javascript]
packages = [
    "@rbee/queen-rbee-sdk",
    "@rbee/rbee-hive-sdk",
]
```

### 2.3 LLM Worker Tier Config

```toml
# .version-tiers/llm-worker.toml
name = "llm-worker"
description = "LLM inference worker"

[rust]
crates = [
    "bin/30_llm_worker_rbee",
    "bin/32_shared_worker_rbee",
]

[javascript]
packages = [
    "@rbee/llm-worker-sdk",
    "@rbee/llm-worker-ui",
]
```

### 2.4 SD Worker Tier Config

```toml
# .version-tiers/sd-worker.toml
name = "sd-worker"
description = "Stable Diffusion worker"

[rust]
crates = [
    "bin/31_sd_worker_rbee",
    "bin/32_shared_worker_rbee",
]

[javascript]
packages = []
```

### 2.5 Commercial Tier Config

```toml
# .version-tiers/commercial.toml
name = "commercial"
description = "Marketing site"

[rust]
crates = []

[javascript]
packages = [
    "@rbee/commercial",
]
```

---

## 📋 Phase 3: Implement xtask Module

### 3.1 File Structure

```
xtask/
├── src/
│   ├── main.rs              # Entry point (add release command)
│   ├── release/             # NEW MODULE
│   │   ├── mod.rs           # Module exports
│   │   ├── cli.rs           # CLI args and interactive prompts
│   │   ├── tiers.rs         # Tier loading and validation
│   │   ├── bump_rust.rs     # Bump Cargo.toml versions
│   │   ├── bump_js.rs       # Bump package.json versions
│   │   ├── git.rs           # Git operations
│   │   ├── preview.rs       # Dry-run preview
│   │   └── errors.rs        # Custom error types
│   ├── chaos/
│   ├── e2e/
│   └── integration/
└── Cargo.toml
```

### 3.2 Main Entry Point

```rust
// xtask/src/main.rs
mod release;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Existing commands
    Chaos { /* ... */ },
    E2e { /* ... */ },
    Integration { /* ... */ },
    
    // NEW: Release command
    Release {
        /// Tier to release (main, llm-worker, sd-worker, commercial)
        #[arg(long)]
        tier: Option<String>,
        
        /// Bump type (patch, minor, major)
        #[arg(long)]
        r#type: Option<String>,
        
        /// Dry run (preview changes without applying)
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Release { tier, r#type, dry_run } => {
            release::run(tier, r#type, dry_run)?;
        }
        // ... existing commands
    }
    
    Ok(())
}
```

### 3.3 Interactive Prompts

```rust
// xtask/src/release/cli.rs
use inquire::{Select, Confirm};
use colored::*;

pub struct ReleaseConfig {
    pub tier: String,
    pub bump_type: BumpType,
    pub dry_run: bool,
}

pub enum BumpType {
    Patch,
    Minor,
    Major,
}

pub fn prompt_release_config(
    tier_arg: Option<String>,
    type_arg: Option<String>,
    dry_run: bool,
) -> anyhow::Result<ReleaseConfig> {
    println!("{}", "🐝 rbee Release Manager".bright_blue().bold());
    println!("{}", "━".repeat(80).bright_blue());
    println!();
    
    // Prompt for tier if not provided
    let tier = if let Some(t) = tier_arg {
        t
    } else {
        Select::new(
            "Which tier to release?",
            vec![
                "main (rbee-keeper, queen-rbee, rbee-hive)",
                "llm-worker (LLM inference worker)",
                "sd-worker (Stable Diffusion worker)",
                "commercial (Marketing site)",
            ],
        )
        .prompt()?
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
    };
    
    // Prompt for bump type if not provided
    let bump_type = if let Some(t) = type_arg {
        parse_bump_type(&t)?
    } else {
        let choice = Select::new(
            "Version bump type?",
            vec![
                "patch (0.3.0 → 0.3.1) - Bug fixes",
                "minor (0.3.0 → 0.4.0) - New features",
                "major (0.3.0 → 1.0.0) - Breaking changes",
            ],
        )
        .prompt()?;
        
        if choice.starts_with("patch") {
            BumpType::Patch
        } else if choice.starts_with("minor") {
            BumpType::Minor
        } else {
            BumpType::Major
        }
    };
    
    Ok(ReleaseConfig { tier, bump_type, dry_run })
}

fn parse_bump_type(s: &str) -> anyhow::Result<BumpType> {
    match s.to_lowercase().as_str() {
        "patch" => Ok(BumpType::Patch),
        "minor" => Ok(BumpType::Minor),
        "major" => Ok(BumpType::Major),
        _ => anyhow::bail!("Invalid bump type: {}", s),
    }
}
```

### 3.4 Tier Loading

```rust
// xtask/src/release/tiers.rs
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct TierConfig {
    pub name: String,
    pub description: String,
    pub rust: RustConfig,
    pub javascript: JavaScriptConfig,
}

#[derive(Debug, Deserialize)]
pub struct RustConfig {
    pub crates: Vec<String>,
    #[serde(default)]
    pub shared_crates: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct JavaScriptConfig {
    pub packages: Vec<String>,
}

pub fn load_tier_config(tier: &str) -> anyhow::Result<TierConfig> {
    let path = PathBuf::from(".version-tiers").join(format!("{}.toml", tier));
    
    if !path.exists() {
        anyhow::bail!("Tier config not found: {}", path.display());
    }
    
    let content = std::fs::read_to_string(&path)?;
    let config: TierConfig = toml::from_str(&content)?;
    
    Ok(config)
}

pub fn expand_glob_patterns(patterns: &[String]) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    
    for pattern in patterns {
        if pattern.contains('*') {
            // Expand glob pattern
            let base = pattern.split('*').next().unwrap();
            // Use walkdir to find matching directories
            // ...
        } else {
            paths.push(PathBuf::from(pattern));
        }
    }
    
    Ok(paths)
}
```

### 3.5 Bump Rust Crates

```rust
// xtask/src/release/bump_rust.rs
use toml_edit::{Document, value};
use semver::Version;
use std::fs;
use std::path::Path;

pub fn bump_cargo_toml(
    path: &Path,
    bump_type: &BumpType,
    dry_run: bool,
) -> anyhow::Result<(Version, Version)> {
    let cargo_toml_path = path.join("Cargo.toml");
    
    if !cargo_toml_path.exists() {
        anyhow::bail!("Cargo.toml not found: {}", cargo_toml_path.display());
    }
    
    // Read and parse
    let content = fs::read_to_string(&cargo_toml_path)?;
    let mut doc = content.parse::<Document>()?;
    
    // Get current version
    let current_version = doc["package"]["version"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No version found"))?;
    
    let mut version = Version::parse(current_version)?;
    
    // Bump version
    match bump_type {
        BumpType::Patch => version.patch += 1,
        BumpType::Minor => {
            version.minor += 1;
            version.patch = 0;
        }
        BumpType::Major => {
            version.major += 1;
            version.minor = 0;
            version.patch = 0;
        }
    }
    
    let old_version = Version::parse(current_version)?;
    let new_version = version.clone();
    
    // Update document
    doc["package"]["version"] = value(version.to_string());
    
    // Write back (unless dry-run)
    if !dry_run {
        fs::write(&cargo_toml_path, doc.to_string())?;
    }
    
    Ok((old_version, new_version))
}
```

### 3.6 Bump JavaScript Packages

```rust
// xtask/src/release/bump_js.rs
use serde_json::Value;
use semver::Version;
use std::fs;
use std::path::Path;

pub fn bump_package_json(
    package_name: &str,
    bump_type: &BumpType,
    dry_run: bool,
) -> anyhow::Result<(Version, Version)> {
    // Find package.json for this package
    let package_json_path = find_package_json(package_name)?;
    
    // Read and parse
    let content = fs::read_to_string(&package_json_path)?;
    let mut json: Value = serde_json::from_str(&content)?;
    
    // Get current version
    let current_version = json["version"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No version found"))?;
    
    let mut version = Version::parse(current_version)?;
    
    // Bump version
    match bump_type {
        BumpType::Patch => version.patch += 1,
        BumpType::Minor => {
            version.minor += 1;
            version.patch = 0;
        }
        BumpType::Major => {
            version.major += 1;
            version.minor = 0;
            version.patch = 0;
        }
    }
    
    let old_version = Version::parse(current_version)?;
    let new_version = version.clone();
    
    // Update JSON
    json["version"] = Value::String(version.to_string());
    
    // Write back (unless dry-run)
    if !dry_run {
        let pretty = serde_json::to_string_pretty(&json)?;
        fs::write(&package_json_path, pretty)?;
    }
    
    Ok((old_version, new_version))
}

fn find_package_json(package_name: &str) -> anyhow::Result<PathBuf> {
    // Search pnpm workspace for package
    // Use walkdir to find package.json with matching name
    // ...
}
```

### 3.7 Preview Changes

```rust
// xtask/src/release/preview.rs
use colored::*;

pub fn preview_changes(
    tier: &str,
    bump_type: &BumpType,
    rust_changes: &[(PathBuf, Version, Version)],
    js_changes: &[(String, Version, Version)],
) {
    println!();
    println!("{}", "📋 Preview:".bright_blue().bold());
    println!("{}", "━".repeat(80).bright_blue());
    println!();
    
    println!("Tier: {}", tier.bright_green());
    println!("Bump: {:?}", bump_type);
    println!();
    
    if !rust_changes.is_empty() {
        println!("{}", format!("Rust crates ({}):", rust_changes.len()).bright_yellow());
        for (path, old, new) in rust_changes {
            println!("  {} {} → {}", 
                "✓".bright_green(),
                path.display().to_string().dimmed(),
                format!("{} → {}", old, new).bright_cyan()
            );
        }
        println!();
    }
    
    if !js_changes.is_empty() {
        println!("{}", format!("JavaScript packages ({}):", js_changes.len()).bright_yellow());
        for (name, old, new) in js_changes {
            println!("  {} {} → {}",
                "✓".bright_green(),
                name.dimmed(),
                format!("{} → {}", old, new).bright_cyan()
            );
        }
        println!();
    }
}
```

---

## 📋 Phase 4: macOS CI Setup

### 4.1 Install Dependencies on Mac

```bash
# On your Mac (mac.home.arpa)

# 1. Install Rust (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install pnpm (if not already)
curl -fsSL https://get.pnpm.io/install.sh | sh -

# 3. Clone repo
cd ~/Projects
git clone git@github.com:rbee-keeper/rbee.git
cd rbee

# 4. Build xtask
cargo build --package xtask --release

# 5. Test it
cargo xtask release --dry-run
```

### 4.2 Create macOS CI Script

```bash
# scripts/mac-ci.sh
#!/bin/bash
# macOS CI script - runs on mac.home.arpa
# Created by: TEAM-451

set -e

echo "🍎 macOS CI - Starting build"
echo ""

# 1. Pull latest changes
git fetch origin
git checkout development
git pull origin development

# 2. Install dependencies
pnpm install

# 3. Build Rust
cargo build --release --bin rbee-keeper
cargo build --release --bin queen-rbee
cargo build --release --bin rbee-hive

# 4. Build macOS workers
cargo build --release --bin llm-worker-rbee-metal --features metal
cargo build --release --bin sd-worker-metal --features metal

# 5. Run tests
cargo test --all

# 6. Package binaries
mkdir -p dist/macos
cp target/release/rbee-keeper dist/macos/
cp target/release/queen-rbee dist/macos/
cp target/release/rbee-hive dist/macos/
cp target/release/llm-worker-rbee-metal dist/macos/
cp target/release/sd-worker-metal dist/macos/

# 7. Create tarball
cd dist/macos
tar -czf ../rbee-macos-$(uname -m).tar.gz *

echo ""
echo "✅ macOS CI - Build complete"
echo "📦 Artifact: dist/rbee-macos-$(uname -m).tar.gz"
```

### 4.3 Setup GitHub Actions Self-Hosted Runner

```bash
# On your Mac

# 1. Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# 2. Download runner (ARM64 for Apple Silicon)
curl -o actions-runner-osx-arm64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-osx-arm64-2.311.0.tar.gz

# 3. Extract
tar xzf ./actions-runner-osx-arm64.tar.gz

# 4. Get registration token
# On your main machine:
gh api repos/rbee-keeper/rbee/actions/runners/registration-token | jq -r .token

# 5. Configure runner
./config.sh --url https://github.com/rbee-keeper/rbee --token <TOKEN>
# Labels: macos,self-hosted,arm64

# 6. Install as service
./svc.sh install

# 7. Start service
./svc.sh start

# 8. Verify
./svc.sh status
```

---

## 📋 Phase 5: Workflow Integration

### 5.1 Pre-Merge Hook (Local)

```bash
# .git/hooks/pre-push (optional)
#!/bin/bash

BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ "$BRANCH" = "development" ]; then
    echo "🔍 Running release checks..."
    cargo xtask release --dry-run
fi
```

### 5.2 GitHub Actions Workflow

```yaml
# .github/workflows/release-check.yml
name: Release Check

on:
  pull_request:
    branches:
      - production

jobs:
  check-version:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        
      - name: Build xtask
        run: cargo build --package xtask
        
      - name: Check version consistency
        run: cargo xtask release --dry-run --tier main --type minor
```

---

## 📋 Phase 6: Usage Instructions

### 6.1 Developer Workflow

```bash
# 1. Make changes on development branch
git checkout development
# ... make changes ...

# 2. Run interactive release
cargo xtask release

# Prompts:
# ? Which tier to release? main
# ? Version bump type? minor
# 
# Preview:
# Tier: main
# Bump: minor (0.3.0 → 0.4.0)
# 
# Rust crates (12):
#   ✓ bin/00_rbee_keeper: 0.3.0 → 0.4.0
#   ...
# 
# ? Proceed? Yes

# 3. Commit and push
git add .
git commit -m "chore: release main v0.4.0"
git push origin development

# 4. Create PR to production
gh pr create --base production --head development --title "Release rbee v0.4.0"

# 5. Merge triggers CI on Mac + Linux
# 6. Automatic release created
```

### 6.2 macOS CI Workflow

```bash
# On your Mac (triggered by GitHub Actions)

# 1. Pull latest from production
git checkout production
git pull origin production

# 2. Run macOS build
./scripts/mac-ci.sh

# 3. Upload artifacts to GitHub Release
gh release upload v0.4.0 dist/rbee-macos-*.tar.gz
```

---

## 📋 Phase 7: Testing Plan

### 7.1 Unit Tests

```rust
// xtask/src/release/tests.rs
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bump_patch() {
        let version = Version::parse("0.3.0").unwrap();
        let bumped = bump_version(&version, BumpType::Patch);
        assert_eq!(bumped.to_string(), "0.3.1");
    }
    
    #[test]
    fn test_bump_minor() {
        let version = Version::parse("0.3.5").unwrap();
        let bumped = bump_version(&version, BumpType::Minor);
        assert_eq!(bumped.to_string(), "0.4.0");
    }
    
    #[test]
    fn test_load_tier_config() {
        let config = load_tier_config("main").unwrap();
        assert_eq!(config.name, "main");
        assert!(!config.rust.crates.is_empty());
    }
}
```

### 7.2 Integration Tests

```bash
# Test on macOS
cargo xtask release --tier main --type patch --dry-run

# Verify output
# Should show all crates that would be bumped
```

---

## 📋 Implementation Checklist

### Phase 1: Setup ✅
- [ ] Update xtask/Cargo.toml with dependencies
- [ ] Test build on macOS
- [ ] Test build on Linux

### Phase 2: Tier Configs ✅
- [ ] Create .version-tiers/main.toml
- [ ] Create .version-tiers/llm-worker.toml
- [ ] Create .version-tiers/sd-worker.toml
- [ ] Create .version-tiers/commercial.toml

### Phase 3: Implementation ✅
- [ ] Implement cli.rs (interactive prompts)
- [ ] Implement tiers.rs (config loading)
- [ ] Implement bump_rust.rs (Cargo.toml updates)
- [ ] Implement bump_js.rs (package.json updates)
- [ ] Implement preview.rs (dry-run display)
- [ ] Implement git.rs (commit, tag)
- [ ] Update main.rs (add release command)

### Phase 4: macOS CI ✅
- [ ] Install Rust on Mac
- [ ] Install pnpm on Mac
- [ ] Setup GitHub Actions runner on Mac
- [ ] Create mac-ci.sh script
- [ ] Test build on Mac

### Phase 5: Workflows ✅
- [ ] Create release-check.yml workflow
- [ ] Update production-release.yml to use xtask
- [ ] Test PR workflow

### Phase 6: Documentation ✅
- [ ] Update RELEASE_WORKFLOW.md
- [ ] Update PUBLISHING_GUIDE.md
- [ ] Create xtask release README

### Phase 7: Testing ✅
- [ ] Write unit tests
- [ ] Test dry-run mode
- [ ] Test actual release (patch)
- [ ] Test on macOS
- [ ] Test on Linux

---

## 🎯 Success Criteria

1. ✅ `cargo xtask release` works on macOS
2. ✅ `cargo xtask release` works on Linux
3. ✅ Interactive prompts guide user
4. ✅ Dry-run shows preview without changes
5. ✅ Bumps all crates/packages in tier
6. ✅ macOS CI builds and uploads artifacts
7. ✅ GitHub Actions validates versions

---

## 📚 Next Steps

1. **Approve this plan**
2. **I'll implement Phase 1-3** (xtask code)
3. **You setup Phase 4** (macOS runner)
4. **We test together**
5. **Deploy!**

Ready to start implementation?
