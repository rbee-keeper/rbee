# rbee Release Guide

**Created by:** TEAM-451  
**Rule Zero Applied:** ONE document for releases

---

## 🎯 Quick Start

```bash
# 1. Bump version (interactive)
cargo xtask release

# 2. Create PR
gh pr create --base production --head development --title "Release v0.4.0"

# 3. Merge → Automatic builds on mac, blep, workstation

# 4. Users install
curl -fsSL https://install.rbee.dev | sh  # macOS/Linux
paru -S rbee                              # Arch Linux
```

---

## 📋 Version Strategy

### Tiers (Hybrid Versioning)

**Tier 1: Main** (Synchronized - all share same version)
- rbee-keeper, queen-rbee, rbee-hive → v0.4.0
- All shared crates → v0.4.0
- Main SDKs → v0.4.0

**Tier 2: Workers** (Independent)
- llm-worker-rbee → v1.4.0
- sd-worker-rbee → v0.6.0

**Tier 3: Frontend** (Independent)
- @rbee/commercial → v2.1.0
- @rbee/marketplace → v1.0.0

### Why Hybrid?

✅ **Simple for users:** "Install rbee v0.4.0"  
✅ **Flexible for workers:** Independent release cycles  
✅ **Maintainable:** Clear tier boundaries

---

## 🛠️ Implementation: cargo xtask release

### Tool: Rust xtask (Interactive CLI)

**Why xtask?**
- ✅ Already in your stack
- ✅ Type-safe (Rust)
- ✅ Interactive prompts
- ✅ Cross-platform (macOS, Linux)
- ✅ No new dependencies

### Usage

```bash
# Interactive mode
cargo xtask release

# Prompts:
# ? Which tier? main
# ? Bump type? minor
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

# Or non-interactive
cargo xtask release --tier main --type minor --dry-run
```

### Implementation Status

**Phase 1:** Define tier configs (`.version-tiers/*.toml`)  
**Phase 2:** Implement xtask module  
**Phase 3:** Setup build machines  

See: `.docs/RELEASE_XTASK_IMPLEMENTATION_PLAN.md`

---

## 📦 Build Targets

### 1. macOS (mac.lan - ARM64)
- **Builder:** GitHub Actions self-hosted runner
- **Output:** `rbee-macos-arm64.tar.gz`
- **Includes:** Metal-accelerated workers

### 2. AUR (blep - Arch Linux)
- **Builder:** Manual (you run script)
- **Output:** PKGBUILD → AUR
- **Install:** `paru -S rbee`

### 3. install.rbee.dev (workstation - Ubuntu)
- **Builder:** GitHub Actions
- **Output:** Universal installer
- **Install:** `curl -fsSL https://install.rbee.dev | sh`

See: `.docs/MULTI_PLATFORM_BUILD_PLAN.md`

---

## 🔒 Branch Protection

### Branches

**development** (default)
- Free pushing
- No restrictions
- All development happens here

**production** (protected)
- Requires PR from development
- Requires 1 approval
- Requires CI to pass
- Triggers automatic releases
- **Cannot push directly**

### Setup

```bash
# Configure protection
./scripts/configure-branch-protection.sh
```

See: `.docs/REPOSITORY_PROTECTION.md`

---

## 🚀 Complete Workflow

### Step 1: Develop on development

```bash
git checkout development
# Make changes
git add .
git commit -m "feat: add feature"
git push  # Free pushing!
```

### Step 2: Bump Version

```bash
cargo xtask release
# Interactive prompts guide you
```

### Step 3: Create Release PR

```bash
gh pr create \
    --base production \
    --head development \
    --title "Release rbee v0.4.0"
```

### Step 4: Merge → Automatic Publishing

When merged to production:
1. ✅ GitHub Actions builds Linux binaries
2. ✅ mac builds macOS binaries
3. ✅ GitHub Release created
4. ✅ Binaries uploaded
5. ⚠️ Manual: Update AUR (run `./scripts/update-aur.sh`)

### Step 5: Verify

```bash
# macOS
curl -fsSL https://install.rbee.dev | sh

# Arch Linux
paru -S rbee

# Check version
rbee-keeper --version
```

---

## 📚 Detailed Documentation

- **Implementation Plan:** `.docs/RELEASE_XTASK_IMPLEMENTATION_PLAN.md`
- **Build Targets:** `.docs/MULTI_PLATFORM_BUILD_PLAN.md`
- **Version Strategy:** `.docs/VERSION_MANAGEMENT_PLAN.md`
- **Repository Security:** `.docs/REPOSITORY_PROTECTION.md`
- **Publishing Details:** `.docs/PUBLISHING_GUIDE.md`

---

## ✅ Setup Checklist

### One-Time Setup

**On mac:**
- [ ] Install pnpm: `curl -fsSL https://get.pnpm.io/install.sh | sh -`
- [ ] Setup GitHub Actions runner
- [ ] Test build: `./scripts/build-macos.sh`

**On workstation:**
- [ ] Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- [ ] Install pnpm: `curl -fsSL https://get.pnpm.io/install.sh | sh -`

**On blep:**
- [ ] Setup AUR SSH keys
- [ ] Test PKGBUILD: `makepkg -sf`

**Implement xtask:**
- [ ] Add dependencies to xtask/Cargo.toml
- [ ] Create `.version-tiers/*.toml` configs
- [ ] Implement xtask release module
- [ ] Test on all platforms

### Per-Release Checklist

- [ ] All tests pass: `cargo test --all`
- [ ] Clippy passes: `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Frontend builds: `pnpm run build`
- [ ] Version bumped: `cargo xtask release`
- [ ] PR created to production
- [ ] CI passes
- [ ] PR approved
- [ ] Merged to production
- [ ] Builds complete (mac, Linux)
- [ ] AUR updated: `./scripts/update-aur.sh`
- [ ] Install script works: `curl -fsSL https://install.rbee.dev | sh`

---

**Status:** Planning Complete → Ready for Implementation  
**Next:** Implement `cargo xtask release` (Phase 1-3)
