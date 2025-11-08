# macOS Setup Guide - TEAM-451

**Machine:** mac (macOS ARM64)  
**Purpose:** Build macOS binaries for releases  
**Status:** In Progress

---

## 🎯 What We're Setting Up

1. ✅ Rust toolchain (already installed)
2. 📦 pnpm (package manager for frontend)
3. 🔧 Build dependencies
4. 🤖 GitHub Actions self-hosted runner (optional)

---

## 🚀 Quick Setup (Automated)

### Option 1: Run via SSH from blep

```bash
# From blep, copy and run the setup script on mac
scp scripts/setup-mac.sh mac:~/
ssh mac "cd ~/Projects/rbee && ./setup-mac.sh"
```

### Option 2: Run directly on mac

```bash
# SSH into mac
ssh mac

# Navigate to rbee
cd ~/Projects/rbee

# Run setup script
./scripts/setup-mac.sh
```

---

## 📋 Manual Setup (Step by Step)

### Step 1: Install pnpm

```bash
ssh mac << 'EOF'
# Install pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -

# Add to PATH
export PNPM_HOME="$HOME/Library/pnpm"
export PATH="$PNPM_HOME:$PATH"

# Add to shell profile
echo 'export PNPM_HOME="$HOME/Library/pnpm"' >> ~/.zshrc
echo 'export PATH="$PNPM_HOME:$PATH"' >> ~/.zshrc

# Verify
pnpm --version
EOF
```

### Step 2: Install Node.js (via pnpm)

```bash
ssh mac << 'EOF'
# Install latest LTS Node.js
pnpm env use --global lts

# Verify
node --version
npm --version
EOF
```

### Step 3: Test Rust Build

```bash
ssh mac << 'EOF'
cd ~/Projects/rbee

# Test build (this will take a while first time)
cargo build --package rbee-keeper --release

# Check binary
ls -lh target/release/rbee-keeper
EOF
```

### Step 4: Test Frontend Build

```bash
ssh mac << 'EOF'
cd ~/Projects/rbee

# Install dependencies
pnpm install

# Build frontend apps
pnpm --filter @rbee/commercial build
pnpm --filter @rbee/marketplace build
pnpm --filter @rbee/user-docs build

# Verify
ls -lh frontend/apps/*/dist
EOF
```

---

## 🤖 GitHub Actions Self-Hosted Runner (Optional)

### Why?

- Build macOS binaries automatically on release
- No need to manually build on mac
- Integrated with GitHub workflow

### Setup

#### 1. Create Runner on GitHub

1. Go to: https://github.com/rbee-keeper/rbee/settings/actions/runners/new
2. Select: **macOS** and **ARM64**
3. Copy the download and configure commands

#### 2. Install Runner on mac

```bash
ssh mac << 'EOF'
# Create actions-runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download (replace with actual URL from GitHub)
curl -o actions-runner-osx-arm64-2.321.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-osx-arm64-2.321.0.tar.gz

# Extract
tar xzf ./actions-runner-osx-arm64-*.tar.gz

# Configure (replace TOKEN with actual token from GitHub)
./config.sh --url https://github.com/rbee-keeper/rbee --token YOUR_TOKEN_HERE

# Install as service
./svc.sh install

# Start service
./svc.sh start

# Check status
./svc.sh status
EOF
```

#### 3. Label the Runner

On GitHub:
1. Go to: https://github.com/rbee-keeper/rbee/settings/actions/runners
2. Click on your runner
3. Add label: `macos`

#### 4. Test the Runner

Create a test workflow:

```yaml
# .github/workflows/test-mac-runner.yml
name: Test macOS Runner

on:
  workflow_dispatch:

jobs:
  test:
    runs-on: [self-hosted, macos]
    steps:
      - uses: actions/checkout@v4
      - name: Test build
        run: |
          rustc --version
          cargo --version
          pnpm --version
          cargo build --package rbee-keeper --release
```

---

## 🔧 Build Commands Reference

### Build Rust Binaries

```bash
# Build all main binaries
ssh mac "cd ~/Projects/rbee && cargo build --release"

# Build specific binary
ssh mac "cd ~/Projects/rbee && cargo build --package rbee-keeper --release"

# Build workers
ssh mac "cd ~/Projects/rbee && cargo build --package llm-worker-rbee --release"
```

### Build Frontend Apps

```bash
# Install dependencies
ssh mac "cd ~/Projects/rbee && pnpm install"

# Build all apps
ssh mac "cd ~/Projects/rbee && pnpm build"

# Build specific app
ssh mac "cd ~/Projects/rbee && pnpm --filter @rbee/commercial build"
```

---

## 📦 Release Workflow

### Manual Release (Without GitHub Actions)

```bash
# 1. On blep: Bump version
cargo xtask release --tier main --type minor

# 2. Commit and push
git add .
git commit -m "chore: release main v0.2.0"
git push origin development

# 3. On mac: Build binaries
ssh mac << 'EOF'
cd ~/Projects/rbee
git pull origin development
cargo build --release

# Package binaries
cd target/release
tar -czf rbee-macos-arm64.tar.gz rbee-keeper queen-rbee rbee-hive
EOF

# 4. Download binaries
scp mac:~/Projects/rbee/target/release/rbee-macos-arm64.tar.gz .

# 5. Create GitHub release and upload
gh release create v0.2.0 rbee-macos-arm64.tar.gz --title "Release v0.2.0"
```

### Automated Release (With GitHub Actions)

```bash
# 1. Bump version
cargo xtask release --tier main --type minor

# 2. Commit and push
git add .
git commit -m "chore: release main v0.2.0"
git push origin development

# 3. Create PR to production
gh pr create --base production --head development --title "Release v0.2.0"

# 4. Merge PR → GitHub Actions builds on mac automatically
# 5. Binaries uploaded to GitHub Release automatically
```

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# Check Rust
ssh mac "rustc --version && cargo --version"

# Check pnpm
ssh mac "pnpm --version && node --version"

# Check build
ssh mac "cd ~/Projects/rbee && cargo build --package rbee-keeper"

# Check frontend
ssh mac "cd ~/Projects/rbee && pnpm install && pnpm --filter @rbee/commercial build"
```

Expected output:
- ✅ Rust: 1.90.0 or newer
- ✅ pnpm: 9.x or newer
- ✅ Node.js: 20.x or newer (LTS)
- ✅ Build completes without errors

---

## 🐛 Troubleshooting

### pnpm not found after installation

```bash
# Restart shell
ssh mac "source ~/.zshrc && pnpm --version"

# Or manually add to PATH
ssh mac "export PNPM_HOME=\"\$HOME/Library/pnpm\" && export PATH=\"\$PNPM_HOME:\$PATH\" && pnpm --version"
```

### Rust build fails

```bash
# Update Rust
ssh mac "rustup update"

# Clean and rebuild
ssh mac "cd ~/Projects/rbee && cargo clean && cargo build --release"
```

### GitHub Actions runner not starting

```bash
# Check status
ssh mac "cd ~/actions-runner && ./svc.sh status"

# View logs
ssh mac "cd ~/actions-runner && tail -f _diag/Runner_*.log"

# Restart
ssh mac "cd ~/actions-runner && ./svc.sh stop && ./svc.sh start"
```

---

## 📊 Current Status

**Installed:**
- ✅ Rust: 1.90.0
- ❌ pnpm: Not installed
- ❌ Node.js: Not installed
- ❌ GitHub Actions runner: Not installed

**Next Steps:**
1. Run `./scripts/setup-mac.sh` on mac
2. Verify build works
3. (Optional) Setup GitHub Actions runner

---

## 🔗 Related Documentation

- **Release Guide:** `.docs/RELEASE_GUIDE.md`
- **Multi-Platform Build Plan:** `.docs/MULTI_PLATFORM_BUILD_PLAN.md`
- **Setup Complete:** `SETUP_COMPLETE.md`

---

**Status:** Ready to execute  
**Time:** ~15 minutes (excluding GitHub Actions runner)
