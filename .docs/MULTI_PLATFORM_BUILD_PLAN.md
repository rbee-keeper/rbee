# Multi-Platform Build Plan

**Created by:** TEAM-451  
**Build Targets:** macOS (mac), AUR (blep), install.rbee.dev (workstation)

## 🖥️ Build Machines

### 1. **mac** (macOS ARM64)
- **OS:** Darwin 25.1.0 (macOS Sequoia)
- **Arch:** arm64 (Apple Silicon)
- **Rust:** 1.90.0 ✅
- **pnpm:** Not installed ❌
- **Access:** `ssh mac`
- **Builds:** macOS binaries (.dmg, .tar.gz)

### 2. **blep** (Arch Linux x86_64)
- **OS:** Linux 6.16.12-hardened (Arch)
- **Arch:** x86_64
- **Rust:** 1.90.0 ✅
- **pnpm:** 10.20.0 ✅
- **Access:** Local machine
- **Builds:** AUR packages (PKGBUILD)

### 3. **workstation** (Ubuntu x86_64)
- **OS:** Ubuntu 24.04 (6.8.0-87-generic)
- **Arch:** x86_64
- **Rust:** Not installed ❌
- **pnpm:** Not installed ❌
- **Access:** `ssh workstation`
- **Builds:** install.rbee.dev script (universal installer)

---

## 📦 Build Matrix

| Platform | Machine | Builder | Output | Distribution |
|----------|---------|---------|--------|--------------|
| **macOS ARM64** | mac | GitHub Actions (self-hosted) | `rbee-macos-arm64.tar.gz` | GitHub Releases |
| **macOS x86_64** | GitHub Actions (cloud) | GitHub Actions | `rbee-macos-x86_64.tar.gz` | GitHub Releases |
| **Linux x86_64** | GitHub Actions (cloud) | GitHub Actions | `rbee-linux-x86_64.tar.gz` | GitHub Releases + install.rbee.dev |
| **AUR** | blep | Manual/GitHub Actions | PKGBUILD | AUR repository |

---

## 🎯 Build Target 1: macOS Build (mac)

### Setup on mac

```bash
# SSH into mac
ssh mac

# Install pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -
source ~/.zshrc  # or ~/.bashrc

# Clone repo
cd ~/Projects
git clone git@github.com:rbee-keeper/rbee.git
cd rbee

# Install dependencies
pnpm install

# Build xtask
cargo build --package xtask --release

# Test build
cargo build --release --bin rbee-keeper
cargo build --release --bin queen-rbee
cargo build --release --bin rbee-hive
cargo build --release --features metal --bin llm-worker-rbee-metal
```

### macOS Build Script

```bash
# scripts/build-macos.sh
#!/bin/bash
# Build macOS binaries on mac.lan
# Created by: TEAM-451

set -e

ARCH=$(uname -m)  # arm64 or x86_64
VERSION=${VERSION:-$(grep '^version = ' Cargo.toml | head -1 | cut -d'"' -f2)}

echo "🍎 Building rbee for macOS ($ARCH)"
echo "Version: $VERSION"
echo ""

# Build main binaries
echo "📦 Building main binaries..."
cargo build --release --bin rbee-keeper
cargo build --release --bin queen-rbee
cargo build --release --bin rbee-hive

# Build Metal workers (macOS only)
echo "🎨 Building Metal workers..."
cargo build --release --features metal --bin llm-worker-rbee-metal
cargo build --release --features metal --bin sd-worker-metal

# Create distribution directory
DIST_DIR="dist/macos-$ARCH"
mkdir -p "$DIST_DIR"

# Copy binaries
echo "📋 Copying binaries..."
cp target/release/rbee-keeper "$DIST_DIR/"
cp target/release/queen-rbee "$DIST_DIR/"
cp target/release/rbee-hive "$DIST_DIR/"
cp target/release/llm-worker-rbee-metal "$DIST_DIR/"
cp target/release/sd-worker-metal "$DIST_DIR/"

# Create install script
cat > "$DIST_DIR/install.sh" << 'EOF'
#!/bin/bash
# rbee installer for macOS
set -e

INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

echo "Installing rbee to $INSTALL_DIR..."

sudo cp rbee-keeper "$INSTALL_DIR/"
sudo cp queen-rbee "$INSTALL_DIR/"
sudo cp rbee-hive "$INSTALL_DIR/"
sudo cp llm-worker-rbee-metal "$INSTALL_DIR/"
sudo cp sd-worker-metal "$INSTALL_DIR/"

sudo chmod +x "$INSTALL_DIR/rbee-keeper"
sudo chmod +x "$INSTALL_DIR/queen-rbee"
sudo chmod +x "$INSTALL_DIR/rbee-hive"
sudo chmod +x "$INSTALL_DIR/llm-worker-rbee-metal"
sudo chmod +x "$INSTALL_DIR/sd-worker-metal"

echo "✅ rbee installed successfully!"
echo "Run: rbee-keeper --help"
EOF
chmod +x "$DIST_DIR/install.sh"

# Create tarball
echo "📦 Creating tarball..."
cd dist
tar -czf "rbee-macos-$ARCH-v$VERSION.tar.gz" "macos-$ARCH"
cd ..

echo ""
echo "✅ Build complete!"
echo "📦 Artifact: dist/rbee-macos-$ARCH-v$VERSION.tar.gz"
```

### GitHub Actions Self-Hosted Runner on mac

```bash
# On mac
ssh mac

# Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download runner (ARM64)
curl -o actions-runner-osx-arm64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-osx-arm64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-osx-arm64.tar.gz

# Get registration token (from your main machine)
# gh api repos/rbee-keeper/rbee/actions/runners/registration-token | jq -r .token

# Configure
./config.sh --url https://github.com/rbee-keeper/rbee --token <TOKEN>
# Labels: macos,self-hosted,arm64

# Install as service
./svc.sh install

# Start service
./svc.sh start

# Verify
./svc.sh status
```

---

## 🎯 Build Target 2: AUR Build (blep)

### AUR Package Structure

```
aur-rbee/
├── PKGBUILD
├── .SRCINFO
└── rbee.install (optional)
```

### PKGBUILD Template

```bash
# PKGBUILD
# Maintainer: Your Name <your@email.com>

pkgname=rbee
pkgver=0.4.0
pkgrel=1
pkgdesc="Distributed LLM inference system"
arch=('x86_64')
url="https://github.com/rbee-keeper/rbee"
license=('GPL3')
depends=('gcc-libs')
makedepends=('rust' 'cargo' 'pnpm')
source=("$pkgname-$pkgver.tar.gz::https://github.com/rbee-keeper/rbee/archive/v$pkgver.tar.gz")
sha256sums=('SKIP')  # Will be generated

build() {
    cd "$pkgname-$pkgver"
    
    # Build Rust binaries
    cargo build --release --bin rbee-keeper
    cargo build --release --bin queen-rbee
    cargo build --release --bin rbee-hive
    cargo build --release --bin llm-worker-rbee-cpu
    cargo build --release --bin llm-worker-rbee-cuda --features cuda
}

package() {
    cd "$pkgname-$pkgver"
    
    # Install binaries
    install -Dm755 target/release/rbee-keeper "$pkgdir/usr/bin/rbee-keeper"
    install -Dm755 target/release/queen-rbee "$pkgdir/usr/bin/queen-rbee"
    install -Dm755 target/release/rbee-hive "$pkgdir/usr/bin/rbee-hive"
    install -Dm755 target/release/llm-worker-rbee-cpu "$pkgdir/usr/bin/llm-worker-rbee-cpu"
    install -Dm755 target/release/llm-worker-rbee-cuda "$pkgdir/usr/bin/llm-worker-rbee-cuda"
    
    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
```

### AUR Update Script

```bash
# scripts/update-aur.sh
#!/bin/bash
# Update AUR package
# Created by: TEAM-451
# Run on: blep (Arch Linux)

set -e

VERSION=${1:-$(grep '^version = ' Cargo.toml | head -1 | cut -d'"' -f2)}

echo "📦 Updating AUR package to v$VERSION"
echo ""

# Clone AUR repo (if not exists)
if [ ! -d "aur-rbee" ]; then
    git clone ssh://aur@aur.archlinux.org/rbee.git aur-rbee
fi

cd aur-rbee

# Update PKGBUILD
sed -i "s/^pkgver=.*/pkgver=$VERSION/" PKGBUILD
sed -i "s/^pkgrel=.*/pkgrel=1/" PKGBUILD

# Update checksums
updpkgsums

# Generate .SRCINFO
makepkg --printsrcinfo > .SRCINFO

# Test build
echo "🔨 Testing build..."
makepkg -sf --noconfirm

# Commit and push
git add PKGBUILD .SRCINFO
git commit -m "Update to version $VERSION"
git push

echo ""
echo "✅ AUR package updated to v$VERSION"
echo "🔗 https://aur.archlinux.org/packages/rbee"
```

### Setup AUR SSH Keys (on blep)

```bash
# On blep (already done, but for reference)

# Generate SSH key for AUR
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/aur

# Add to ssh config
cat >> ~/.ssh/config << EOF
Host aur.archlinux.org
  IdentityFile ~/.ssh/aur
  User aur
EOF

# Add public key to AUR account
cat ~/.ssh/aur.pub
# Go to https://aur.archlinux.org/account/ and add the key

# Test
ssh -T aur@aur.archlinux.org
```

---

## 🎯 Build Target 3: install.rbee.dev (workstation)

### Setup on workstation

```bash
# SSH into workstation
ssh workstation

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.bashrc

# Install pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -
source ~/.bashrc

# Clone repo
cd ~/Projects
git clone git@github.com:rbee-keeper/rbee.git
cd rbee

# Install dependencies
pnpm install

# Test build
cargo build --release --bin rbee-keeper
```

### Universal Install Script

```bash
# install.sh (already created, but enhanced)
#!/bin/bash
# rbee Universal Installer
# Usage: curl -fsSL https://install.rbee.dev | sh

set -e

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)
        PLATFORM="linux"
        ;;
    Darwin*)
        PLATFORM="macos"
        ;;
    *)
        echo "❌ Unsupported OS: $OS"
        exit 1
        ;;
esac

# Normalize architecture
case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        echo "❌ Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

VERSION="${VERSION:-latest}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
GITHUB_REPO="rbee-keeper/rbee"

echo "🐝 rbee Installer"
echo "Platform: $PLATFORM-$ARCH"
echo "Version: $VERSION"
echo ""

# Get latest version if not specified
if [ "$VERSION" = "latest" ]; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/$GITHUB_REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
    VERSION="${VERSION#v}"
fi

# Download URL
TARBALL="rbee-$PLATFORM-$ARCH-v$VERSION.tar.gz"
DOWNLOAD_URL="https://github.com/$GITHUB_REPO/releases/download/v$VERSION/$TARBALL"

echo "📥 Downloading from:"
echo "  $DOWNLOAD_URL"
echo ""

# Create temp directory
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Download and extract
curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/$TARBALL"
tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"

# Install binaries
mkdir -p "$INSTALL_DIR"

BINARIES=("rbee-keeper" "queen-rbee" "rbee-hive")
for binary in "${BINARIES[@]}"; do
    if [ -f "$TMP_DIR/$PLATFORM-$ARCH/$binary" ]; then
        cp "$TMP_DIR/$PLATFORM-$ARCH/$binary" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/$binary"
        echo "  ✓ $binary"
    fi
done

echo ""
echo "✅ rbee installed to $INSTALL_DIR"
echo ""
echo "Add to PATH:"
echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
echo ""
echo "Get started:"
echo "  rbee-keeper --help"
```

### Hosting install.rbee.dev

**Option 1: GitHub Pages**
```bash
# Create gh-pages branch
git checkout --orphan gh-pages
git rm -rf .
echo "curl -fsSL https://raw.githubusercontent.com/rbee-keeper/rbee/production/install.sh | sh" > index.html
git add index.html
git commit -m "Add install script redirect"
git push origin gh-pages

# Setup custom domain in GitHub repo settings
# CNAME: install.rbee.dev → rbee-keeper.github.io
```

**Option 2: Cloudflare Pages**
```bash
# Deploy install.sh to Cloudflare Pages
# Point install.rbee.dev to Cloudflare
```

**Option 3: Simple redirect**
```bash
# Just use raw GitHub URL
curl -fsSL https://raw.githubusercontent.com/rbee-keeper/rbee/production/install.sh | sh
```

---

## 🔄 Complete Release Workflow

### Step 1: Version Bump (on blep)

```bash
# On blep (your main machine)
cargo xtask release

# Interactive prompts:
# ? Which tier? main
# ? Bump type? minor
# 
# ✅ Version bumped to 0.4.0

# Commit and push
git add .
git commit -m "chore: release main v0.4.0"
git push origin development
```

### Step 2: Create Release PR

```bash
# Create PR to production
gh pr create \
    --base production \
    --head development \
    --title "Release rbee v0.4.0" \
    --body "Release notes..."
```

### Step 3: Merge Triggers Builds

When PR is merged to `production`:

1. **GitHub Actions (Linux x86_64)**
   - Builds Linux binaries
   - Creates GitHub Release
   - Uploads `rbee-linux-x86_64-v0.4.0.tar.gz`

2. **GitHub Actions (macOS ARM64 - self-hosted on mac)**
   - Builds macOS ARM64 binaries
   - Uploads `rbee-macos-arm64-v0.4.0.tar.gz`

3. **GitHub Actions (macOS x86_64 - cloud)**
   - Builds macOS x86_64 binaries
   - Uploads `rbee-macos-x86_64-v0.4.0.tar.gz`

4. **Manual (AUR - on blep)**
   - Run `./scripts/update-aur.sh 0.4.0`
   - Updates PKGBUILD
   - Pushes to AUR

### Step 4: Verify Installations

```bash
# macOS (ARM64)
curl -fsSL https://install.rbee.dev | sh

# macOS (x86_64)
curl -fsSL https://install.rbee.dev | sh

# Linux
curl -fsSL https://install.rbee.dev | sh

# Arch Linux
paru -S rbee
# or
yay -S rbee
```

---

## 📋 Setup Checklist

### mac (macOS ARM64)
- [ ] Install pnpm: `curl -fsSL https://get.pnpm.io/install.sh | sh -`
- [ ] Clone repo: `git clone git@github.com:rbee-keeper/rbee.git`
- [ ] Install deps: `pnpm install`
- [ ] Setup GitHub Actions runner
- [ ] Test build: `./scripts/build-macos.sh`

### blep (Arch Linux - AUR)
- [x] Rust installed ✅
- [x] pnpm installed ✅
- [ ] Setup AUR SSH keys
- [ ] Clone AUR repo: `git clone ssh://aur@aur.archlinux.org/rbee.git`
- [ ] Test PKGBUILD: `makepkg -sf`

### workstation (Ubuntu - install.rbee.dev)
- [ ] Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- [ ] Install pnpm: `curl -fsSL https://get.pnpm.io/install.sh | sh -`
- [ ] Clone repo: `git clone git@github.com:rbee-keeper/rbee.git`
- [ ] Test build: `cargo build --release`
- [ ] Setup install.rbee.dev hosting

---

## 🎯 Next Steps

1. **Setup mac** (install pnpm, setup runner)
2. **Setup workstation** (install Rust, pnpm)
3. **Setup AUR** (SSH keys, test PKGBUILD)
4. **Implement xtask release** (Phase 1-3 from previous plan)
5. **Test complete workflow**
6. **Deploy!**

Ready to proceed?
