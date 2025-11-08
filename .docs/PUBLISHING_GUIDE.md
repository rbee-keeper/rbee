# Publishing Guide

**Created by:** TEAM-451  
**Multi-Platform Publishing Strategy**

## 🎯 Overview

```
development → production (merge only) → Automatic Publishing
```

**You cannot push to production** - only merge from development via PR.

## 📦 What Gets Published

When you merge `development` → `production`:

### 1. GitHub Release (Automatic)
- ✅ Git tag created (e.g., `v0.2.0`)
- ✅ Changelog generated
- ✅ Linux binaries attached
- ✅ macOS binaries attached (from mac.home.arpa)
- ✅ Release notes published

### 2. Installation Methods

| Platform | Method | Builder | Status |
|----------|--------|---------|--------|
| **Linux** | `curl -fsSL https://install.rbee.dev \| sh` | GitHub Actions (Ubuntu) | ✅ Auto |
| **macOS** | `curl -fsSL https://install.rbee.dev \| sh` | Self-hosted (mac.home.arpa) | ✅ Auto |
| **Arch Linux** | `paru -S rbee` or `yay -S rbee` | Self-hosted (Arch) | ✅ Auto |
| **Homebrew** | `brew install rbee` | Manual (future) | 🔜 TODO |
| **Cargo** | `cargo install rbee` | Manual (future) | 🔜 TODO |

### 3. Commercial Site (Git Submodule)

**Location:** `frontend/apps/commercial` (git submodule)

**Dependencies:** Uses packages from parent repo:
- `@rbee/ui` (frontend/packages/rbee-ui)
- `@rbee/shared-config` (frontend/packages/shared-config)
- Other workspace packages

**How it works:**
1. Commercial site is a separate git repository
2. Included as submodule in main repo
3. pnpm workspace resolves dependencies from parent
4. Builds work because pnpm hoists packages

**Deployment:**
- Separate workflow (commercial site has its own CI/CD)
- Or: Build in main repo and deploy artifacts

---

## 🚀 Release Workflow

### Step 1: Bump Version

```bash
# Interactive version bumper
./scripts/bump-version.sh

# Or specify type directly
./scripts/bump-version.sh patch   # 0.1.0 → 0.1.1
./scripts/bump-version.sh minor   # 0.1.0 → 0.2.0
./scripts/bump-version.sh major   # 0.1.0 → 1.0.0
```

This will:
1. Ask you: Patch, Minor, or Major?
2. Update `Cargo.toml` and `package.json`
3. Generate changelog entry
4. Commit the version bump
5. Show you next steps

### Step 2: Push to Development

```bash
git push origin development
```

### Step 3: Create Release PR

```bash
gh pr create \
    --base production \
    --head development \
    --title "Release v0.2.0" \
    --body "$(cat <<EOF
## Release v0.2.0

### New Features
- Feature 1
- Feature 2

### Bug Fixes
- Fix 1
- Fix 2

### Breaking Changes
- None
EOF
)"
```

### Step 4: Wait for CI

PR triggers these checks:
- ✅ Build (cargo build --release)
- ✅ Test (cargo test --all)
- ✅ Clippy (cargo clippy)
- ✅ Frontend Build (pnpm run build)

**All must pass before merge!**

### Step 5: Merge to Production

```bash
# After approval + CI passes
gh pr merge --squash
```

**You cannot push directly to production!**

### Step 6: Automatic Publishing

When merged to production, GitHub Actions automatically:

1. **Linux Build** (GitHub Actions Ubuntu runner)
   - Builds binaries
   - Creates `install.sh` script
   - Uploads to release

2. **macOS Build** (Self-hosted on mac.home.arpa)
   - Triggers when tag is created
   - Builds binaries
   - Creates macOS package
   - Uploads to release

3. **AUR Publish** (Self-hosted on Arch)
   - Updates PKGBUILD
   - Generates checksums
   - Pushes to AUR

4. **GitHub Release**
   - Creates release with all binaries
   - Publishes changelog

---

## 🖥️ Self-Hosted Runners Setup

### Mac (mac.home.arpa)

**Install runner:**
```bash
# On your Mac
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download latest runner
curl -o actions-runner-osx-arm64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-osx-arm64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-osx-arm64.tar.gz

# Configure
./config.sh --url https://github.com/rbee-keeper/rbee --token <TOKEN>
# When prompted for labels, add: macos,self-hosted

# Install as service
./svc.sh install
./svc.sh start
```

**Get token:**
```bash
# On your main machine
gh api repos/rbee-keeper/rbee/actions/runners/registration-token | jq -r .token
```

### Arch Linux

**Install runner:**
```bash
# On your Arch machine
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download latest runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64.tar.gz

# Configure
./config.sh --url https://github.com/rbee-keeper/rbee --token <TOKEN>
# When prompted for labels, add: arch,linux,self-hosted

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

**Setup AUR access:**
```bash
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
```

---

## 📋 Installation Scripts

### install.rbee.dev

Create a simple install script that detects platform:

```bash
#!/bin/bash
# Universal installer for rbee
# Usage: curl -fsSL https://install.rbee.dev | sh

set -e

VERSION="${VERSION:-latest}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

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
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

echo "Installing rbee $VERSION for $PLATFORM-$ARCH..."

# Download from GitHub releases
RELEASE_URL="https://github.com/rbee-keeper/rbee/releases/download/v$VERSION/rbee-$PLATFORM-$ARCH.tar.gz"

# Download and extract
curl -fsSL "$RELEASE_URL" | tar -xz -C /tmp

# Install binaries
mkdir -p "$INSTALL_DIR"
cp /tmp/rbee-keeper "$INSTALL_DIR/"
cp /tmp/queen-rbee "$INSTALL_DIR/"
cp /tmp/rbee-hive "$INSTALL_DIR/"

chmod +x "$INSTALL_DIR/rbee-keeper"
chmod +x "$INSTALL_DIR/queen-rbee"
chmod +x "$INSTALL_DIR/rbee-hive"

echo "✅ rbee installed to $INSTALL_DIR"
echo ""
echo "Add to PATH:"
echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
echo ""
echo "Run: rbee-keeper --help"
```

**Host this script:**
- Option 1: GitHub Pages (create `install.rbee.dev` CNAME)
- Option 2: Cloudflare Pages
- Option 3: Simple static hosting

---

## 🔄 Commercial Site Submodule

### How It Works

```
rbee/
├── frontend/
│   ├── packages/
│   │   ├── rbee-ui/          # Shared UI components
│   │   ├── shared-config/    # Shared config
│   │   └── ...
│   └── apps/
│       ├── commercial/        # Git submodule (separate repo)
│       │   ├── package.json   # Depends on @rbee/ui
│       │   └── ...
│       └── marketplace/
└── pnpm-workspace.yaml        # Includes all packages
```

**pnpm workspace configuration:**

```yaml
# pnpm-workspace.yaml
packages:
  - 'frontend/packages/*'
  - 'frontend/apps/*'
  - 'frontend/apps/commercial'  # Submodule included!
```

**Commercial site package.json:**

```json
{
  "name": "@rbee/commercial",
  "dependencies": {
    "@rbee/ui": "workspace:*",
    "@rbee/shared-config": "workspace:*"
  }
}
```

**How pnpm resolves:**
1. pnpm sees `workspace:*` dependency
2. Looks in workspace for `@rbee/ui`
3. Finds it in `frontend/packages/rbee-ui`
4. Links it (symlink in node_modules)
5. Build works!

### Updating Submodule

```bash
# Update commercial site
cd frontend/apps/commercial
git pull origin main

# Commit submodule update in main repo
cd ../../..
git add frontend/apps/commercial
git commit -m "chore: update commercial site submodule"
git push origin development
```

### Building Commercial Site

```bash
# In main repo
pnpm install  # Installs all workspace packages
cd frontend/apps/commercial
pnpm run build  # Works! Uses @rbee/ui from parent workspace
```

---

## 🎯 Publishing Checklist

Before creating release PR:

- [ ] Version bumped (`./scripts/bump-version.sh`)
- [ ] Changelog updated
- [ ] All tests pass locally
- [ ] Commercial site submodule updated (if needed)
- [ ] Self-hosted runners are online
  - [ ] Mac (mac.home.arpa)
  - [ ] Arch Linux
- [ ] AUR SSH keys configured

After merging to production:

- [ ] GitHub Release created
- [ ] Linux binaries attached
- [ ] macOS binaries attached
- [ ] AUR package updated
- [ ] Install script works: `curl -fsSL https://install.rbee.dev | sh`

---

## 🚨 Troubleshooting

### "Cannot push to production"
✅ **Correct!** You can only merge via PR.

### "Self-hosted runner offline"
```bash
# On the runner machine
cd ~/actions-runner
./svc.sh status
./svc.sh start
```

### "AUR push failed"
```bash
# Check SSH key
ssh -T aur@aur.archlinux.org

# Should output: "Hi username! You've successfully authenticated..."
```

### "Commercial site build fails"
```bash
# Ensure submodule is initialized
git submodule update --init --recursive

# Reinstall dependencies
pnpm install
```

### "Install script 404"
- Ensure binaries are attached to release
- Check release URL matches script
- Verify GitHub release is published (not draft)

---

## 📚 References

- [GitHub Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [AUR Submission Guidelines](https://wiki.archlinux.org/title/AUR_submission_guidelines)
- [pnpm Workspaces](https://pnpm.io/workspaces)
- [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
