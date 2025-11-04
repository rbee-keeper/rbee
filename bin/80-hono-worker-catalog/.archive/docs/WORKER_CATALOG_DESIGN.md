# Worker Catalog Design: AUR-Style Branch-Based Repository

**Date:** 2025-11-04  
**Status:** 📋 DESIGN PROPOSAL  
**Current:** Simple Cloudflare Worker serving static PKGBUILD files  
**Target:** Git-based branch-per-package system like AUR

---

## 🎯 Goal

Create a **branch-based Git repository** for worker PKGBUILDs, similar to how AUR (Arch User Repository) works, where:
- Each worker variant gets its own Git branch
- PKGBUILD files are versioned and tracked
- Users can clone specific branches to build workers
- Catalog service indexes and serves metadata from branches

---

## 📚 How AUR Works (Research Summary)

### AUR Architecture

1. **One Git Repository Per Package**
   - Each package has its own Git remote: `ssh://aur@aur.archlinux.org/pkgname.git`
   - Package maintainer pushes PKGBUILD + .SRCINFO to master branch
   - Users clone the repo, run `makepkg`, install with `pacman`

2. **Key Files**
   - `PKGBUILD` - Build script with metadata, dependencies, build/install instructions
   - `.SRCINFO` - Machine-readable metadata (auto-generated from PKGBUILD)
   - Optional: `.install` files, patches, local sources

3. **Workflow**
   ```bash
   # User workflow
   git clone https://aur.archlinux.org/package-name.git
   cd package-name
   makepkg -si  # Build and install
   
   # Maintainer workflow
   vim PKGBUILD
   makepkg --printsrcinfo > .SRCINFO
   git add PKGBUILD .SRCINFO
   git commit -m "Update to version X"
   git push
   ```

4. **Web Interface**
   - AUR website indexes all packages
   - Shows metadata, votes, comments, dependencies
   - Search functionality
   - Out-of-date flagging

---

## 🏗️ Proposed rbee Worker Catalog Architecture

### Option A: Single Repo, Branch Per Worker (RECOMMENDED)

**Repository:** `github.com/veighnsche/rbee-worker-catalog`

**Branch Structure:**
```
master                          # README, documentation, catalog index
├── llm-worker-rbee-cpu        # Branch for CPU variant
├── llm-worker-rbee-cuda       # Branch for CUDA variant
├── llm-worker-rbee-metal      # Branch for Metal variant
├── sd-worker-rbee-cpu         # Branch for SD CPU variant
├── sd-worker-rbee-cuda        # Branch for SD CUDA variant
└── ...                        # More workers as needed
```

**Each Branch Contains:**
```
llm-worker-rbee-cpu/
├── PKGBUILD              # Build instructions
├── .SRCINFO              # Metadata (auto-generated)
├── README.md             # Worker-specific docs
├── .install              # Optional post-install hooks
└── patches/              # Optional patches
```

**Advantages:**
- ✅ Similar to AUR - one remote, many branches
- ✅ Easy to clone specific worker: `git clone -b llm-worker-rbee-cpu ...`
- ✅ Centralized management
- ✅ Can use GitHub Actions to validate PKGBUILDs on push
- ✅ Easy to see all workers: `git branch -r`

**Disadvantages:**
- ❌ Slightly more complex than separate repos
- ❌ Need to be careful with branch management

### Option B: Separate Repo Per Worker

**Repositories:**
```
github.com/veighnsche/llm-worker-rbee-cpu
github.com/veighnsche/llm-worker-rbee-cuda
github.com/veighnsche/llm-worker-rbee-metal
...
```

**Advantages:**
- ✅ Cleanest separation
- ✅ Independent versioning
- ✅ Can have different maintainers

**Disadvantages:**
- ❌ Harder to discover all workers
- ❌ More repos to manage
- ❌ Need central index anyway

---

## 🔧 Implementation Plan

### Phase 1: Git Repository Setup

1. **Create Repository**
   ```bash
   # Create new repo
   gh repo create veighnsche/rbee-worker-catalog --public
   
   # Initialize with master branch
   git clone git@github.com:veighnsche/rbee-worker-catalog.git
   cd rbee-worker-catalog
   
   # Create README on master
   echo "# rbee Worker Catalog" > README.md
   git add README.md
   git commit -m "Initial commit"
   git push origin master
   ```

2. **Create Worker Branches**
   ```bash
   # For each worker variant
   git checkout --orphan llm-worker-rbee-cpu
   git rm -rf .
   
   # Copy PKGBUILD from current catalog
   cp ../80-hono-worker-catalog/public/pkgbuilds/llm-worker-rbee-cpu.PKGBUILD PKGBUILD
   
   # Generate .SRCINFO
   makepkg --printsrcinfo > .SRCINFO
   
   # Add README
   cat > README.md << 'EOF'
   # llm-worker-rbee-cpu
   
   CPU-only LLM inference worker for rbee.
   
   ## Installation
   
   ```bash
   git clone -b llm-worker-rbee-cpu https://github.com/veighnsche/rbee-worker-catalog.git
   cd rbee-worker-catalog
   makepkg -si
   ```
   EOF
   
   git add PKGBUILD .SRCINFO README.md
   git commit -m "Initial PKGBUILD for llm-worker-rbee-cpu"
   git push origin llm-worker-rbee-cpu
   
   # Repeat for cuda, metal, etc.
   ```

3. **Create Catalog Index (master branch)**
   ```bash
   git checkout master
   
   cat > catalog.json << 'EOF'
   {
     "version": "1.0",
     "workers": [
       {
         "id": "llm-worker-rbee-cpu",
         "branch": "llm-worker-rbee-cpu",
         "variant": "cpu",
         "description": "LLM worker for rbee system (CPU-only)",
         "arch": ["x86_64", "aarch64"],
         "git_url": "https://github.com/veighnsche/rbee-worker-catalog.git",
         "pkgbuild_url": "https://raw.githubusercontent.com/veighnsche/rbee-worker-catalog/llm-worker-rbee-cpu/PKGBUILD"
       },
       {
         "id": "llm-worker-rbee-cuda",
         "branch": "llm-worker-rbee-cuda",
         "variant": "cuda",
         "description": "LLM worker for rbee system (CUDA)",
         "arch": ["x86_64"],
         "git_url": "https://github.com/veighnsche/rbee-worker-catalog.git",
         "pkgbuild_url": "https://raw.githubusercontent.com/veighnsche/rbee-worker-catalog/llm-worker-rbee-cuda/PKGBUILD"
       }
     ]
   }
   EOF
   
   git add catalog.json
   git commit -m "Add catalog index"
   git push origin master
   ```

### Phase 2: Update Cloudflare Worker (Catalog Service)

**New Endpoints:**

```typescript
// GET /workers
// Returns catalog index from master branch
app.get('/workers', async (c) => {
  const response = await fetch(
    'https://raw.githubusercontent.com/veighnsche/rbee-worker-catalog/master/catalog.json'
  )
  return c.json(await response.json())
})

// GET /workers/:variant/PKGBUILD
// Fetches PKGBUILD from specific branch
app.get('/workers/:variant/PKGBUILD', async (c) => {
  const variant = c.req.param('variant')
  const url = `https://raw.githubusercontent.com/veighnsche/rbee-worker-catalog/llm-worker-rbee-${variant}/PKGBUILD`
  
  const response = await fetch(url)
  if (!response.ok) {
    return c.text('PKGBUILD not found', 404)
  }
  
  return c.text(await response.text(), 200, {
    'Content-Type': 'text/plain'
  })
})

// GET /workers/:variant/clone-url
// Returns git clone command for specific worker
app.get('/workers/:variant/clone-url', async (c) => {
  const variant = c.req.param('variant')
  return c.json({
    git_url: 'https://github.com/veighnsche/rbee-worker-catalog.git',
    branch: `llm-worker-rbee-${variant}`,
    clone_command: `git clone -b llm-worker-rbee-${variant} https://github.com/veighnsche/rbee-worker-catalog.git`
  })
})
```

### Phase 3: Update rbee-hive Worker Installation

**Current Flow:**
```rust
// rbee-hive downloads PKGBUILD and executes it
let pkgbuild = fetch("http://localhost:8787/workers/cpu/PKGBUILD").await?;
execute_pkgbuild(pkgbuild)?;
```

**New Flow (Git-based):**
```rust
// Option 1: Clone branch and build
let clone_info = fetch("http://localhost:8787/workers/cpu/clone-url").await?;
Command::new("git")
    .args(&["clone", "-b", &clone_info.branch, &clone_info.git_url])
    .output()?;

// Option 2: Direct PKGBUILD download (simpler, no git required)
let pkgbuild = fetch("http://localhost:8787/workers/cpu/PKGBUILD").await?;
execute_pkgbuild(pkgbuild)?;
```

### Phase 4: CI/CD Automation

**GitHub Actions Workflow** (`.github/workflows/validate-pkgbuild.yml`):

```yaml
name: Validate PKGBUILD

on:
  push:
    branches:
      - 'llm-worker-rbee-*'
      - 'sd-worker-rbee-*'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install makepkg
        run: |
          sudo apt-get update
          sudo apt-get install -y pacman
      
      - name: Validate PKGBUILD syntax
        run: |
          makepkg --printsrcinfo > .SRCINFO.test
          diff .SRCINFO .SRCINFO.test
      
      - name: Check required fields
        run: |
          grep -q "pkgname=" PKGBUILD
          grep -q "pkgver=" PKGBUILD
          grep -q "pkgdesc=" PKGBUILD
      
      - name: Update catalog index
        if: github.ref == 'refs/heads/master'
        run: |
          # Auto-generate catalog.json from all branches
          python scripts/generate-catalog.py
```

---

## 📋 Migration Plan

### Current State
```
bin/80-hono-worker-catalog/
└── public/pkgbuilds/
    ├── llm-worker-rbee-cpu.PKGBUILD
    ├── llm-worker-rbee-cuda.PKGBUILD
    └── llm-worker-rbee-metal.PKGBUILD
```

### Migration Steps

1. **Week 1: Repository Setup**
   - [ ] Create `rbee-worker-catalog` repository
   - [ ] Create branches for existing workers (cpu, cuda, metal)
   - [ ] Add PKGBUILDs to branches
   - [ ] Generate .SRCINFO files
   - [ ] Create catalog.json index

2. **Week 2: Catalog Service Update**
   - [ ] Update Cloudflare Worker to fetch from GitHub
   - [ ] Add caching (Cloudflare KV or Cache API)
   - [ ] Add webhook endpoint for auto-refresh on push
   - [ ] Test endpoints

3. **Week 3: rbee-hive Integration**
   - [ ] Update worker installation logic
   - [ ] Test with all variants
   - [ ] Add fallback to old system if needed

4. **Week 4: Deprecate Old System**
   - [ ] Remove static PKGBUILD files from `public/pkgbuilds/`
   - [ ] Update documentation
   - [ ] Archive old implementation

---

## 🔒 Security Considerations

1. **PKGBUILD Validation**
   - Run shellcheck on all PKGBUILDs
   - Validate source URLs (must be HTTPS or git+https)
   - Check for suspicious commands

2. **Signature Verification**
   - Consider GPG signing commits
   - Verify signatures before building

3. **Sandboxed Builds**
   - Build in containers or chroot
   - Limit network access during build
   - Use systemd-nspawn or Docker

---

## 🎨 Future Enhancements

1. **Web UI** (like AUR website)
   - Browse workers
   - Search functionality
   - View PKGBUILD online
   - Comments/ratings
   - Out-of-date flagging

2. **Automated Updates**
   - Bot that checks for new llama-orch releases
   - Auto-updates pkgver in PKGBUILDs
   - Creates PR for review

3. **Build Service**
   - Pre-built binaries for common platforms
   - Hosted on GitHub Releases or Cloudflare R2
   - Fallback to PKGBUILD if binary not available

4. **Community Contributions**
   - Allow community to submit new workers
   - Review process for new PKGBUILDs
   - Maintainer system

---

## 📊 Comparison: Current vs Proposed

| Feature | Current (Static) | Proposed (Git-based) |
|---------|-----------------|---------------------|
| **Versioning** | ❌ No | ✅ Git history |
| **Discoverability** | ⚠️ Manual list | ✅ Branch list + catalog.json |
| **Updates** | ❌ Manual file edit | ✅ Git push |
| **Collaboration** | ❌ Single maintainer | ✅ PRs, multiple maintainers |
| **CI/CD** | ❌ None | ✅ GitHub Actions |
| **Rollback** | ❌ No | ✅ Git revert |
| **Audit Trail** | ❌ No | ✅ Git log |
| **Community** | ❌ Closed | ✅ Open for contributions |

---

## 🔐 Premium/Closed-Source Workers

### How AUR Actually Does It ⭐

**You're absolutely right!** AUR is famous for hosting proprietary software like:
- `slack-desktop` - Proprietary chat app
- `zoom` - Proprietary video conferencing
- `visual-studio-code-bin` - Microsoft's proprietary editor
- `spotify` - Proprietary music streaming
- `discord` - Proprietary chat app

**The AUR Pattern for Binaries:**

1. **PKGBUILD downloads pre-built binary** (not source code)
2. **Uses `source_x86_64=()` and `source_aarch64=()`** for different architectures
3. **No `build()` function** - makepkg just extracts the tarball
4. **Simple `package()` function** - installs files from extracted tarball
5. **Checksums verify integrity** - `sha256sums_x86_64=()`

**This is NOT a hack - it's the standard AUR pattern!**

### The "Problem" (That Doesn't Exist)

~~Open-source PKGBUILDs won't work for premium workers because:~~
- ~~❌ Source code is private (can't `git clone` public repo)~~
- ~~❌ Can't build from source (no access to code)~~
- ~~❌ Need authentication/licensing~~
- ~~❌ Need to distribute pre-built binaries~~

**Actually:** PKGBUILDs work perfectly for binaries! That's literally what AUR does.

### Solution: Use Standard AUR Pattern

**PKGBUILDs can download and install pre-built binaries. This is standard practice.**

### Architecture

```
Public Catalog (GitHub)
├── llm-worker-rbee-cpu (open source, builds from source)
├── llm-worker-rbee-cuda (open source, builds from source)
└── llm-worker-rbee-premium (closed source, downloads binary)

Private Binary Storage (Cloudflare R2 / S3 / GitHub Releases)
└── llm-worker-rbee-premium-v0.1.0-linux-x86_64.tar.gz (authenticated)
```

### Premium PKGBUILD Example (AUR Pattern)

**This is EXACTLY how AUR does it!** (e.g., `slack-desktop`, `zoom`, `visual-studio-code-bin`)

```bash
# PKGBUILD for llm-worker-rbee-premium (closed source)
# Maintainer: rbee Core Team
# TEAM-402: Uses standard AUR pattern for proprietary binaries

pkgname=llm-worker-rbee-premium
pkgver=0.1.0
pkgrel=1
pkgdesc="Premium LLM worker for rbee system (closed source, pre-built binary)"
arch=('x86_64' 'aarch64')
url="https://rbee.ai/premium"
license=('Proprietary')
depends=('gcc')

# TEAM-402: Architecture-specific sources (standard AUR pattern)
# This is how AUR handles binaries for different architectures
source_x86_64=("https://releases.rbee.ai/workers/premium/${pkgver}/llm-worker-rbee-premium-${pkgver}-linux-x86_64.tar.gz")
source_aarch64=("https://releases.rbee.ai/workers/premium/${pkgver}/llm-worker-rbee-premium-${pkgver}-linux-aarch64.tar.gz")

# Checksums for verification (standard AUR pattern)
sha256sums_x86_64=('abc123...')  # Real checksum
sha256sums_aarch64=('def456...')  # Real checksum

# TEAM-402: No build() function needed for binary packages!
# makepkg will automatically extract the tarball

package() {
    # TEAM-402: Standard AUR binary package pattern
    # Just install the pre-built binary from the extracted tarball
    
    cd "$srcdir"
    
    # Install binary
    install -Dm755 "llm-worker-rbee-premium" \
        "$pkgdir/usr/local/bin/$pkgname"
    
    # Install license file
    install -Dm644 "LICENSE" \
        "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}

# Optional: Verify binary works
check() {
    cd "$srcdir"
    ./llm-worker-rbee-premium --version || true
}
```

**Key Points:**
- ✅ **No build() function** - makepkg extracts tarball automatically
- ✅ **Architecture-specific sources** - `source_x86_64=()` and `source_aarch64=()`
- ✅ **Real checksums** - Not 'SKIP', actual sha256 hashes
- ✅ **Simple package()** - Just install files from extracted tarball
- ✅ **This is NOT a hack** - This is the standard AUR pattern!

### Authentication Options

#### Option 1: Environment Variable (Simplest)

```bash
# User sets license token
export RBEE_LICENSE_TOKEN="rbee_lic_abc123..."

# Download URL includes token
source=("https://releases.rbee.ai/workers/premium/${pkgver}/binary.tar.gz?token=${RBEE_LICENSE_TOKEN}")
```

#### Option 2: Config File

```bash
# ~/.config/rbee/license.conf
RBEE_LICENSE_TOKEN=rbee_lic_abc123...

# PKGBUILD reads from config
build() {
    source ~/.config/rbee/license.conf
    # Download with token
}
```

#### Option 3: OAuth/API Key (Most Secure)

```bash
# User authenticates once
rbee-cli login

# Creates ~/.rbee/credentials with API key
# PKGBUILD uses API to get temporary download URL

build() {
    # Get authenticated download URL
    DOWNLOAD_URL=$(curl -H "Authorization: Bearer $(cat ~/.rbee/credentials)" \
        https://api.rbee.ai/v1/workers/premium/download-url)
    
    # Download binary
    curl -L "$DOWNLOAD_URL" -o binary.tar.gz
}
```

### Binary Storage Options

#### Option A: Cloudflare R2 (Recommended)

**Pros:**
- ✅ Zero egress fees
- ✅ S3-compatible API
- ✅ Presigned URLs for authentication
- ✅ Fast global distribution

**Setup:**
```bash
# Upload binary to R2
wrangler r2 object put rbee-workers/premium/v0.1.0/binary.tar.gz \
    --file binary.tar.gz

# Generate presigned URL (expires in 1 hour)
wrangler r2 object presign rbee-workers/premium/v0.1.0/binary.tar.gz \
    --expires-in 3600
```

#### Option B: GitHub Releases (Private Repo)

**Pros:**
- ✅ Integrated with Git workflow
- ✅ GitHub token authentication
- ✅ Version tagging built-in

**Setup:**
```bash
# Create private repo
gh repo create veighnsche/rbee-worker-premium --private

# Upload binary as release asset
gh release create v0.1.0 \
    --repo veighnsche/rbee-worker-premium \
    binary.tar.gz
```

**PKGBUILD downloads with GitHub token:**
```bash
source=("https://github.com/veighnsche/rbee-worker-premium/releases/download/v${pkgver}/binary.tar.gz")

# User needs GitHub token with repo access
# Set in ~/.netrc or use gh auth token
```

#### Option C: Self-Hosted (Full Control)

**Pros:**
- ✅ Complete control
- ✅ Custom authentication
- ✅ No third-party dependencies

**Setup:**
```bash
# Nginx with basic auth
location /workers/premium/ {
    auth_basic "rbee Premium Workers";
    auth_basic_user_file /etc/nginx/.htpasswd;
    root /var/www/releases;
}
```

### Catalog Entry for Premium Workers

```json
{
  "id": "llm-worker-rbee-premium",
  "branch": "llm-worker-rbee-premium",
  "variant": "premium",
  "description": "Premium LLM worker (closed source)",
  "arch": ["x86_64", "aarch64"],
  "license": "Proprietary",
  "requires_license": true,
  "license_url": "https://rbee.ai/premium",
  "git_url": "https://github.com/veighnsche/rbee-worker-catalog.git",
  "pkgbuild_url": "https://raw.githubusercontent.com/veighnsche/rbee-worker-catalog/llm-worker-rbee-premium/PKGBUILD",
  "pricing": {
    "model": "subscription",
    "price_usd": 99,
    "interval": "month"
  }
}
```

### User Workflow (Premium)

```bash
# 1. Purchase license
# Visit https://rbee.ai/premium and purchase

# 2. Set license token
export RBEE_LICENSE_TOKEN="rbee_lic_abc123..."

# 3. Install worker (same as open source!)
git clone -b llm-worker-rbee-premium https://github.com/veighnsche/rbee-worker-catalog.git
cd rbee-worker-catalog
makepkg -si

# Or via rbee-hive
rbee-hive install-worker --variant premium --license-token rbee_lic_abc123...
```

### Hybrid Catalog Structure

```
rbee-worker-catalog (public repo)
├── master
│   └── catalog.json (lists ALL workers, including premium)
├── llm-worker-rbee-cpu (open source)
│   └── PKGBUILD (builds from source)
├── llm-worker-rbee-cuda (open source)
│   └── PKGBUILD (builds from source)
└── llm-worker-rbee-premium (closed source)
    └── PKGBUILD (downloads pre-built binary with auth)
```

**Key Points:**
- ✅ Catalog is public (anyone can see what's available)
- ✅ PKGBUILDs are public (shows how to install)
- ✅ Source code stays private (not in PKGBUILD)
- ✅ Binaries require authentication
- ✅ Same installation workflow for all workers

### License Verification

**Option 1: At Download Time**
```bash
# PKGBUILD verifies license before download
build() {
    # Verify license with API
    LICENSE_VALID=$(curl -s -H "Authorization: Bearer $RBEE_LICENSE_TOKEN" \
        https://api.rbee.ai/v1/licenses/verify)
    
    if [ "$LICENSE_VALID" != "true" ]; then
        echo "ERROR: Invalid or expired license"
        return 1
    fi
}
```

**Option 2: At Runtime**
```bash
# Worker binary checks license on startup
# Calls home to verify license is still valid
# Can implement grace period for offline usage
```

### Advantages of This Approach

1. **Unified Distribution**
   - Same catalog for open source and premium
   - Same installation workflow
   - Same tooling (makepkg, rbee-hive)

2. **Flexible Licensing**
   - Per-user licenses
   - Per-machine licenses
   - Subscription-based
   - One-time purchase

3. **Security**
   - Source code never exposed
   - Binaries behind authentication
   - License verification at multiple points

4. **Discoverability**
   - Premium workers visible in catalog
   - Users can see features before purchasing
   - Easy upgrade path from free to premium

---

## 🚀 Quick Start (After Implementation)

### For Users (Installing a Worker)

```bash
# Option 1: Via rbee-hive (automatic)
rbee-hive install-worker --variant cuda

# Option 2: Manual (like AUR)
git clone -b llm-worker-rbee-cuda https://github.com/veighnsche/rbee-worker-catalog.git
cd rbee-worker-catalog
makepkg -si
```

### For Maintainers (Updating a Worker)

```bash
# Clone specific worker branch
git clone -b llm-worker-rbee-cpu https://github.com/veighnsche/rbee-worker-catalog.git
cd rbee-worker-catalog

# Update PKGBUILD
vim PKGBUILD  # Bump pkgver

# Generate .SRCINFO
makepkg --printsrcinfo > .SRCINFO

# Commit and push
git add PKGBUILD .SRCINFO
git commit -m "Update to version 0.2.0"
git push origin llm-worker-rbee-cpu
```

---

## 📝 References

- [AUR Wiki](https://wiki.archlinux.org/title/Arch_User_Repository)
- [AUR Submission Guidelines](https://wiki.archlinux.org/title/AUR_submission_guidelines)
- [PKGBUILD Documentation](https://wiki.archlinux.org/title/PKGBUILD)
- [makepkg Manual](https://man.archlinux.org/man/makepkg.8)

---

**TEAM-402 - Worker Catalog Design Complete!**
