# Production Marketplace Plan (AUR-Like)

**Purpose:** Transform worker catalog into production-ready marketplace  
**Model:** AUR (Arch User Repository) - Community-driven package system  
**Status:** Consolidation of existing plans + AUR research

---

## 🎯 Vision: The AUR for AI Workers

**Goal:** Anyone can submit workers, just like AUR!

**Inspiration from AUR:**
- ✅ Community-driven submissions
- ✅ Git-based workflow
- ✅ PKGBUILDs for both source AND binaries
- ✅ Decentralized (anyone can host)
- ✅ Trusted through community review

---

## 📚 Existing Documentation (Already Done!)

### 1. AUR Binary Pattern Research
**File:** `.archive/docs/AUR_BINARY_PATTERN.md`

**Key Discoveries:**
- ✅ PKGBUILDs can distribute binaries (not just source!)
- ✅ Slack, VS Code, Zoom all use this pattern
- ✅ Architecture-specific sources built-in (`source_x86_64=()`)
- ✅ No `build()` function needed for binaries
- ✅ This is the STANDARD pattern, not a hack

**Example from AUR:**
```bash
# VS Code PKGBUILD
pkgname=visual-studio-code-bin
source_x86_64=("https://update.code.visualstudio.com/${pkgver}/linux-x64/stable")
source_aarch64=("https://update.code.visualstudio.com/${pkgver}/linux-arm64/stable")

package() {
    # Just install the pre-built binary!
    install -Dm755 "code" "$pkgdir/usr/bin/code"
}
```

### 2. Dynamic Worker Catalog Plan
**File:** `DYNAMIC_WORKER_CATALOG_PLAN.md`

**Key Points:**
- ✅ Workers discovered from `bin/` directory
- ✅ Metadata in `worker.toml` per worker
- ✅ Version read from `Cargo.toml` automatically
- ✅ PKGBUILDs generated on-the-fly
- ✅ Marketplace-ready architecture

### 3. Vision Document
**File:** `.archive/docs/VISION.md`

**Long-term Goals:**
- Community features (ratings, reviews)
- Advanced search & discovery
- Build service (automated builds)
- Enterprise features
- Marketplace UI
- Developer SDK

---

## 🏗️ How AUR Works (Our Model)

### AUR Architecture

```
1. Developer creates PKGBUILD
   ↓
2. Submits to AUR (Git repository)
   ↓
3. Community reviews & votes
   ↓
4. Users discover via search
   ↓
5. Users install with makepkg
```

### AUR Submission Process

```bash
# 1. Clone AUR package
git clone ssh://aur@aur.archlinux.org/mypackage.git

# 2. Create PKGBUILD
cat > PKGBUILD << 'EOF'
pkgname=mypackage
pkgver=1.0.0
source=("https://releases.example.com/mypackage.tar.gz")
package() {
    install -Dm755 "mypackage" "$pkgdir/usr/bin/mypackage"
}
EOF

# 3. Generate metadata
makepkg --printsrcinfo > .SRCINFO

# 4. Commit and push
git add PKGBUILD .SRCINFO
git commit -m "Initial commit"
git push
```

**That's it!** Package is now in AUR.

---

## ✅ rbee Marketplace Architecture (AUR-Like)

### Phase 1: Git-Based Catalog (MVP - Week 1-2)

**Structure:**
```
github.com/rbee-keeper/worker-catalog/
├── llm-worker-rbee-cpu/
│   ├── PKGBUILD              ← Source build
│   ├── PKGBUILD.bin          ← Binary build
│   ├── worker.toml           ← Metadata
│   └── README.md
├── llm-worker-rbee-cuda/
│   ├── PKGBUILD
│   ├── PKGBUILD.bin
│   ├── worker.toml
│   └── README.md
└── community/                 ← Community submissions!
    ├── audio-worker/
    ├── video-worker/
    └── tts-worker/
```

**Submission Process:**
```bash
# 1. Fork worker-catalog repo
gh repo fork rbee-keeper/worker-catalog

# 2. Create worker directory
mkdir -p community/my-custom-worker
cd community/my-custom-worker

# 3. Create PKGBUILD (binary)
cat > PKGBUILD.bin << 'EOF'
pkgname=my-custom-worker-bin
pkgver=1.0.0
source_x86_64=("https://github.com/me/my-worker/releases/download/v${pkgver}/my-worker-linux-x86_64.tar.gz")

package() {
    install -Dm755 "my-worker" "$pkgdir/usr/local/bin/my-custom-worker"
}
EOF

# 4. Create worker.toml
cat > worker.toml << 'EOF'
[worker]
id = "my-custom-worker"
name = "My Custom Worker"
description = "Does amazing things"
license = "MIT"

[capabilities]
supported_formats = ["gguf"]
supports_streaming = true
EOF

# 5. Submit PR
git add .
git commit -m "Add my-custom-worker"
gh pr create --title "Add my-custom-worker"
```

**Review Process:**
1. Automated checks (PKGBUILD syntax, checksums)
2. Community review (security, quality)
3. Maintainer approval
4. Merge to main
5. Worker appears in catalog!

### Phase 2: Dynamic Discovery (Week 2-3)

**Worker Catalog API:**
```typescript
// Discovers workers from Git repo
GET /v1/workers
{
  "workers": [
    {
      "id": "llm-worker-rbee-cpu",
      "version": "0.1.0",  // ← From Cargo.toml
      "source": "official",
      "pkgbuild_url": "https://raw.githubusercontent.com/rbee-keeper/worker-catalog/main/llm-worker-rbee-cpu/PKGBUILD.bin"
    },
    {
      "id": "my-custom-worker",
      "version": "1.0.0",
      "source": "community",
      "pkgbuild_url": "https://raw.githubusercontent.com/rbee-keeper/worker-catalog/main/community/my-custom-worker/PKGBUILD.bin"
    }
  ]
}
```

**Installation:**
```bash
# rbee-keeper downloads PKGBUILD from catalog
rbee worker install my-custom-worker

# Behind the scenes:
# 1. Fetch PKGBUILD from catalog
# 2. Download binary from GitHub releases
# 3. Verify checksums
# 4. Install to ~/.local/bin
```

### Phase 3: Community Features (Month 2-3)

**Ratings & Reviews:**
```typescript
POST /v1/workers/:id/reviews
{
  "rating": 5,
  "comment": "Amazing performance!",
  "verified_install": true
}
```

**Stars & Favorites:**
```bash
rbee worker star llm-worker-rbee-cpu
rbee worker list --starred
```

**Tags & Discovery:**
```bash
rbee worker search --tag llm --tag cuda
rbee worker search "fast inference"
```

### Phase 4: Build Service (Month 4-5)

**Automated Builds:**
```yaml
# .github/workflows/build-worker.yml
on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu, macos, windows]
        arch: [x86_64, aarch64]
        variant: [cpu, cuda, metal]
    
    steps:
      - name: Build
        run: cargo build --release --features ${{ matrix.variant }}
      
      - name: Upload to GitHub Releases
        run: gh release upload ${{ github.ref_name }} binary.tar.gz
      
      - name: Update PKGBUILD
        run: |
          # Auto-update PKGBUILD with new version and checksums
          ./scripts/update-pkgbuild.sh
```

**Result:** Every release automatically:
- ✅ Builds for all platforms
- ✅ Uploads to GitHub Releases
- ✅ Updates PKGBUILDs
- ✅ Updates catalog

---

## 🔐 Premium Workers (AUR Pattern)

### How AUR Handles Proprietary Software

**Option 1: Public Binary (Most Common)**
```bash
# Slack, VS Code, Zoom all use this
source_x86_64=("https://downloads.slack-edge.com/releases/linux/${pkgver}/prod/x64/slack-desktop-${pkgver}-amd64.deb")
```

**Option 2: Authenticated Download**
```bash
# User provides token
source_x86_64=("https://releases.example.com/premium-worker.tar.gz?token=${RBEE_TOKEN}")
```

**Option 3: Manual Download**
```bash
# PKGBUILD instructs user to download manually
source=("premium-worker.tar.gz::SKIP")
# User must download to same directory as PKGBUILD
```

**For rbee Premium Workers:**
```bash
# PKGBUILD for premium worker
pkgname=premium-llm-worker-bin
pkgver=1.0.0

# Requires license key
if [ -z "$RBEE_LICENSE_KEY" ]; then
    error "RBEE_LICENSE_KEY not set. Get your key from https://rbee.dev/licenses"
    exit 1
fi

source_x86_64=("https://releases.rbee.dev/premium/${pkgname}-${pkgver}.tar.gz?key=${RBEE_LICENSE_KEY}")

package() {
    install -Dm755 "premium-llm-worker" "$pkgdir/usr/local/bin/premium-llm-worker"
}
```

**Installation:**
```bash
# User sets license key
export RBEE_LICENSE_KEY="your-key-here"

# Install works normally
rbee worker install premium-llm-worker
```

---

## 📊 Comparison: AUR vs rbee Marketplace

| Feature | AUR | rbee Marketplace |
|---------|-----|------------------|
| **Submission** | Git PR | Git PR |
| **Review** | Community | Community + Automated |
| **Distribution** | PKGBUILDs | PKGBUILDs + API |
| **Binaries** | ✅ Supported | ✅ Supported |
| **Source** | ✅ Supported | ✅ Supported |
| **Premium** | ✅ Supported | ✅ Supported |
| **Discovery** | Web search | API + Web + CLI |
| **Analytics** | ❌ None | ✅ Built-in |
| **Ratings** | ❌ None | ✅ Planned |

---

## 🚀 Implementation Roadmap

### Week 1-2: Git Catalog MVP
- [ ] Create `worker-catalog` repository
- [ ] Move existing workers to catalog
- [ ] Create submission guidelines
- [ ] Add automated checks (CI)
- [ ] Document submission process

### Week 2-3: Dynamic Discovery
- [ ] Implement worker discovery from Git
- [ ] Read metadata from `worker.toml`
- [ ] Generate PKGBUILDs on-the-fly
- [ ] Update API endpoints
- [ ] Deploy to production

### Week 3-4: Community Features
- [ ] Add worker ratings
- [ ] Add worker reviews
- [ ] Add stars/favorites
- [ ] Add tag system
- [ ] Add search API

### Month 2-3: Build Service
- [ ] Set up GitHub Actions
- [ ] Automated builds for all platforms
- [ ] Auto-update PKGBUILDs
- [ ] Binary hosting on R2
- [ ] Checksum verification

### Month 4+: Advanced Features
- [ ] Marketplace UI
- [ ] Advanced analytics
- [ ] Enterprise features
- [ ] Developer SDK
- [ ] AI-powered recommendations

---

## ✅ Success Criteria

**MVP (Week 2):**
- ✅ Anyone can submit a worker via PR
- ✅ Workers appear in catalog automatically
- ✅ Users can install community workers
- ✅ PKGBUILDs support binaries

**Production (Month 1):**
- ✅ 10+ community workers
- ✅ Automated review process
- ✅ Ratings & reviews live
- ✅ Search working

**Scale (Month 3):**
- ✅ 50+ workers in catalog
- ✅ 1,000+ active users
- ✅ Automated builds working
- ✅ Premium workers supported

---

## 🎯 Key Insights from AUR Research

1. **PKGBUILDs are NOT just for source builds**
   - Slack, VS Code, Zoom all use PKGBUILDs for binaries
   - This is the STANDARD pattern, not a hack

2. **Git-based submission works at scale**
   - AUR has 80,000+ packages
   - All submitted via Git
   - Community review works

3. **Premium software is supported**
   - Many proprietary apps in AUR
   - Authentication via tokens/keys
   - Manual download option

4. **Architecture-specific sources are built-in**
   - `source_x86_64=()`, `source_aarch64=()`
   - makepkg handles it automatically

5. **Community trust through transparency**
   - All PKGBUILDs are public
   - Anyone can review
   - Voting system for quality

---

## 📝 Next Steps

1. **Read existing docs:**
   - `.archive/docs/AUR_BINARY_PATTERN.md`
   - `.archive/docs/VISION.md`
   - `DYNAMIC_WORKER_CATALOG_PLAN.md`

2. **Implement Git catalog:**
   - Create `worker-catalog` repo
   - Move existing workers
   - Document submission process

3. **Enable community submissions:**
   - Create PR template
   - Add automated checks
   - Write contributor guide

4. **Deploy dynamic discovery:**
   - Implement worker discovery
   - Update API
   - Deploy to production

**The architecture is already designed. The research is done. Time to build!** 🚀
