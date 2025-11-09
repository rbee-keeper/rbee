# AUR Binary Pattern: How Proprietary Software Works

**Date:** 2025-11-04  
**Discovery:** PKGBUILDs are NOT just for source builds!

---

## 🎯 The Revelation

**AUR successfully hosts thousands of proprietary binaries using PKGBUILDs.**

Examples:
- `slack-desktop` - Slack chat app
- `zoom` - Zoom video conferencing  
- `visual-studio-code-bin` - VS Code
- `spotify` - Spotify music
- `discord` - Discord chat
- `google-chrome` - Chrome browser
- `dropbox` - Dropbox sync

**All of these use PKGBUILDs to download and install pre-built binaries!**

---

## 📦 The Pattern

### Standard Source Build PKGBUILD

```bash
pkgname=myapp
pkgver=1.0.0
source=("git+https://github.com/user/myapp.git")

build() {
    cd "$srcdir/myapp"
    cargo build --release
}

package() {
    install -Dm755 "target/release/myapp" "$pkgdir/usr/bin/myapp"
}
```

### Binary Package PKGBUILD (AUR Standard)

```bash
pkgname=myapp-bin  # Note: -bin suffix
pkgver=1.0.0
source_x86_64=("https://releases.example.com/myapp-${pkgver}-linux-x86_64.tar.gz")
source_aarch64=("https://releases.example.com/myapp-${pkgver}-linux-aarch64.tar.gz")
sha256sums_x86_64=('abc123...')
sha256sums_aarch64=('def456...')

# NO build() function!
# makepkg automatically extracts the tarball

package() {
    cd "$srcdir"
    install -Dm755 "myapp" "$pkgdir/usr/bin/myapp"
}
```

**Key Differences:**
- ✅ Uses `source_x86_64=()` instead of `source=()`
- ✅ Downloads binary tarball, not source code
- ✅ NO `build()` function
- ✅ Simple `package()` - just install files
- ✅ Usually has `-bin` suffix in package name

---

## 🔍 Real Examples from AUR

### Example 1: Slack Desktop

```bash
# From AUR: slack-desktop
pkgname=slack-desktop
pkgver=4.35.126
arch=('x86_64')
license=('custom')

source_x86_64=("https://downloads.slack-edge.com/releases/linux/${pkgver}/prod/x64/slack-desktop-${pkgver}-amd64.deb")
sha256sums_x86_64=('...')

package() {
    # Extract .deb and install files
    bsdtar -xf data.tar.xz -C "$pkgdir/"
}
```

### Example 2: Visual Studio Code

```bash
# From AUR: visual-studio-code-bin
pkgname=visual-studio-code-bin
pkgver=1.84.2
arch=('x86_64' 'aarch64')
license=('custom: commercial')

source_x86_64=("code_x64_${pkgver}.tar.gz::https://update.code.visualstudio.com/${pkgver}/linux-x64/stable")
source_aarch64=("code_arm64_${pkgver}.tar.gz::https://update.code.visualstudio.com/${pkgver}/linux-arm64/stable")

sha256sums_x86_64=('...')
sha256sums_aarch64=('...')

package() {
    install -d "$pkgdir/usr/share/code"
    cp -r VSCode-linux-*/* "$pkgdir/usr/share/code"
    install -Dm755 "$pkgdir/usr/share/code/bin/code" "$pkgdir/usr/bin/code"
}
```

### Example 3: Zoom

```bash
# From AUR: zoom
pkgname=zoom
pkgver=5.16.10
arch=('x86_64')
license=('custom')

source_x86_64=("https://zoom.us/client/${pkgver}/zoom_x86_64.pkg.tar.xz")
sha256sums_x86_64=('...')

package() {
    bsdtar -xf data.tar.xz -C "$pkgdir/"
}
```

---

## 🎨 Naming Conventions

### AUR Naming Pattern

- **Source build:** `myapp` - Builds from source
- **Binary package:** `myapp-bin` - Pre-built binary
- **Git version:** `myapp-git` - Latest from Git
- **Development:** `myapp-dev` - Development version

**For rbee workers:**
- `llm-worker-rbee-cpu` - Builds from source (optional)
- `llm-worker-rbee-cpu-bin` - Pre-built binary (fast)
- `llm-worker-rbee-premium` - Premium binary (no -bin suffix needed)

---

## 🔐 Authentication for Premium Binaries

### How AUR Handles Restricted Downloads

**Option 1: Public URL (Most Common)**
```bash
# Binary is publicly downloadable
source_x86_64=("https://releases.example.com/app.tar.gz")
```

**Option 2: Token in URL**
```bash
# User provides token via environment variable
source_x86_64=("https://releases.example.com/app.tar.gz?token=${MY_TOKEN}")
```

**Option 3: Custom DLAGENT**
```bash
# Use custom download agent with authentication
# Set in /etc/makepkg.conf or ~/.makepkg.conf
DLAGENTS=('https::/usr/bin/curl -fLC - --retry 3 --retry-delay 3 -H "Authorization: Bearer ${MY_TOKEN}" -o %o %u')
```

**Option 4: Manual Download**
```bash
# PKGBUILD instructs user to download manually
source=("app.tar.gz::SKIP")
# User must download app.tar.gz to same directory as PKGBUILD
```

---

## 💡 Key Insights

### 1. PKGBUILDs Are Not Just for Source Builds

**Wrong assumption:** "PKGBUILDs are for building from source"  
**Reality:** PKGBUILDs are for **packaging**, not necessarily **building**

### 2. makepkg Handles Extraction Automatically

**You don't need a build() function!**
- makepkg automatically extracts `.tar.gz`, `.tar.xz`, `.zip`, etc.
- Just use the extracted files in `package()`

### 3. Architecture-Specific Sources Are Built-In

**Standard feature:**
```bash
source_x86_64=(...)
source_aarch64=(...)
sha256sums_x86_64=(...)
sha256sums_aarch64=(...)
```

makepkg automatically selects the right source based on `uname -m`.

### 4. This Is NOT a Hack

**This is the official, documented, standard way to distribute binaries in AUR.**

From the Arch Wiki:
> "Files can also be supplied... and their names added to this array. Before the actual build process starts, all the files referenced in this array will be downloaded..."

It doesn't say "source code" - it says "files". Binaries are files!

---

## 🚀 Implications for rbee

### What This Means

1. **No need for a separate binary registry** (for MVP)
   - PKGBUILDs can handle binaries perfectly
   - Standard AUR pattern works out of the box

2. **Premium workers are straightforward**
   - Use `source_x86_64=()` with authenticated URLs
   - Or use custom DLAGENT
   - Or manual download

3. **Hybrid approach is still valuable**
   - But not because "PKGBUILDs can't handle binaries"
   - Because binary registry adds: analytics, versioning, discovery

4. **The Git catalog approach is even better**
   - It's EXACTLY what AUR does
   - Proven to work at scale
   - Handles both source and binary packages

### Revised Recommendation

**Phase 1: Git Catalog with PKGBUILDs (Week 1-2)**
- Create Git branches for each worker
- Add PKGBUILDs (source OR binary)
- This is 100% AUR-compatible
- Works for free AND premium workers

**Phase 2: Binary Registry (Optional, Week 3-4)**
- Add if you need:
  - Advanced analytics
  - Automatic version management
  - Web UI for browsing
  - License management API
- But it's not required for basic functionality

---

## 📊 Comparison: What We Thought vs Reality

| Aspect | What We Thought | Reality |
|--------|----------------|---------|
| **PKGBUILDs** | Only for source builds | Can handle binaries perfectly |
| **Premium Support** | Needs custom solution | Standard AUR pattern works |
| **Architecture** | Need separate handling | Built-in with `source_x86_64=()` |
| **Complexity** | Need binary registry | Git + PKGBUILDs is enough |
| **Is it a hack?** | Yes, feels hacky | No, it's the standard pattern |

---

## ✅ Conclusion

**The PKGBUILD approach is NOT a hack. It's the standard AUR pattern.**

AUR has been successfully distributing proprietary binaries for years using exactly this approach:
- Slack
- Zoom  
- VS Code
- Spotify
- Discord
- Chrome
- Dropbox
- And thousands more...

**For rbee workers:**
1. Start with Git catalog + PKGBUILDs (AUR pattern)
2. Add binary registry later if needed (for analytics, UI, etc.)
3. Both approaches are valid and complementary

**The hybrid approach is still good, but not because PKGBUILDs can't handle binaries - they can and do!**

---

**TEAM-402 - AUR Binary Pattern Documented!** 🎉
