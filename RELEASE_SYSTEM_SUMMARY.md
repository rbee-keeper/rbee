# Release System Summary

**Created by:** TEAM-451  
**Rule Zero Applied:** Consolidated from 11 documents → 1 guide + implementation plans

---

## 🎯 What Needs to Happen

### 1. **Implement `cargo xtask release`** (Interactive Version Manager)

**What:** Rust-based interactive CLI that bumps versions across tiers

**Why:** 
- Type-safe (catches errors at compile time)
- Interactive (asks patch/minor/major)
- Cross-platform (works on mac, blep, workstation)
- Already in your stack (xtask)

**How:**
```rust
// xtask/src/release/
├── cli.rs          // Interactive prompts (inquire)
├── tiers.rs        // Load .version-tiers/*.toml
├── bump_rust.rs    // Update Cargo.toml files
├── bump_js.rs      // Update package.json files
└── git.rs          // Commit changes
```

**Status:** Not implemented yet (detailed plan in `.docs/RELEASE_XTASK_IMPLEMENTATION_PLAN.md`)

---

### 2. **Setup Build Machines**

**mac (macOS ARM64):**
- Install pnpm
- Setup GitHub Actions self-hosted runner
- Builds macOS binaries with Metal workers

**workstation (Ubuntu):**
- Install Rust + pnpm
- Builds Linux binaries for install.rbee.dev

**blep (Arch Linux):**
- Already ready!
- Updates AUR packages

**Status:** Machines exist, need setup (instructions in `.docs/MULTI_PLATFORM_BUILD_PLAN.md`)

---

### 3. **Configure Branch Protection**

**What:** Protect production branch, allow free pushing to development

**How:**
```bash
./scripts/configure-branch-protection.sh
```

**Result:**
- development: Free pushing ✅
- production: Requires PR + CI + approval ✅

**Status:** Script ready, not run yet

---

### 4. **Create Tier Configurations**

**What:** Define which crates/packages belong to each tier

**Files:**
```
.version-tiers/
├── main.toml        # rbee-keeper, queen, hive
├── llm-worker.toml  # LLM worker
├── sd-worker.toml   # SD worker
└── commercial.toml  # Frontend apps
```

**Status:** Not created yet (templates in implementation plan)

---

### 5. **Setup CI/CD Workflows**

**What:** GitHub Actions that build and publish on merge to production

**Files:**
- `.github/workflows/production-release.yml` - Linux build + release
- `.github/workflows/build-mac.yml` - macOS build (self-hosted)
- `.github/workflows/publish-aur.yml` - AUR publish (optional)

**Status:** Workflow files created, not tested

---

## 🔄 The Workflow (Once Implemented)

```bash
# 1. Develop on development branch
git checkout development
# ... make changes ...
git push  # Free pushing!

# 2. Bump version (interactive)
cargo xtask release
# ? Which tier? main
# ? Bump type? minor
# ✅ Version bumped to 0.4.0

# 3. Create release PR
gh pr create --base production --head development --title "Release v0.4.0"

# 4. Merge → Automatic publishing
# - Linux binaries built (GitHub Actions)
# - macOS binaries built (mac runner)
# - GitHub Release created
# - Binaries uploaded

# 5. Manual: Update AUR
./scripts/update-aur.sh 0.4.0

# 6. Users install
curl -fsSL https://install.rbee.dev | sh  # macOS/Linux
paru -S rbee                              # Arch
```

---

## 📦 What Gets Published

| Platform | Method | Builder | Output |
|----------|--------|---------|--------|
| **macOS ARM64** | install.rbee.dev | mac (self-hosted) | rbee-macos-arm64.tar.gz |
| **macOS x86_64** | install.rbee.dev | GitHub Actions | rbee-macos-x86_64.tar.gz |
| **Linux x86_64** | install.rbee.dev | GitHub Actions | rbee-linux-x86_64.tar.gz |
| **Arch Linux** | AUR | Manual (blep) | PKGBUILD |

---

## 🎯 Version Strategy (Hybrid)

**Tier 1: Main** (Synchronized)
- rbee-keeper, queen-rbee, rbee-hive → Same version (e.g., v0.4.0)
- All shared crates → Same version
- User installs: "rbee v0.4.0"

**Tier 2: Workers** (Independent)
- llm-worker-rbee → v1.4.0
- sd-worker-rbee → v0.6.0
- Hive installs workers separately

**Tier 3: Frontend** (Independent)
- @rbee/commercial → v2.1.0
- Marketing updates don't need daemon releases

---

## 📋 Implementation Phases

### Phase 1: xtask Implementation ⏳
- [ ] Add dependencies to xtask/Cargo.toml
- [ ] Create tier config files
- [ ] Implement release module
- [ ] Test on blep

**Estimated:** 4-6 hours of coding

### Phase 2: Build Machine Setup ⏳
- [ ] Setup mac (install pnpm, GitHub runner)
- [ ] Setup workstation (install Rust, pnpm)
- [ ] Setup AUR (SSH keys, test PKGBUILD)

**Estimated:** 2-3 hours of setup

### Phase 3: Testing & Deployment ⏳
- [ ] Test version bump (dry-run)
- [ ] Test actual release (patch version)
- [ ] Test builds on all platforms
- [ ] Verify installations work

**Estimated:** 2-3 hours of testing

**Total:** ~8-12 hours to fully implement

---

## 📚 Documentation Structure (Rule Zero Applied)

### Main Documents (Keep)
1. **`RELEASE_GUIDE.md`** - Quick start + overview (NEW - consolidated)
2. **`RELEASE_XTASK_IMPLEMENTATION_PLAN.md`** - Detailed implementation
3. **`MULTI_PLATFORM_BUILD_PLAN.md`** - Build targets
4. **`VERSION_MANAGEMENT_PLAN.md`** - Strategy explanation
5. **`REPOSITORY_PROTECTION.md`** - Security setup
6. **`PUBLISHING_GUIDE.md`** - Publishing details

### Deleted (Rule Zero - Redundant)
- ❌ `VERSION_MANAGEMENT.md` - Redundant with PLAN
- ❌ `VERSION_TIERS_DIAGRAM.md` - Merged into PLAN
- ❌ `RELEASE_WORKFLOW.md` - Merged into GUIDE
- ❌ `QUICK_PROTECTION_SETUP.md` - Merged into PROTECTION
- ❌ `WORKFLOW_QUICKSTART.md` - Merged into GUIDE
- ❌ `RELEASE_SETUP_COMPLETE.md` - Redundant summary

**Result:** 11 documents → 6 focused documents

---

## 🚀 Next Steps (Priority Order)

1. **Implement xtask release** (Phase 1)
   - Most critical
   - Blocks everything else
   - ~4-6 hours

2. **Setup mac** (Phase 2a)
   - Install pnpm
   - Setup GitHub runner
   - ~1 hour

3. **Test workflow** (Phase 3)
   - Dry-run version bump
   - Test builds
   - ~2 hours

4. **Setup workstation** (Phase 2b)
   - Install Rust, pnpm
   - ~30 minutes

5. **Setup AUR** (Phase 2c)
   - SSH keys
   - Test PKGBUILD
   - ~1 hour

6. **First release** (Phase 3)
   - Real release (patch version)
   - Verify everything works
   - ~1 hour

---

## ✅ What's Ready Now

- ✅ Branch protection script
- ✅ Install script (install.sh)
- ✅ GitHub workflows (not tested)
- ✅ Documentation (consolidated)
- ✅ Implementation plan (detailed)
- ✅ Build scripts (templates)

## ⏳ What's Not Ready

- ❌ xtask release module (not implemented)
- ❌ Tier config files (not created)
- ❌ Build machines (not setup)
- ❌ CI/CD (not tested)
- ❌ AUR (not setup)

---

## 🎯 Decision Point

**Ready to implement?**

**Option A:** I implement Phase 1 (xtask) now (~4-6 hours of my work)  
**Option B:** You want to review/modify the plan first  
**Option C:** Start with machine setup (Phase 2) first  

**Recommendation:** Option A - Implement xtask first, then setup machines

---

**Status:** Planning Complete ✅  
**Next:** Implementation Phase 1 (xtask release module)  
**Blocker:** None - ready to proceed
