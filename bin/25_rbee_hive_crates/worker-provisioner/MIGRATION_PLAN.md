# Worker Provisioner Migration Plan

**Date:** 2025-11-04  
**Team:** TEAM-402  
**Status:** 🚧 IN PROGRESS

---

## 🎯 Goal

Migrate worker installation logic from `rbee-hive` to a dedicated `worker-provisioner` crate following the `model-provisioner` pattern, then enhance with proper AUR binary support.

---

## 📦 Files to Migrate

### From `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/`

1. **`pkgbuild_parser.rs`** (673 lines)
   - → `worker-provisioner/src/pkgbuild/parser.rs`
   - Parses PKGBUILD files
   - Already has comprehensive tests

2. **`pkgbuild_executor.rs`** (908 lines)
   - → `worker-provisioner/src/pkgbuild/executor.rs`
   - Executes build() and package() functions
   - Has cancellation support (TEAM-388)
   - Already has comprehensive tests

3. **`source_fetcher.rs`** (224 lines)
   - → `worker-provisioner/src/pkgbuild/source_fetcher.rs`
   - Fetches sources (git clone, etc.)
   - Has tests

4. **`worker_install.rs`** (460 lines)
   - → `worker-provisioner/src/provisioner.rs`
   - Main orchestration logic
   - Will be refactored to follow `ArtifactProvisioner` trait

---

## 🏗️ New Structure

```
worker-provisioner/
├── Cargo.toml
├── README.md
├── MIGRATION_PLAN.md (this file)
├── AUR_ENHANCEMENTS.md (new features)
└── src/
    ├── lib.rs
    ├── provisioner.rs (main WorkerProvisioner)
    ├── catalog_client.rs (HTTP client for worker catalog)
    ├── pkgbuild/
    │   ├── mod.rs
    │   ├── parser.rs (migrated from pkgbuild_parser.rs)
    │   ├── executor.rs (migrated from pkgbuild_executor.rs)
    │   └── source_fetcher.rs (migrated from source_fetcher.rs)
    └── aur/
        ├── mod.rs
        ├── binary_handler.rs (NEW - handles binary-only PKGBUILDs)
        └── arch_detector.rs (NEW - detects source_x86_64 vs source_aarch64)
```

---

## 🔄 Migration Steps

### Phase 1: Create Crate Structure ✅
- [x] Create `worker-provisioner/` directory
- [x] Create `Cargo.toml` with dependencies
- [x] Create migration plan (this file)

### Phase 2: Migrate Core Modules
- [ ] Create `src/lib.rs`
- [ ] Migrate `pkgbuild_parser.rs` → `src/pkgbuild/parser.rs`
- [ ] Migrate `pkgbuild_executor.rs` → `src/pkgbuild/executor.rs`
- [ ] Migrate `source_fetcher.rs` → `src/pkgbuild/source_fetcher.rs`
- [ ] Create `src/pkgbuild/mod.rs`

### Phase 3: Create Provisioner
- [ ] Create `src/catalog_client.rs` (extract from worker_install.rs)
- [ ] Create `src/provisioner.rs` (refactor worker_install.rs)
- [ ] Implement `ArtifactProvisioner<WorkerBinary>` trait
- [ ] Add cancellation support throughout

### Phase 4: Add AUR Binary Support
- [ ] Create `src/aur/mod.rs`
- [ ] Create `src/aur/arch_detector.rs`
  - Detect current architecture (x86_64, aarch64)
  - Select correct source array (source_x86_64 vs source_aarch64)
- [ ] Create `src/aur/binary_handler.rs`
  - Handle binary-only PKGBUILDs (no build() function)
  - Extract tarballs automatically
  - Skip build phase for binaries
- [ ] Update parser to support:
  - `source_x86_64=()`
  - `source_aarch64=()`
  - `sha256sums_x86_64=()`
  - `sha256sums_aarch64=()`
  - `noextract=()`
- [ ] Update executor to:
  - Skip build() if not present (binary packages)
  - Handle noextract files

### Phase 5: Update rbee-hive
- [ ] Remove old files from rbee-hive:
  - `src/pkgbuild_parser.rs`
  - `src/pkgbuild_executor.rs`
  - `src/source_fetcher.rs`
  - `src/worker_install.rs`
- [ ] Add `worker-provisioner` dependency to rbee-hive
- [ ] Update `src/handlers/worker_install.rs` to use new crate
- [ ] Update `src/lib.rs` to remove old module declarations

### Phase 6: Testing
- [ ] Run existing tests (migrated with code)
- [ ] Add new tests for AUR binary support
- [ ] Test with real PKGBUILDs:
  - Source build (llm-worker-rbee-cpu)
  - Binary package (future premium workers)
- [ ] Integration test with rbee-hive

---

## 🆕 AUR Enhancements

### Architecture-Specific Sources

**Before (current):**
```bash
source=("https://github.com/user/repo.git")
```

**After (AUR pattern):**
```bash
source_x86_64=("https://releases.rbee.ai/worker-x86_64.tar.gz")
source_aarch64=("https://releases.rbee.ai/worker-aarch64.tar.gz")
sha256sums_x86_64=('abc123...')
sha256sums_aarch64=('def456...')
```

### Binary-Only Packages

**Before:** Required build() function  
**After:** build() is optional for binary packages

```bash
pkgname=llm-worker-rbee-premium
source_x86_64=("https://releases.rbee.ai/premium.tar.gz")
sha256sums_x86_64=('abc123...')

# NO build() function!

package() {
    install -Dm755 "llm-worker-rbee-premium" "$pkgdir/usr/local/bin/$pkgname"
}
```

### No-Extract Support

**For files that shouldn't be auto-extracted:**
```bash
source=("app.tar.gz")
noextract=('app.tar.gz')

package() {
    # Manually extract in package()
    tar -xzf "$srcdir/app.tar.gz" -C "$pkgdir/usr/local/bin/"
}
```

---

## 📊 Code Statistics

### Before (in rbee-hive)
- `pkgbuild_parser.rs`: 673 lines
- `pkgbuild_executor.rs`: 908 lines
- `source_fetcher.rs`: 224 lines
- `worker_install.rs`: 460 lines
- **Total:** 2,265 lines

### After (in worker-provisioner)
- Core modules: ~2,265 lines (migrated)
- New AUR support: ~300 lines (estimated)
- Provisioner refactoring: ~200 lines
- **Total:** ~2,765 lines

---

## ✅ Success Criteria

1. **All existing tests pass** after migration
2. **Source builds work** (llm-worker-rbee-cpu)
3. **Binary packages work** (new AUR pattern)
4. **Architecture detection works** (x86_64 vs aarch64)
5. **Cancellation works** (TEAM-388 requirement)
6. **rbee-hive compiles** with new dependency
7. **No functionality lost** in migration

---

## 🚨 Breaking Changes

### None Expected

This is a **pure refactoring** - moving code to a better location. The API should remain the same for rbee-hive.

---

## 📝 Notes

- Keep all TEAM-XXX comments during migration
- Preserve all tests
- Add new tests for AUR features
- Update documentation as we go
- Follow model-provisioner pattern closely

---

**TEAM-402 - Migration Plan Complete!**
