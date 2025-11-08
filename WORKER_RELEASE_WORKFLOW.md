# Worker Release Workflow

**Purpose:** End-to-end workflow for releasing workers  
**When to use:** When you want to release a new worker or update an existing one

---

## 🎯 Complete Workflow

### Step 1: Build Worker Binaries

```bash
# Build on both machines (mac and blep)

# On blep (Linux x86_64)
cd ~/Projects/rbee
cargo build --release --package llm-worker-rbee --features cpu
cargo build --release --package llm-worker-rbee --features cuda
cargo build --release --package sd-worker-rbee --features cpu
cargo build --release --package sd-worker-rbee --features cuda

# On mac (macOS arm64)
ssh mac << 'EOF'
cd ~/Projects/rbee
cargo build --release --package llm-worker-rbee --features cpu
cargo build --release --package llm-worker-rbee --features metal
cargo build --release --package sd-worker-rbee --features cpu
EOF
```

### Step 2: Package Binaries

```bash
# On blep
cd target/release
tar -czf llm-worker-rbee-linux-x86_64-v0.1.0.tar.gz llm-worker-rbee
tar -czf sd-worker-rbee-linux-x86_64-v0.1.0.tar.gz sd-worker-rbee

# On mac
ssh mac << 'EOF'
cd ~/Projects/rbee/target/release
tar -czf llm-worker-rbee-macos-arm64-v0.1.0.tar.gz llm-worker-rbee
tar -czf sd-worker-rbee-macos-arm64-v0.1.0.tar.gz sd-worker-rbee
EOF
```

### Step 3: Create GitHub Release

```bash
# Create release
gh release create v0.1.0 \
  --title "Release v0.1.0" \
  --notes "Worker binaries for v0.1.0"

# Upload binaries
gh release upload v0.1.0 \
  target/release/llm-worker-rbee-linux-x86_64-v0.1.0.tar.gz \
  target/release/sd-worker-rbee-linux-x86_64-v0.1.0.tar.gz

# Upload mac binaries (from mac)
ssh mac << 'EOF'
cd ~/Projects/rbee
gh release upload v0.1.0 \
  target/release/llm-worker-rbee-macos-arm64-v0.1.0.tar.gz \
  target/release/sd-worker-rbee-macos-arm64-v0.1.0.tar.gz
EOF
```

### Step 4: Update Worker Catalog (GWC)

**Current (Manual - Hardcoded):**
```bash
# Edit src/data.ts manually
vim bin/80-hono-worker-catalog/src/data.ts
# Change version: "0.1.0" -> "0.2.0"
# Commit and deploy
```

**Future (Automatic - Dynamic):**
```bash
# Worker catalog reads version from Cargo.toml automatically
# No manual updates needed!
# Just deploy:
cargo xtask deploy --app worker
```

### Step 5: Deploy Worker Catalog

```bash
# Deploy to gwc.rbee.dev
cargo xtask deploy --app worker
```

---

## 🔄 Current Problems

### Problem 1: Manual Version Updates
```typescript
// src/data.ts - HARDCODED!
{
  id: "llm-worker-rbee-cpu",
  version: "0.1.0",  // ← Must manually update!
  // ...
}
```

**Impact:**
- ❌ Easy to forget to update
- ❌ Version mismatches between binary and catalog
- ❌ Manual work for every release

### Problem 2: PKGBUILD Versions Not Updated
```bash
# public/pkgbuilds/arch/prod/llm-worker-rbee-cpu.PKGBUILD
pkgver=0.1.0  # ← Hardcoded! Not updated by version bump!
```

**Impact:**
- ❌ PKGBUILDs point to wrong version
- ❌ Users download old binaries
- ❌ 16 files to manually update

### Problem 3: No Automation
```bash
# Current workflow (manual):
1. Build binaries (manual)
2. Package binaries (manual)
3. Create GitHub release (manual)
4. Upload binaries (manual)
5. Update worker catalog (manual)
6. Update PKGBUILDs (manual - often forgotten!)
7. Deploy worker catalog (manual)
```

**Impact:**
- ❌ Error-prone
- ❌ Time-consuming
- ❌ Easy to miss steps

---

## ✅ Solution: Automated Worker Release

### Architecture

```
cargo xtask release-worker --worker llm-worker --version 0.2.0
  ↓
1. Bump version in Cargo.toml
  ↓
2. Build binaries (mac + blep)
  ↓
3. Package binaries
  ↓
4. Create GitHub release
  ↓
5. Upload binaries
  ↓
6. Worker catalog auto-discovers new version (reads Cargo.toml)
  ↓
7. Deploy worker catalog
  ↓
Done!
```

### Implementation Plan

**1. Add `release-worker` command to xtask:**
```rust
// xtask/src/cli.rs
Commands::ReleaseWorker {
    worker: String,
    version: String,
} => {
    // 1. Bump version in Cargo.toml
    bump_worker_version(&worker, &version)?;
    
    // 2. Build on both machines
    build_worker_binaries(&worker)?;
    
    // 3. Package binaries
    package_worker_binaries(&worker, &version)?;
    
    // 4. Create GitHub release
    create_github_release(&version)?;
    
    // 5. Upload binaries
    upload_worker_binaries(&worker, &version)?;
    
    // 6. Deploy worker catalog (auto-discovers new version)
    deploy_worker_catalog()?;
}
```

**2. Worker catalog reads version dynamically:**
```typescript
// src/discovery.ts
async function loadWorkerMetadata(dirName: string): Promise<WorkerMetadata> {
  // Read Cargo.toml
  const cargoToml = await readFile(`bin/${dirName}/Cargo.toml`);
  const version = parseCargoVersion(cargoToml);  // ← Always current!
  
  // Read worker.toml
  const workerToml = await readFile(`bin/${dirName}/worker.toml`);
  const metadata = parseToml(workerToml);
  
  return { ...metadata, version };  // ← Version from Cargo.toml!
}
```

**3. PKGBUILDs generated with current version:**
```typescript
// src/routes.ts
routes.get("/workers/:id/PKGBUILD/:platform/:build", async (c) => {
  const worker = await findWorker(id);
  
  // Generate PKGBUILD with current version from Cargo.toml
  const pkgbuild = generatePKGBUILD(template, worker);
  
  return new Response(pkgbuild, {
    headers: { "Content-Type": "text/plain" }
  });
});

function generatePKGBUILD(template: string, worker: WorkerCatalogEntry): string {
  return template
    .replace('{{VERSION}}', worker.version)  // ← From Cargo.toml!
    .replace('{{DOWNLOAD_URL}}', `https://github.com/rbee-keeper/rbee/releases/download/v${worker.version}/...`)
    // ...
}
```

---

## 🚀 Future Workflow (Automated)

### One Command Release

```bash
# Release a worker (everything automated!)
cargo xtask release-worker --worker llm-worker --version 0.2.0
```

**What it does:**
1. ✅ Bumps version in `bin/30_llm_worker_rbee/Cargo.toml`
2. ✅ Builds binaries on mac and blep (SSH)
3. ✅ Packages binaries as tarballs
4. ✅ Creates GitHub release v0.2.0
5. ✅ Uploads all binaries to release
6. ✅ Deploys worker catalog (auto-discovers new version)
7. ✅ PKGBUILDs auto-generate with new version

**Result:**
- ✅ Worker catalog shows v0.2.0
- ✅ PKGBUILDs download v0.2.0 binaries
- ✅ GitHub release has all binaries
- ✅ No manual steps!

### Adding a New Worker

```bash
# 1. Create worker directory
mkdir bin/32_audio_worker_rbee
cd bin/32_audio_worker_rbee

# 2. Create Cargo.toml
cat > Cargo.toml << 'EOF'
[package]
name = "audio-worker-rbee"
version = "0.1.0"
# ...
EOF

# 3. Create worker.toml
cat > worker.toml << 'EOF'
[worker]
id = "audio-worker"
name = "Audio Worker"
description = "Audio processing worker"
# ...
EOF

# 4. Implement worker
# ... write code ...

# 5. Release it!
cargo xtask release-worker --worker audio-worker --version 0.1.0
```

**That's it!** Worker automatically:
- ✅ Appears in worker catalog
- ✅ Has PKGBUILDs generated
- ✅ Available for download
- ✅ No code changes to catalog needed

---

## 📋 Implementation Checklist

### Phase 1: Dynamic Worker Catalog
- [ ] Create `worker.toml` schema
- [ ] Add `worker.toml` to existing workers
- [ ] Implement worker discovery service
- [ ] Read version from `Cargo.toml`
- [ ] Update API to use discovery
- [ ] Deploy dynamic catalog

### Phase 2: PKGBUILD Generation
- [ ] Create PKGBUILD templates
- [ ] Implement on-the-fly generation
- [ ] Generate with version from `Cargo.toml`
- [ ] Test with existing workers

### Phase 3: Automated Release
- [ ] Add `release-worker` command to xtask
- [ ] Implement version bumping
- [ ] Implement binary building (SSH to mac)
- [ ] Implement packaging
- [ ] Implement GitHub release creation
- [ ] Implement binary upload
- [ ] Implement catalog deployment

### Phase 4: Integration
- [ ] Test complete workflow
- [ ] Update documentation
- [ ] Train team on new workflow

---

## 🎯 Success Criteria

1. ✅ One command releases a worker
2. ✅ Worker catalog auto-discovers new versions
3. ✅ PKGBUILDs always have current version
4. ✅ No manual version updates needed
5. ✅ Adding new worker requires 0 catalog changes

---

## 📊 Before vs After

### Before (Current - Manual)
```bash
# 7 manual steps, ~30 minutes, error-prone
1. Build on blep (manual)
2. Build on mac (manual SSH)
3. Package binaries (manual)
4. Create release (manual)
5. Upload binaries (manual)
6. Edit src/data.ts (manual - often forgotten!)
7. Edit 16 PKGBUILDs (manual - often forgotten!)
8. Deploy catalog (manual)
```

### After (Future - Automated)
```bash
# 1 command, ~5 minutes, automated
cargo xtask release-worker --worker llm-worker --version 0.2.0

# Everything else happens automatically!
```

---

## 🔗 Related Documents

- `bin/80-hono-worker-catalog/DYNAMIC_WORKER_CATALOG_PLAN.md` - Dynamic catalog architecture
- `bin/00_rbee_keeper/GITHUB_RELEASES_INSTALL_PLAN.md` - GitHub releases support
- `MANUAL_RELEASE_GUIDE.md` - General release workflow
- `CLOUDFLARE_DEPLOY_USAGE.md` - Deployment commands

---

**Next:** Implement dynamic worker catalog, then automate release workflow!
