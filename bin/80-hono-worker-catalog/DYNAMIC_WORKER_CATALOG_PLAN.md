# Dynamic Worker Catalog Plan

**Created by:** TEAM-451  
**Problem:** Worker catalog is completely hardcoded - not scalable for marketplace/AUR-like system

---

## 🚨 Current Problems

### 1. Hardcoded Worker Data (`src/data.ts`)
```typescript
export const WORKERS: WorkerCatalogEntry[] = [
  { id: "llm-worker-rbee-cpu", ... },    // ← Hardcoded!
  { id: "llm-worker-rbee-cuda", ... },   // ← Hardcoded!
  { id: "sd-worker-rbee-cpu", ... },     // ← Hardcoded!
  // ... 8 total hardcoded workers
];
```

**Problems:**
- ❌ Adding a new worker requires code changes
- ❌ Version is hardcoded (`version: "0.1.0"`)
- ❌ Not scalable for marketplace
- ❌ Can't have community-contributed workers

### 2. Hardcoded PKGBUILDs (16 files!)
```
public/pkgbuilds/
├── arch/prod/
│   ├── llm-worker-rbee-cpu.PKGBUILD     ← pkgver=0.1.0 hardcoded!
│   ├── llm-worker-rbee-cuda.PKGBUILD    ← pkgver=0.1.0 hardcoded!
│   └── ... (5 total)
├── arch/dev/
│   └── ... (5 total)
├── homebrew/prod/
│   └── ... (3 total)
└── homebrew/dev/
    └── ... (3 total)
```

**Problems:**
- ❌ Version bumps don't update PKGBUILDs automatically
- ❌ 16 files to manually update for each version
- ❌ Easy to have version mismatches

### 3. Hardcoded Tests
```typescript
// data.test.ts
const requiredWorkers = [
  'llm-worker-rbee-cpu',     // ← Hardcoded!
  'llm-worker-rbee-cuda',    // ← Hardcoded!
  // ...
];
```

**Problems:**
- ❌ Tests break when adding new workers
- ❌ Not scalable

### 4. Hardcoded Deployment Gates
```rust
// xtask/src/deploy/gates.rs
let arch_prod_files = vec![
    "llm-worker-rbee-cpu.PKGBUILD",    // ← Hardcoded!
    "llm-worker-rbee-cuda.PKGBUILD",   // ← Hardcoded!
    // ...
];
```

**Problems:**
- ❌ Gates break when adding new workers
- ❌ Not scalable

---

## ✅ Solution: Dynamic Worker Discovery

### Architecture: AUR-Like System

```
bin/
├── 30_llm_worker_rbee/
│   ├── Cargo.toml              ← Read version from here
│   └── worker.toml             ← NEW: Worker metadata
├── 31_sd_worker_rbee/
│   ├── Cargo.toml
│   └── worker.toml
└── 32_audio_worker_rbee/       ← Future worker
    ├── Cargo.toml
    └── worker.toml
```

**`worker.toml` format:**
```toml
[worker]
id = "llm-worker"
name = "LLM Worker"
description = "Candle-based LLM inference worker"
license = "GPL-3.0-or-later"

[build]
features = ["cpu", "cuda", "metal", "rocm"]
default_feature = "cpu"

[capabilities]
supported_formats = ["gguf", "safetensors"]
max_context_length = 32768
supports_streaming = true
supports_batching = false

[platforms]
linux = ["x86_64", "aarch64"]
macos = ["aarch64"]
windows = ["x86_64"]
```

---

## 📋 Implementation Plan

### Phase 1: Dynamic Worker Discovery (Backend)

**1. Create worker discovery service:**
```typescript
// src/discovery.ts
export async function discoverWorkers(): Promise<WorkerCatalogEntry[]> {
  const workers: WorkerCatalogEntry[] = [];
  
  // Scan bin/ directory for worker crates
  const binDir = '../../../bin';
  const entries = await readdir(binDir);
  
  for (const entry of entries) {
    if (entry.includes('worker')) {
      const worker = await loadWorkerMetadata(entry);
      if (worker) {
        workers.push(...generateWorkerVariants(worker));
      }
    }
  }
  
  return workers;
}

async function loadWorkerMetadata(dirName: string): Promise<WorkerMetadata | null> {
  // Read worker.toml
  const workerToml = await readFile(`bin/${dirName}/worker.toml`);
  const metadata = parseToml(workerToml);
  
  // Read version from Cargo.toml
  const cargoToml = await readFile(`bin/${dirName}/Cargo.toml`);
  const version = parseCargoVersion(cargoToml);
  
  return { ...metadata, version };
}

function generateWorkerVariants(worker: WorkerMetadata): WorkerCatalogEntry[] {
  const variants: WorkerCatalogEntry[] = [];
  
  // Generate variant for each feature (cpu, cuda, metal, rocm)
  for (const feature of worker.build.features) {
    variants.push({
      id: `${worker.id}-${feature}`,
      version: worker.version,  // ← From Cargo.toml!
      workerType: feature,
      // ... rest from worker.toml
    });
  }
  
  return variants;
}
```

**2. Update API routes:**
```typescript
// src/routes.ts
routes.get("/workers", async (c) => {
  // Discover workers dynamically
  const workers = await discoverWorkers();
  return c.json({ workers });
});
```

**Benefits:**
- ✅ Workers discovered automatically
- ✅ Version read from Cargo.toml
- ✅ No hardcoded worker list
- ✅ Add new worker = just create directory

### Phase 2: Dynamic PKGBUILD Generation

**1. PKGBUILD templates:**
```
public/pkgbuilds/templates/
├── arch-prod.template
├── arch-dev.template
├── homebrew-prod.template
└── homebrew-dev.template
```

**2. Generate PKGBUILDs on-the-fly:**
```typescript
routes.get("/workers/:id/PKGBUILD/:platform/:build", async (c) => {
  const { id, platform, build } = c.req.param();
  
  // Discover worker
  const worker = await findWorker(id);
  
  // Load template
  const template = await loadTemplate(platform, build);
  
  // Generate PKGBUILD with current version
  const pkgbuild = generatePKGBUILD(template, worker);
  
  return new Response(pkgbuild, {
    headers: { "Content-Type": "text/plain" }
  });
});

function generatePKGBUILD(template: string, worker: WorkerCatalogEntry): string {
  return template
    .replace('{{PKGNAME}}', worker.binaryName)
    .replace('{{VERSION}}', worker.version)  // ← Always current!
    .replace('{{DESCRIPTION}}', worker.description)
    .replace('{{FEATURES}}', worker.build.features.join(','))
    .replace('{{DEPENDS}}', worker.depends.join(' '))
    // ... etc
}
```

**Benefits:**
- ✅ PKGBUILDs always have current version
- ✅ No manual updates needed
- ✅ Version bumps automatically propagate

### Phase 3: Dynamic Tests

**1. Update data.test.ts:**
```typescript
describe('Worker Catalog Data', () => {
  it('should discover all workers from bin/ directory', async () => {
    const workers = await discoverWorkers();
    
    // Test that workers were discovered
    expect(workers.length).toBeGreaterThan(0);
    
    // Test that all workers have required fields
    for (const worker of workers) {
      expect(worker).toHaveProperty('id');
      expect(worker).toHaveProperty('version');
      // ...
    }
  });
  
  it('should have matching versions in Cargo.toml', async () => {
    const workers = await discoverWorkers();
    
    for (const worker of workers) {
      const cargoVersion = await readCargoVersion(worker.source.path);
      expect(worker.version).toBe(cargoVersion);
    }
  });
});
```

**Benefits:**
- ✅ Tests work for any number of workers
- ✅ No hardcoded worker lists

### Phase 4: Dynamic Deployment Gates

**1. Update gates.rs:**
```rust
fn validate_pkgbuilds() -> Result<()> {
    // Discover workers dynamically
    let workers = discover_workers_from_bin()?;
    
    // For each worker, check PKGBUILDs exist
    for worker in workers {
        for platform in &["arch", "homebrew"] {
            for build in &["prod", "dev"] {
                // Check if PKGBUILD can be generated
                validate_pkgbuild_template(platform, build, &worker)?;
            }
        }
    }
    
    Ok(())
}
```

**Benefits:**
- ✅ Gates work for any number of workers
- ✅ No hardcoded file lists

---

## 🎯 Migration Path

### Step 1: Add worker.toml to existing workers
```bash
# For each worker
cd bin/30_llm_worker_rbee
cat > worker.toml << 'EOF'
[worker]
id = "llm-worker"
name = "LLM Worker"
# ...
EOF
```

### Step 2: Implement discovery service
- Create `src/discovery.ts`
- Test with existing workers

### Step 3: Update API routes
- Switch from hardcoded `WORKERS` array to `discoverWorkers()`
- Test endpoints

### Step 4: Create PKGBUILD templates
- Extract common patterns from existing PKGBUILDs
- Create templates

### Step 5: Implement PKGBUILD generation
- Generate on-the-fly from templates
- Test with existing workers

### Step 6: Update tests
- Remove hardcoded worker lists
- Use dynamic discovery

### Step 7: Update deployment gates
- Remove hardcoded file lists
- Use dynamic discovery

### Step 8: Delete hardcoded files
- Delete `src/data.ts`
- Delete static PKGBUILDs (keep templates)

---

## 🚀 Future: Marketplace Support

Once dynamic discovery is working:

### Community Workers
```
bin/
├── 30_llm_worker_rbee/        ← Official
├── 31_sd_worker_rbee/         ← Official
└── community/
    ├── audio_worker/          ← Community
    ├── video_worker/          ← Community
    └── tts_worker/            ← Community
```

### Worker Registry
```typescript
// Workers can be:
// 1. Local (in bin/)
// 2. Remote (GitHub repos)
// 3. Community (submitted via PR)

interface WorkerSource {
  type: 'local' | 'remote' | 'community';
  location: string;
  verified: boolean;
}
```

### AUR-Like Features
- ✅ Anyone can submit a worker
- ✅ Community voting/ratings
- ✅ Automated testing
- ✅ Version tracking
- ✅ Dependency resolution

---

## 📊 Benefits Summary

**Before (Hardcoded):**
- ❌ 8 workers hardcoded in data.ts
- ❌ 16 PKGBUILD files to manually update
- ❌ Version bumps don't propagate
- ❌ Adding worker = 20+ file changes
- ❌ Not scalable

**After (Dynamic):**
- ✅ Infinite workers supported
- ✅ 0 PKGBUILD files (generated on-the-fly)
- ✅ Version bumps automatic
- ✅ Adding worker = create 1 directory
- ✅ Marketplace-ready

---

## 🎯 Success Criteria

1. ✅ Worker catalog discovers workers from bin/
2. ✅ Version read from Cargo.toml automatically
3. ✅ PKGBUILDs generated on-the-fly with current version
4. ✅ Tests work for any number of workers
5. ✅ Deployment gates work for any number of workers
6. ✅ Adding new worker requires 0 code changes

---

## 📝 Next Steps

1. Create `worker.toml` schema
2. Add `worker.toml` to existing workers
3. Implement discovery service
4. Create PKGBUILD templates
5. Implement PKGBUILD generation
6. Update tests
7. Update deployment gates
8. Delete hardcoded files

**Estimated effort:** 1-2 days  
**Impact:** Transforms static catalog into dynamic marketplace
