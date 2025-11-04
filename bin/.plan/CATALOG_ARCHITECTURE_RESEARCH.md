# Complete Catalog Architecture Research

**Date:** 2025-11-04  
**Status:** 📋 RESEARCH COMPLETE  
**Purpose:** Understand all catalog/provisioner crates before worker spawning implementation

---

## 🏗️ Current Architecture Overview

### Catalog Hierarchy

```
artifact-catalog (base abstraction)
    ├── Artifact trait
    ├── ArtifactCatalog trait
    ├── FilesystemCatalog<T>
    ├── VendorSource trait
    └── ArtifactProvisioner trait
    
model-catalog (LLM models)
    ├── ModelEntry implements Artifact
    ├── ModelCatalog wraps FilesystemCatalog<ModelEntry>
    └── Uses artifact-catalog abstractions
    
worker-catalog (Worker binaries)
    ├── WorkerBinary implements Artifact
    ├── WorkerCatalog wraps FilesystemCatalog<WorkerBinary>
    ├── WorkerType enum (CpuLlm, CudaLlm, MetalLlm)
    └── Platform enum (Linux, MacOS, Windows)
```

---

## 📦 Crate 1: artifact-catalog

**Location:** `bin/25_rbee_hive_crates/artifact-catalog/`  
**Purpose:** Shared abstraction for all catalogs  
**Status:** ✅ COMPLETE

### Core Types

**Artifact trait:**
```rust
pub trait Artifact: Clone + Serialize + for<'de> Deserialize<'de> {
    fn id(&self) -> &str;
    fn path(&self) -> &Path;
    fn size(&self) -> u64;
    fn status(&self) -> &ArtifactStatus;
    fn set_status(&mut self, status: ArtifactStatus);
    fn name(&self) -> &str { self.id() }
}
```

**ArtifactStatus enum:**
```rust
pub enum ArtifactStatus {
    Available,
    Downloading,
    Failed { error: String },
}
```

**ArtifactCatalog trait:**
```rust
pub trait ArtifactCatalog<T: Artifact> {
    fn add(&self, artifact: T) -> Result<()>;
    fn get(&self, id: &str) -> Result<T>;
    fn list(&self) -> Vec<T>;
    fn remove(&self, id: &str) -> Result<()>;
    fn contains(&self, id: &str) -> bool;
    fn len(&self) -> usize;
}
```

**FilesystemCatalog<T>:**
- Stores artifacts as JSON files in directory
- Each artifact gets subdirectory with `metadata.json`
- Sanitizes IDs (replaces `/` and `:` with `-`)
- Generic over any type implementing `Artifact`

**VendorSource trait:**
```rust
#[async_trait]
pub trait VendorSource: Send + Sync {
    async fn download(&self, id: &str, dest: &Path, job_id: &str, cancel_token: CancellationToken) -> Result<u64>;
    fn supports(&self, id: &str) -> bool;
    fn name(&self) -> &str;
}
```

**ArtifactProvisioner trait:**
```rust
#[async_trait]
pub trait ArtifactProvisioner<T: Artifact>: Send + Sync {
    async fn provision(&self, id: &str, job_id: &str, cancel_token: CancellationToken) -> Result<T>;
    fn supports(&self, id: &str) -> bool;
}
```

### Key Insights

✅ **Well-designed abstraction** - Clean separation of concerns  
✅ **Generic over artifact type** - Reusable for models, workers, etc.  
✅ **Filesystem-based** - No database needed  
✅ **Cancellation support** - Built-in via CancellationToken  
✅ **Narration-aware** - job_id propagation for SSE routing

---

## 📦 Crate 2: model-catalog

**Location:** `bin/25_rbee_hive_crates/model-catalog/`  
**Purpose:** Catalog for LLM models (GGUF files)  
**Status:** ✅ COMPLETE (basic), ⚠️ MISSING metadata

### ModelEntry Structure

**Current:**
```rust
pub struct ModelEntry {
    id: String,           // e.g., "meta-llama/Llama-2-7b"
    name: String,         // Human-readable name
    path: PathBuf,        // Path to GGUF file
    size: u64,            // File size in bytes
    status: ArtifactStatus,
    added_at: DateTime<Utc>,
}
```

**Missing (for proper worker spawning):**
- ❌ `model_type` - LLM vs StableDiffusion
- ❌ `architecture` - llama, mistral, sdxl, etc.
- ❌ `quantization` - Q4_K_M, Q5_K_M, F16, etc.
- ❌ `parameter_count` - 7B, 13B, 70B, etc.
- ❌ `context_length` - 4096, 8192, 32768, etc.
- ❌ `gguf_metadata` - Parsed from GGUF header
- ❌ `download_progress` - For active downloads
- ❌ `checksum` - SHA256 for verification
- ❌ `vendor` - HuggingFace, CivitAI, local, etc.

### Storage Location

- Linux/Mac: `~/.cache/rbee/models/`
- Windows: `%LOCALAPPDATA%\rbee\models\`
- Structure: `models/{sanitized-id}/metadata.json`

### Key Insights

⚠️ **Needs enhancement** - Missing critical metadata for worker spawning  
✅ **Works for basic use** - Can store and retrieve models  
❌ **No model type distinction** - Can't tell LLM from SD models  
❌ **No compatibility checking** - Can't verify model works with worker

---

## 📦 Crate 3: model-provisioner

**Location:** `bin/25_rbee_hive_crates/model-provisioner/`  
**Purpose:** Download models from HuggingFace  
**Status:** ✅ COMPLETE (HF only), ❌ MISSING CivitAI

### Current Implementation

**HuggingFaceVendor:**
- Uses official `hf-hub` crate (same as Candle)
- Supports GGUF auto-detection (Q4_K_M, Q5_K_M, etc.)
- Real-time progress via `hf-hub-simple-progress`
- Cancellation support via CancellationToken
- Narration integration for SSE streaming

**Supported ID Formats:**
```
meta-llama/Llama-2-7b-chat-hf           # Auto-detect GGUF
TheBloke/Llama-2-7B-Chat-GGUF:model-Q4_K_M.gguf  # Explicit file
```

**ModelProvisioner:**
- Wraps HuggingFaceVendor
- Creates ModelEntry artifacts
- Handles download → catalog flow

### Missing Vendors

❌ **CivitAI Vendor** - For Stable Diffusion models  
❌ **GitHub Release Vendor** - For custom builds  
❌ **Local Build Vendor** - For development

### CivitAI API Research

✅ **API EXISTS!**

**Download URL Format:**
```
https://civitai.com/api/download/models/{modelVersionId}?token={api_token}
```

**Authentication:**
- Requires API token from user account
- Pass as query param `?token=xxx` or `Authorization: Bearer xxx`

**Model ID Format:**
```
civitai:123456                    # Model version ID
civitai:123456:filename.safetensors  # Explicit filename
```

**Implementation Plan:**
```rust
pub struct CivitAIVendor {
    api_token: String,
    client: reqwest::Client,
}

impl VendorSource for CivitAIVendor {
    async fn download(&self, id: &str, dest: &Path, job_id: &str, cancel_token: CancellationToken) -> Result<u64> {
        // Parse id: "civitai:123456" or "civitai:123456:filename.safetensors"
        let (model_id, filename) = parse_civitai_id(id)?;
        
        // Build download URL
        let url = format!("https://civitai.com/api/download/models/{}?token={}", model_id, self.api_token);
        
        // Stream download with progress tracking
        // Similar to HuggingFaceVendor implementation
    }
    
    fn supports(&self, id: &str) -> bool {
        id.starts_with("civitai:")
    }
    
    fn name(&self) -> &str {
        "CivitAI"
    }
}
```

### Key Insights

✅ **HuggingFace works great** - Production-ready for LLM models  
✅ **CivitAI is feasible** - API exists, straightforward implementation  
⚠️ **Need multi-vendor routing** - Different vendors for different model types  
❌ **No SD model support yet** - Blocked on CivitAI vendor

---

## 📦 Crate 4: model-preloader

**Location:** `bin/25_rbee_hive_crates/model-preloader/`  
**Purpose:** Pre-load models into RAM for faster worker startup  
**Status:** ⚠️ STUB ONLY

### Concept

**Problem:** Loading large GGUF from disk → VRAM is slow (5-10 seconds)  
**Solution:** Pre-load into RAM → VRAM is fast (1-2 seconds)

**Architecture:**
```
Model Pre-loader
    ↓
Load GGUF from disk into RAM (mmap or read)
    ↓
Keep in RAM cache (LRU eviction)
    ↓
When worker spawns, transfer from RAM → VRAM (fast!)
```

**Implementation Status:**
- ✅ Stub types defined
- ✅ API designed
- ❌ No actual implementation
- ❌ No cache management
- ❌ No mmap support

### Key Insights

⏸️ **Not needed for MVP** - Nice optimization, not critical  
⏸️ **Implement later** - After basic worker spawning works  
✅ **Good API design** - Ready to implement when needed

---

## 📦 Crate 5: worker-catalog

**Location:** `bin/25_rbee_hive_crates/worker-catalog/`  
**Purpose:** Catalog for worker binaries  
**Status:** ✅ COMPLETE (basic), ⚠️ READ ONLY

### WorkerBinary Structure

```rust
pub struct WorkerBinary {
    id: String,                    // "cpu-llm-worker-rbee-v0.1.0-linux"
    worker_type: WorkerType,       // CpuLlm, CudaLlm, MetalLlm
    platform: Platform,            // Linux, MacOS, Windows
    path: PathBuf,                 // Path to binary
    size: u64,                     // Binary size
    status: ArtifactStatus,
    version: String,               // "0.1.0"
    added_at: DateTime<Utc>,
}
```

### WorkerType Enum

**Current (LLM only):**
```rust
pub enum WorkerType {
    CpuLlm,
    CudaLlm,
    MetalLlm,
}
```

**Missing (SD workers):**
```rust
pub enum WorkerType {
    // LLM workers
    CpuLlm,
    CudaLlm,
    MetalLlm,
    
    // SD workers (MISSING!)
    CpuSd,      // ❌ Not defined
    CudaSd,     // ❌ Not defined
    MetalSd,    // ❌ Not defined
}
```

### Key Insights

⚠️ **READ ONLY from Hive** - Queen installs workers via SSH  
⚠️ **Missing SD worker types** - Only LLM workers defined  
✅ **Good structure** - Easy to extend for SD workers  
❌ **No worker provisioner** - Installation logic scattered

---

## 📦 Worker Installation (rbee-hive)

**Location:** `bin/20_rbee_hive/src/`  
**Files:** `pkgbuild_parser.rs`, `pkgbuild_executor.rs`, `source_fetcher.rs`, `worker_install.rs`  
**Status:** ✅ COMPLETE (for LLM workers)

### Current Flow

```
1. Fetch worker metadata from catalog
2. Check platform compatibility
3. Download PKGBUILD
4. Parse PKGBUILD (Arch Linux format)
5. Check dependencies
6. Fetch sources (git clone)
7. Execute build() function
8. Execute package() function
9. Install binary
10. Update capabilities
11. Cleanup temp files
```

### PKGBUILD Format

```bash
pkgname=llm-worker-rbee-cpu
pkgver=0.1.0
pkgrel=1
arch=('x86_64')
license=('GPL-3.0-or-later')
depends=()
makedepends=('rust' 'cargo')
source=('git+https://github.com/user/repo.git#tag=v0.1.0')

build() {
    cd "$srcdir/repo"
    cargo build --release --bin llm-worker-rbee-cpu --features cpu
}

package() {
    cd "$srcdir/repo"
    install -Dm755 target/release/llm-worker-rbee-cpu "$pkgdir/usr/local/bin/llm-worker-rbee-cpu"
}
```

### Key Insights

✅ **Works for LLM workers** - Production-ready  
✅ **PKGBUILD is flexible** - Can handle any build process  
✅ **Git source fetching** - Supports tags, branches, commits  
⚠️ **Scattered across hive** - Should be in worker-provisioner crate  
❌ **No SD worker PKGBUILDs** - Need to create them

---

## 🎯 Critical Gaps Identified

### 1. Model Type Distinction ❌ CRITICAL

**Problem:** ModelEntry doesn't distinguish LLM from SD models

**Impact:**
- Can't validate model compatibility with worker
- Can't filter models by type in UI
- Can't auto-select correct worker for model

**Solution:**
```rust
pub enum ModelType {
    Llm { architecture: String },  // llama, mistral, etc.
    StableDiffusion { version: SdVersion },  // v1.5, v2.1, xl, turbo
}

pub struct ModelEntry {
    // ... existing fields ...
    model_type: ModelType,  // NEW!
}
```

### 2. Worker Type Extension ❌ CRITICAL

**Problem:** WorkerType only has LLM workers

**Impact:**
- Can't spawn SD workers
- Can't catalog SD binaries
- Can't match SD models to SD workers

**Solution:**
```rust
pub enum WorkerType {
    // LLM workers
    CpuLlm,
    CudaLlm,
    MetalLlm,
    
    // SD workers (NEW!)
    CpuSd,
    CudaSd,
    MetalSd,
}
```

### 3. CivitAI Vendor ❌ CRITICAL

**Problem:** No way to download SD models

**Impact:**
- Users can't get SD models
- Manual download required
- No progress tracking

**Solution:**
- Implement `CivitAIVendor` in model-provisioner
- Add API token configuration
- Support SafeTensors and GGUF formats

### 4. Worker Provisioner Crate ⚠️ IMPORTANT

**Problem:** Worker installation logic scattered in rbee-hive

**Impact:**
- Hard to test
- Hard to reuse
- Violates separation of concerns

**Solution:**
- Create `bin/25_rbee_hive_crates/worker-provisioner/`
- Move PKGBUILD logic there
- Implement `WorkerProvisioner` trait

### 5. Model-Worker Compatibility ⚠️ IMPORTANT

**Problem:** No validation that model works with worker

**Impact:**
- Can spawn LLM worker with SD model (fails at runtime)
- Can spawn SD worker with LLM model (fails at runtime)
- Poor user experience

**Solution:**
```rust
impl ModelEntry {
    pub fn compatible_worker_types(&self) -> Vec<WorkerType> {
        match &self.model_type {
            ModelType::Llm { .. } => vec![
                WorkerType::CpuLlm,
                WorkerType::CudaLlm,
                WorkerType::MetalLlm,
            ],
            ModelType::StableDiffusion { .. } => vec![
                WorkerType::CpuSd,
                WorkerType::CudaSd,
                WorkerType::MetalSd,
            ],
        }
    }
}
```

---

## 📊 Architecture Recommendations

### Recommendation 1: Extend ModelEntry ✅ DO THIS

**Add to ModelEntry:**
```rust
pub struct ModelEntry {
    // Existing fields
    id: String,
    name: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
    added_at: DateTime<Utc>,
    
    // NEW FIELDS
    model_type: ModelType,           // LLM or SD
    vendor: String,                  // "huggingface", "civitai", "local"
    quantization: Option<String>,    // "Q4_K_M", "F16", etc.
    metadata: HashMap<String, String>, // Flexible for GGUF/SafeTensors metadata
}

pub enum ModelType {
    Llm {
        architecture: String,      // "llama", "mistral", etc.
        parameter_count: Option<String>, // "7B", "13B", etc.
        context_length: Option<u32>,
    },
    StableDiffusion {
        version: SdVersion,        // V1_5, V2_1, XL, Turbo
    },
}
```

### Recommendation 2: Extend WorkerType ✅ DO THIS

```rust
pub enum WorkerType {
    // LLM workers
    CpuLlm,
    CudaLlm,
    MetalLlm,
    
    // SD workers
    CpuSd,
    CudaSd,
    MetalSd,
}

impl WorkerType {
    pub fn model_type(&self) -> ModelTypeCategory {
        match self {
            Self::CpuLlm | Self::CudaLlm | Self::MetalLlm => ModelTypeCategory::Llm,
            Self::CpuSd | Self::CudaSd | Self::MetalSd => ModelTypeCategory::StableDiffusion,
        }
    }
    
    pub fn binary_name(&self) -> &str {
        match self {
            Self::CpuLlm => "llm-worker-rbee-cpu",
            Self::CudaLlm => "llm-worker-rbee-cuda",
            Self::MetalLlm => "llm-worker-rbee-metal",
            Self::CpuSd => "sd-worker-rbee-cpu",
            Self::CudaSd => "sd-worker-rbee-cuda",
            Self::MetalSd => "sd-worker-rbee-metal",
        }
    }
}
```

### Recommendation 3: Create CivitAIVendor ✅ DO THIS

**New file:** `bin/25_rbee_hive_crates/model-provisioner/src/civitai.rs`

```rust
pub struct CivitAIVendor {
    api_token: String,
    client: reqwest::Client,
}

impl CivitAIVendor {
    pub fn new(api_token: String) -> Result<Self> {
        let client = reqwest::Client::new();
        Ok(Self { api_token, client })
    }
}

#[async_trait]
impl VendorSource for CivitAIVendor {
    async fn download(&self, id: &str, dest: &Path, job_id: &str, cancel_token: CancellationToken) -> Result<u64> {
        // Parse "civitai:123456" or "civitai:123456:filename.safetensors"
        let (model_id, filename) = parse_civitai_id(id)?;
        
        // Build download URL
        let url = format!(
            "https://civitai.com/api/download/models/{}?token={}",
            model_id, self.api_token
        );
        
        // Stream download with progress (similar to HuggingFaceVendor)
        download_with_progress(&url, dest, job_id, cancel_token).await
    }
    
    fn supports(&self, id: &str) -> bool {
        id.starts_with("civitai:")
    }
    
    fn name(&self) -> &str {
        "CivitAI"
    }
}
```

### Recommendation 4: Create worker-provisioner Crate ⚠️ OPTIONAL

**Location:** `bin/25_rbee_hive_crates/worker-provisioner/`

**Move from rbee-hive:**
- `pkgbuild_parser.rs` → `worker-provisioner/src/pkgbuild/parser.rs`
- `pkgbuild_executor.rs` → `worker-provisioner/src/pkgbuild/executor.rs`
- `source_fetcher.rs` → `worker-provisioner/src/source_fetcher.rs`
- `worker_install.rs` → `worker-provisioner/src/provisioner.rs`

**Benefits:**
- ✅ Cleaner separation of concerns
- ✅ Easier to test
- ✅ Reusable by queen-rbee

**Drawbacks:**
- ⚠️ More crates to maintain
- ⚠️ Not critical for MVP

**Decision:** ⏸️ DEFER - Keep in rbee-hive for now, refactor later if needed

---

## 🎨 UI Architecture Issues

### Current Hive UI Problems

**Location:** `bin/20_rbee_hive/ui/app/src/components/`

**ModelManagement/** (9 files, messy):
- `DownloadedModelsView.tsx` - Shows downloaded models
- `LoadedModelsView.tsx` - Shows loaded models
- `SearchResultsView.tsx` - Shows search results
- `FilterPanel.tsx` - Filters
- `ModelDetailsPanel.tsx` - Details
- `index.tsx` - Main component
- `types.ts`, `utils.ts`, `README.md`

**WorkerManagement/** (8 files, messy):
- `ActiveWorkersView.tsx` - Shows active workers
- `InstalledWorkersView.tsx` - Shows installed workers
- `SpawnWorkerView.tsx` - Spawn form
- `WorkerCatalogView.tsx` - Catalog browser
- `WorkerCard.tsx` - Worker card component
- `index.tsx` - Main component
- `types.ts`, `README.md`

**Problems:**
- ❌ Too many separate views
- ❌ No unified model/worker pairing
- ❌ Can't see "which models work with which workers"
- ❌ No dynamic tabs
- ❌ No side-by-side LLM + SD demo

---

## 🚀 WOW Factor UI Vision

### Goal: Split-Screen Demo

**Left Side:** LLM Chat Interface  
**Right Side:** Image Generation Interface

**Demo Scenario:**
```
User has 2 GPUs:
- GPU 0: Running LLM worker (llama-3.2-7b)
- GPU 1: Running SD worker (stable-diffusion-xl)

Left panel: Chat with LLM in real-time
Right panel: Generate images in real-time

BOTH RUNNING SIMULTANEOUSLY!
```

### Dynamic Tabs Architecture

**Main Layout:**
```
┌─────────────────────────────────────────────┐
│  Keeper Shell                               │
├─────────────────────────────────────────────┤
│  [Workers] [Models] [Hives] [Settings]      │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┬─────────────┐             │
│  │ LLM Chat    │ Image Gen   │  ← Dynamic! │
│  │             │             │             │
│  │ GPU 0       │ GPU 1       │             │
│  └─────────────┴─────────────┘             │
│                                             │
└─────────────────────────────────────────────┘
```

**Tab System:**
- Tabs are created when workers spawn
- Each worker gets a tab
- Tabs can be arranged side-by-side
- Tabs show worker type icon (💬 for LLM, 🎨 for SD)
- Tabs show GPU assignment

**Implementation:**
```tsx
<WorkerTabs>
  {activeWorkers.map(worker => (
    <WorkerTab
      key={worker.id}
      worker={worker}
      icon={worker.type === 'llm' ? '💬' : '🎨'}
      gpu={worker.device}
    >
      {worker.type === 'llm' ? (
        <LlmChatInterface worker={worker} />
      ) : (
        <SdGenerationInterface worker={worker} />
      )}
    </WorkerTab>
  ))}
</WorkerTabs>
```

---

## 📋 Implementation Priority

### Phase 0: Critical Fixes (3-5 days)

1. **Extend ModelEntry** (1 day)
   - Add `model_type` field
   - Add `vendor` field
   - Add `metadata` field
   - Update catalog to handle new fields

2. **Extend WorkerType** (1 day)
   - Add SD worker types (CpuSd, CudaSd, MetalSd)
   - Update binary_name() method
   - Update crate_name() method
   - Update build_features() method

3. **Implement CivitAIVendor** (2-3 days)
   - Create `civitai.rs` in model-provisioner
   - Implement VendorSource trait
   - Add API token configuration
   - Test with real SD models

### Phase 1: Worker Spawning (5-7 days)

1. **Model-Worker Compatibility** (1 day)
   - Add `compatible_worker_types()` to ModelEntry
   - Add validation in worker spawn
   - Add UI hints for compatibility

2. **SD Worker Spawning** (2-3 days)
   - Create SD worker PKGBUILDs
   - Update worker_install.rs for SD workers
   - Test spawning flow

3. **End-to-End Testing** (2-3 days)
   - Test LLM worker spawning
   - Test SD worker spawning
   - Test model download (HF + CivitAI)
   - Test worker-model pairing

### Phase 2: UI Overhaul (10-15 days)

1. **Dynamic Tab System** (3-4 days)
   - Create WorkerTabs component
   - Create WorkerTab component
   - Implement tab management
   - Test side-by-side layout

2. **LLM Chat Interface** (3-4 days)
   - Chat UI component
   - Message streaming
   - Token count display
   - Stop generation button

3. **SD Generation Interface** (3-4 days)
   - Generation form
   - Progress bar
   - Image preview
   - Download button

4. **Integration & Polish** (1-3 days)
   - Wire up to backend
   - Test dual-worker demo
   - Polish UI
   - Add animations

---

## 🎯 Next Steps

**Immediate (You):**
1. Review this research document
2. Approve architecture changes
3. Decide on implementation phases

**Next (3-5 days):**
1. Extend ModelEntry with model_type
2. Extend WorkerType with SD workers
3. Implement CivitAIVendor

**Then (5-7 days):**
1. Implement worker spawning
2. Test end-to-end flow
3. Verify model-worker compatibility

**Finally (10-15 days):**
1. Build dynamic tab UI
2. Create split-screen demo
3. Polish for launch

---

## 📊 Summary

### What Works ✅
- artifact-catalog abstraction
- model-catalog (basic)
- model-provisioner (HuggingFace)
- worker-catalog (LLM workers)
- worker installation (LLM workers)

### What's Missing ❌
- Model type distinction (LLM vs SD)
- SD worker types
- CivitAI vendor
- Model-worker compatibility validation
- Unified UI for model/worker management
- Dynamic tab system

### Critical Path 🎯
1. Model type distinction → Worker type extension → CivitAI vendor
2. Worker spawning implementation → End-to-end testing
3. Dynamic tab UI → Split-screen demo → Launch!

**Total Estimated Time:** 18-27 days (3-4 weeks)

**Blocker:** None - all pieces are in place, just need implementation

**Ready to proceed!** 🚀
