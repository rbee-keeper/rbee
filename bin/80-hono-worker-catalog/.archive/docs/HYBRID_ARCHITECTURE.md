# Hybrid Worker Catalog Architecture

**Date:** 2025-11-04  
**Status:** 📋 DESIGN PROPOSAL  
**Current:** Hono + Cloudflare Worker serving static PKGBUILDs  
**Target:** Hybrid Git Catalog + Binary Registry

---

## 🎯 Vision

Create a **hybrid system** that combines:
1. **Git-based catalog** for discovery, documentation, and source builds
2. **Binary registry** for efficient distribution of pre-built workers
3. **Unified API** via Hono/Cloudflare Worker

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Git Catalog (github.com/veighnsche/rbee-worker-catalog)     │
│                                                              │
│ ├── master (catalog index)                                  │
│ ├── llm-worker-rbee-cpu (metadata + PKGBUILD)              │
│ ├── llm-worker-rbee-cuda (metadata + PKGBUILD)             │
│ └── llm-worker-rbee-premium (metadata only)                │
│                                                              │
│ Purpose: Discovery, documentation, source builds            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Hono API (Cloudflare Worker - THIS SERVICE)                 │
│                                                              │
│ ├── GET /v1/workers (list all)                             │
│ ├── GET /v1/workers/:id (get metadata)                     │
│ ├── GET /v1/workers/:id/pkgbuild (source build)            │
│ ├── GET /v1/workers/:id/versions (list versions)           │
│ └── GET /v1/workers/:id/:version/download (binary)         │
│                                                              │
│ Purpose: Unified API for both source and binary            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Cloudflare R2 (Binary Storage)                              │
│                                                              │
│ /workers/                                                    │
│   ├── llm-worker-rbee-cpu/                                 │
│   │   ├── 0.1.0/                                           │
│   │   │   ├── linux-x86_64.tar.gz                          │
│   │   │   └── linux-aarch64.tar.gz                         │
│   │   └── 0.2.0/...                                        │
│   └── llm-worker-rbee-premium/                             │
│       └── 0.1.0/... (requires auth)                        │
│                                                              │
│ Purpose: Fast binary distribution                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Cloudflare D1 (Metadata Database)                           │
│                                                              │
│ Tables:                                                      │
│ ├── workers (id, name, description, license, type)         │
│ ├── versions (worker_id, version, platforms, checksum)     │
│ ├── downloads (worker_id, version, count, timestamp)       │
│ └── licenses (user_id, worker_id, token, expires_at)       │
│                                                              │
│ Purpose: Fast queries, analytics, license management        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Distribution Types

### Type 1: Open Source (Source Build)

**Example:** `llm-worker-rbee-cpu`

**Git Branch Contains:**
- `metadata.json` - Worker info
- `PKGBUILD` - Build instructions
- `README.md` - Documentation

**Installation:**
```bash
# Option A: Build from source
git clone -b llm-worker-rbee-cpu https://github.com/veighnsche/rbee-worker-catalog.git
cd rbee-worker-catalog
makepkg -si

# Option B: Download pre-built binary (faster)
rbee-hive install llm-worker-rbee-cpu --version 0.1.0
```

**API Flow:**
1. `GET /v1/workers/llm-worker-rbee-cpu` → Returns metadata
2. User chooses: source build OR binary download
3. Source: `GET /v1/workers/llm-worker-rbee-cpu/pkgbuild`
4. Binary: `GET /v1/workers/llm-worker-rbee-cpu/0.1.0/download`

### Type 2: Open Source (Binary Only)

**Example:** `llm-worker-rbee-cuda` (CUDA builds are complex)

**Git Branch Contains:**
- `metadata.json` - Worker info
- `README.md` - Documentation
- NO PKGBUILD (binary-only distribution)

**Installation:**
```bash
# Only binary download available
rbee-hive install llm-worker-rbee-cuda --version 0.1.0
```

**API Flow:**
1. `GET /v1/workers/llm-worker-rbee-cuda` → Returns metadata (no PKGBUILD)
2. `GET /v1/workers/llm-worker-rbee-cuda/0.1.0/download` → Returns R2 URL

### Type 3: Premium (Binary + License)

**Example:** `llm-worker-rbee-premium`

**Git Branch Contains:**
- `metadata.json` - Worker info (requires_license: true)
- `README.md` - Documentation + pricing

**Installation:**
```bash
# Requires license token
export RBEE_LICENSE_TOKEN="rbee_lic_abc123..."
rbee-hive install llm-worker-rbee-premium --version 0.1.0
```

**API Flow:**
1. `GET /v1/workers/llm-worker-rbee-premium` → Returns metadata (requires license)
2. `GET /v1/workers/llm-worker-rbee-premium/0.1.0/download` → Checks license
3. If valid: Returns presigned R2 URL
4. If invalid: Returns 403 with purchase link

---

## 🔧 Hono API Design

### Current Endpoints (Keep)

```typescript
GET /workers                    // List all workers
GET /workers/:id                // Get worker metadata
GET /workers/:id/PKGBUILD       // Get PKGBUILD (if exists)
```

### New Endpoints (Add)

```typescript
// Version management
GET /v1/workers/:id/versions
// Response: { versions: ["0.1.0", "0.2.0"], latest: "0.2.0" }

// Binary download
GET /v1/workers/:id/:version/download?platform=linux-x86_64
// Response: { download_url: "https://r2.../worker.tar.gz", checksum: "sha256:..." }
// For premium: Requires Authorization header

// License verification (premium only)
POST /v1/licenses/verify
// Body: { token: "rbee_lic_..." }
// Response: { valid: true, worker_ids: [...], expires_at: "..." }

// Analytics (optional)
GET /v1/workers/:id/stats
// Response: { downloads: 1234, stars: 56, last_updated: "..." }
```

### Backward Compatibility

**Old clients** (using current API):
- Still work! `/workers` and `/workers/:id/PKGBUILD` unchanged
- Get PKGBUILD, build from source

**New clients** (using v1 API):
- Can choose binary download for faster installation
- Can access premium workers with license

---

## 📁 Worker Metadata Schema

### metadata.json (in Git branch)

```json
{
  "id": "llm-worker-rbee-cpu",
  "name": "LLM Worker (CPU)",
  "description": "CPU-only LLM inference worker for rbee",
  "version": "0.1.0",
  "license": "GPL-3.0-or-later",
  "author": "rbee Core Team",
  "homepage": "https://github.com/veighnsche/llama-orch",
  
  "distribution": {
    "type": "hybrid",
    "source_available": true,
    "binary_available": true,
    "requires_license": false
  },
  
  "platforms": [
    {
      "os": "linux",
      "arch": "x86_64",
      "binary_size": "45MB",
      "checksum": "sha256:abc123..."
    },
    {
      "os": "linux",
      "arch": "aarch64",
      "binary_size": "42MB",
      "checksum": "sha256:def456..."
    }
  ],
  
  "dependencies": {
    "system": ["gcc", "glibc >= 2.31"],
    "runtime": []
  },
  
  "features": [
    "CPU-optimized inference",
    "Multi-threading support",
    "GGUF model support"
  ],
  
  "tags": ["llm", "cpu", "inference", "llama"]
}
```

### Premium Worker Metadata

```json
{
  "id": "llm-worker-rbee-premium",
  "name": "LLM Worker (Premium)",
  "description": "High-performance LLM worker with advanced features",
  "version": "0.1.0",
  "license": "Proprietary",
  
  "distribution": {
    "type": "binary-only",
    "source_available": false,
    "binary_available": true,
    "requires_license": true
  },
  
  "pricing": {
    "model": "subscription",
    "price_usd": 99,
    "interval": "month",
    "trial_days": 14,
    "purchase_url": "https://rbee.ai/premium"
  },
  
  "features": [
    "🚀 10x faster inference",
    "🎯 Advanced caching",
    "📊 Built-in analytics",
    "🔒 Priority support"
  ]
}
```

---

## 🗄️ Database Schema (Cloudflare D1)

```sql
-- Workers table
CREATE TABLE workers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    license TEXT NOT NULL,
    distribution_type TEXT NOT NULL, -- 'source', 'binary', 'hybrid'
    requires_license BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Versions table
CREATE TABLE versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    version TEXT NOT NULL,
    platforms TEXT NOT NULL, -- JSON array
    checksum TEXT NOT NULL,
    binary_size INTEGER,
    release_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (worker_id) REFERENCES workers(id),
    UNIQUE(worker_id, version)
);

-- Downloads table (analytics)
CREATE TABLE downloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    version TEXT NOT NULL,
    platform TEXT NOT NULL,
    user_agent TEXT,
    ip_hash TEXT, -- Hashed for privacy
    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (worker_id) REFERENCES workers(id)
);

-- Licenses table (premium workers)
CREATE TABLE licenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT UNIQUE NOT NULL,
    user_email TEXT NOT NULL,
    worker_ids TEXT NOT NULL, -- JSON array of allowed worker IDs
    status TEXT NOT NULL, -- 'active', 'expired', 'cancelled'
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_versions_worker ON versions(worker_id);
CREATE INDEX idx_downloads_worker ON downloads(worker_id);
CREATE INDEX idx_downloads_date ON downloads(downloaded_at);
CREATE INDEX idx_licenses_token ON licenses(token);
CREATE INDEX idx_licenses_status ON licenses(status);
```

---

## 🔐 Authentication Flow (Premium)

### License Token Format

```
rbee_lic_[base64(user_id:worker_ids:expires_at:signature)]

Example:
rbee_lic_dXNlcjEyMzpsb
```

### Verification Flow

```typescript
// 1. Client requests download
GET /v1/workers/llm-worker-rbee-premium/0.1.0/download
Authorization: Bearer rbee_lic_abc123...

// 2. Hono verifies license
const license = await verifyLicense(token, workerId)
if (!license.valid) {
  return c.json({ 
    error: "Invalid license",
    purchase_url: "https://rbee.ai/premium"
  }, 403)
}

// 3. Generate presigned R2 URL (expires in 1 hour)
const downloadUrl = await generatePresignedUrl(workerId, version, platform)

// 4. Return download URL
return c.json({
  download_url: downloadUrl,
  expires_in: 3600,
  checksum: "sha256:..."
})
```

---

## 📦 Binary Storage Structure (R2)

```
rbee-workers/
├── llm-worker-rbee-cpu/
│   ├── 0.1.0/
│   │   ├── linux-x86_64.tar.gz
│   │   ├── linux-x86_64.tar.gz.sha256
│   │   ├── linux-aarch64.tar.gz
│   │   └── linux-aarch64.tar.gz.sha256
│   └── 0.2.0/
│       └── ...
├── llm-worker-rbee-cuda/
│   └── 0.1.0/
│       ├── linux-x86_64.tar.gz
│       └── linux-x86_64.tar.gz.sha256
└── llm-worker-rbee-premium/ (private bucket)
    └── 0.1.0/
        ├── linux-x86_64.tar.gz
        └── linux-x86_64.tar.gz.sha256
```

### Binary Package Structure

```
llm-worker-rbee-cpu-0.1.0-linux-x86_64.tar.gz
├── bin/
│   └── llm-worker-rbee-cpu (executable)
├── LICENSE
├── README.md
└── metadata.json
```

---

## 🚀 Migration Path

### Phase 1: Current State (✅ DONE)
- Hono server serving static PKGBUILDs
- Simple `/workers` and `/workers/:id/PKGBUILD` endpoints

### Phase 2: Git Catalog (Week 1)
- [ ] Create `rbee-worker-catalog` repository
- [ ] Create branches for each worker
- [ ] Add metadata.json to each branch
- [ ] Update Hono to fetch from GitHub

### Phase 3: Binary Registry (Week 2)
- [ ] Set up Cloudflare R2 bucket
- [ ] Upload pre-built binaries
- [ ] Add `/v1/workers/:id/:version/download` endpoint
- [ ] Implement checksum verification

### Phase 4: Database (Week 3)
- [ ] Set up Cloudflare D1 database
- [ ] Create tables
- [ ] Sync metadata from Git to D1
- [ ] Add analytics endpoints

### Phase 5: Premium Support (Week 4)
- [ ] Implement license verification
- [ ] Add authentication middleware
- [ ] Set up private R2 bucket
- [ ] Create license management UI

---

## ✅ Hono Compatibility

**YES! The current Hono setup is 100% compatible.**

### What Works Already
- ✅ Cloudflare Worker runtime
- ✅ Asset serving (for PKGBUILDs)
- ✅ CORS configured
- ✅ TypeScript setup

### What Needs to be Added
- 🆕 Cloudflare R2 binding (binary storage)
- 🆕 Cloudflare D1 binding (database)
- 🆕 New API routes (v1 endpoints)
- 🆕 Authentication middleware

### Updated wrangler.jsonc

```jsonc
{
  "name": "rbee-worker-catalog",
  "main": "src/index.ts",
  "compatibility_date": "2025-11-01",
  
  // Keep existing asset binding
  "assets": {
    "binding": "ASSETS",
    "directory": "./public"
  },
  
  // Add R2 binding for binary storage
  "r2_buckets": [
    {
      "binding": "WORKER_BINARIES",
      "bucket_name": "rbee-workers",
      "preview_bucket_name": "rbee-workers-preview"
    }
  ],
  
  // Add D1 binding for metadata
  "d1_databases": [
    {
      "binding": "DB",
      "database_name": "rbee-catalog",
      "database_id": "xxx-xxx-xxx",
      "preview_database_id": "yyy-yyy-yyy"
    }
  ],
  
  // Add KV for caching (optional)
  "kv_namespaces": [
    {
      "binding": "CACHE",
      "id": "xxx-xxx-xxx",
      "preview_id": "yyy-yyy-yyy"
    }
  ],
  
  // Environment variables
  "vars": {
    "GITHUB_CATALOG_REPO": "veighnsche/rbee-worker-catalog",
    "GITHUB_CATALOG_BRANCH": "master"
  },
  
  "dev": {
    "port": 8787
  }
}
```

---

## 📝 Next Steps

See [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md) for detailed tasks.

**TEAM-402 - Hybrid Architecture Design Complete!** 🎉
