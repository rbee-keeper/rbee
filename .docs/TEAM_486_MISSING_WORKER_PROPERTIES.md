# TEAM-486: Missing Properties in GWCWorker Type

**Date:** 2025-11-12  
**Status:** ⚠️ ANALYSIS COMPLETE - Action Required

## Critical Finding

The `convertGWCWorker` function at line 23-50 in `/frontend/packages/marketplace-core/src/adapters/gwc/list.ts` **does NOT map `coverImage` to `imageUrl`**, even though the UI explicitly uses it!

```typescript
// ❌ CURRENT (BROKEN)
export function convertGWCWorker(worker: GWCWorker): MarketplaceModel {
  return {
    id: worker.id,
    name: worker.name,
    // ... other fields ...
    // ❌ imageUrl is MISSING! (line 23-50 has NO imageUrl assignment)
  }
}

// UI expects imageUrl (workers/page.tsx line 71)
...(model.imageUrl ? { imageUrl: model.imageUrl } : {})
```

**Result:** Worker cover images are NOT showing in the marketplace!

## Missing Properties Analysis

### 1. **imageUrl** (CRITICAL - BROKEN)

**Status:** ⚠️ **Data exists but not mapped**

**Current State:**
- `GWCWorker.coverImage` exists ✅
- Converter does NOT map it to `imageUrl` ❌
- UI expects `model.imageUrl` ✅

**Used By:**
- `/apps/marketplace/app/workers/page.tsx` line 71
- `/packages/rbee-ui/src/marketplace/organisms/WorkerListCard/WorkerListCard.tsx` line 47-53

**Fix Required:**
```typescript
// Add to convertGWCWorker() at line 40
imageUrl: worker.coverImage,
```

---

### 2. **Category / Worker Type Classification** (HIGH PRIORITY)

**Status:** ❌ **Missing**

**What Users Need:**
- "Is this a text worker, image worker, audio worker, or video worker?"
- Easy filtering: "Show me all image generation workers"
- Quick decision-making without reading full description

**Current Workaround:**
- UI extracts from tags: `model.tags.filter(...)` (line 41-42 in workers/[slug]/page.tsx)
- Unreliable (what if tags are inconsistent?)

**Proposed Addition:**
```typescript
interface GWCWorker {
  // Add to Metadata section
  /** Worker category for easy filtering */
  category: 'text' | 'image' | 'audio' | 'video' | 'multimodal' | 'utility'
}
```

**Benefits:**
- Reliable filtering
- Better UX (category badges)
- Marketplace can group by category

---

### 3. **Hardware Requirements** (HIGH PRIORITY)

**Status:** ❌ **Missing**

**What Users Need:**
- "Can I run this worker on my hardware?"
- "How much RAM/VRAM do I need?"
- "How much disk space for the binary?"

**Current State:**
- Users must guess from variant backends
- No minimum RAM/VRAM info
- No disk space requirements

**Proposed Addition:**
```typescript
interface WorkerCapabilities {
  // Add hardware requirements
  /** Minimum hardware requirements */
  hardwareRequirements?: {
    /** Minimum RAM (bytes) - e.g., 8GB for LLM worker */
    minRamBytes?: number
    /** Minimum VRAM (bytes) - e.g., 6GB for CUDA variant */
    minVramBytes?: number
    /** Approximate binary size (bytes) */
    binarySizeBytes?: number
    /** Disk space for models cache (bytes) */
    modelCacheSizeBytes?: number
  }
}
```

**Benefits:**
- Users know if they can run it BEFORE downloading
- Marketplace can filter by available hardware
- Better user experience (no failed installations)

---

### 4. **Performance Metrics** (MEDIUM PRIORITY)

**Status:** ❌ **Missing**

**What Users Need:**
- "How fast is this worker?"
- "What's the throughput?"
- "Compare performance across workers"

**Current State:**
- No performance data
- Users can't compare workers
- No expectations set

**Proposed Addition:**
```typescript
interface WorkerCapabilities {
  /** Performance benchmarks (optional) */
  performance?: {
    /** Tokens per second (for LLM workers) */
    tokensPerSecond?: number
    /** Images per minute (for SD workers) */
    imagesPerMinute?: number
    /** Hardware used for benchmark */
    benchmarkHardware?: string  // e.g., "RTX 4090"
  }
}
```

**Benefits:**
- Users can compare workers
- Set performance expectations
- Help choose the right worker for their use case

---

### 5. **Release Dates** (MEDIUM PRIORITY)

**Status:** ⚠️ **Hardcoded to Wrong Values**

**Current State:**
```typescript
// Line 37-38 in gwc/list.ts
createdAt: new Date(), // ❌ Wrong! Returns current date, not actual release date
updatedAt: new Date(), // ❌ Wrong! Returns current date, not last update
```

**Problem:**
- All workers show "created today"
- No way to sort by "newest" or "recently updated"
- Misleading for users

**Proposed Addition:**
```typescript
interface GWCWorker {
  // Add to Identity section
  /** Initial release date (ISO 8601 string) */
  releasedAt: string  // e.g., "2025-01-15T00:00:00Z"
  /** Last updated date (ISO 8601 string) */
  updatedAt: string   // e.g., "2025-11-12T16:30:00Z"
}
```

**Benefits:**
- Accurate sorting by date
- Show "Recently Updated" badge
- Users can see if worker is actively maintained

---

### 6. **Author Information** (MEDIUM PRIORITY)

**Status:** ⚠️ **Hardcoded**

**Current State:**
```typescript
author: 'rbee', // Line 26 - All workers show "rbee" as author
```

**Problem:**
- No way to distinguish official vs community workers
- No credit to actual developers
- Future marketplace (community workers) will have wrong author

**Proposed Addition:**
```typescript
interface GWCWorker {
  // Add to Metadata section
  /** Author/organization name */
  author: string  // e.g., "rbee" or "CommunityUser"
  /** Author type */
  authorType: 'official' | 'community' | 'verified' | 'unverified'
  /** Author URL (optional) */
  authorUrl?: string
}
```

**Benefits:**
- Proper credit
- Trust indicators (official badge)
- Future marketplace ready

---

### 7. **Pricing / Tier Information** (MEDIUM PRIORITY)

**Status:** ❌ **Missing**

**What Users Need:**
- "Is this free or paid?"
- "Is this GPL-3.0 core or premium?"
- "Marketplace worker pricing?"

**Current State:**
- License field exists (GPL-3.0-or-later)
- No pricing/tier info
- Can't filter by "free only" or "premium"

**Proposed Addition:**
```typescript
interface GWCWorker {
  // Add to Metadata section
  /** Pricing tier */
  tier: 'free' | 'premium' | 'marketplace'
  /** Price (for marketplace workers, in cents) */
  priceInCents?: number  // e.g., 9900 = $99.00
  /** One-time or subscription */
  pricingModel?: 'one-time' | 'subscription' | 'free'
}
```

**Benefits:**
- Clear pricing transparency
- Filter by price
- Marketplace monetization ready

---

### 8. **Binary Size** (LOW PRIORITY)

**Status:** ⚠️ **Partially Available**

**Current State:**
- `MarketplaceModel.sizeBytes` exists
- NOT populated from GWC (line 23-50 has no sizeBytes)
- UI uses it: line 81 in civitai/page.tsx

**Used For:**
- Download size display
- Disk space planning
- "This will download 350 MB"

**Proposed Addition:**
```typescript
interface BuildVariant {
  // Add to Installation section
  /** Approximate binary size in bytes */
  binarySizeBytes: number
}
```

**Why per-variant:**
- CPU binary: ~50MB
- CUDA binary: ~350MB (includes CUDA libs)
- Metal binary: ~80MB
- Different sizes per backend

---

### 9. **Status / Maturity Level** (LOW PRIORITY)

**Status:** ❌ **Missing**

**What Users Need:**
- "Is this stable or experimental?"
- "Can I use this in production?"
- "Is this deprecated?"

**Proposed Addition:**
```typescript
interface GWCWorker {
  // Add to Metadata section
  /** Development status */
  status: 'alpha' | 'beta' | 'stable' | 'deprecated' | 'archived'
  /** Stability level (0-100) */
  stability?: number  // e.g., 95 = very stable
}
```

**Benefits:**
- Users know what to expect
- Can filter out alpha/beta if needed
- Clear deprecation warnings

---

### 10. **Platform-Specific Notes** (LOW PRIORITY)

**Status:** ❌ **Missing**

**What Users Need:**
- "CUDA requires NVIDIA drivers 525+"
- "Metal only works on macOS 13+"
- "ROCm requires AMD GPU"

**Proposed Addition:**
```typescript
interface BuildVariant {
  // Add to Dependencies section
  /** Platform-specific installation notes */
  installNotes?: string
  /** Required driver versions */
  requiredDrivers?: {
    nvidia?: string  // e.g., "525.0 or higher"
    amd?: string
  }
}
```

**Benefits:**
- Prevent installation failures
- Clear requirements upfront
- Better user experience

---

## Priority Summary

| Priority | Property | Status | Impact |
|----------|----------|--------|--------|
| 🔴 **CRITICAL** | `imageUrl` mapping | Broken | Cover images don't show |
| 🟠 **HIGH** | `category` | Missing | No easy filtering |
| 🟠 **HIGH** | `hardwareRequirements` | Missing | Users can't assess if they can run it |
| 🟡 **MEDIUM** | `performance` | Missing | Can't compare workers |
| 🟡 **MEDIUM** | `releasedAt`/`updatedAt` | Wrong values | Misleading dates |
| 🟡 **MEDIUM** | `author`/`authorType` | Hardcoded | No community support |
| 🟡 **MEDIUM** | `tier`/`pricing` | Missing | No pricing transparency |
| 🟢 **LOW** | `binarySizeBytes` | Missing | No download size info |
| 🟢 **LOW** | `status` | Missing | No maturity indicator |
| 🟢 **LOW** | `installNotes` | Missing | No platform-specific guidance |

---

## Immediate Actions Required

### 1. Fix `imageUrl` Mapping (CRITICAL)

**File:** `/frontend/packages/marketplace-core/src/adapters/gwc/list.ts`

```typescript
// Line 40 - Add this:
export function convertGWCWorker(worker: GWCWorker): MarketplaceModel {
  return {
    // ... existing fields ...
    imageUrl: worker.coverImage,  // ✅ ADD THIS LINE
    license: worker.license,
    // ... rest ...
  }
}
```

**Impact:** Worker cover images will now display correctly!

### 2. Add `category` Field (HIGH PRIORITY)

**File:** `/frontend/packages/marketplace-core/src/adapters/gwc/types.ts`

```typescript
export interface GWCWorker {
  // Add after description
  /** Worker category for filtering and organization */
  category: 'text' | 'image' | 'audio' | 'video' | 'multimodal' | 'utility'
}
```

**Data Update:** `/bin/80-global-worker-catalog/src/data.ts`
```typescript
{
  id: 'llm-worker-rbee',
  category: 'text',  // ✅ ADD THIS
  // ...
},
{
  id: 'sd-worker-rbee',
  category: 'image',  // ✅ ADD THIS
  // ...
}
```

### 3. Add `hardwareRequirements` (HIGH PRIORITY)

**File:** `/frontend/packages/marketplace-core/src/adapters/gwc/types.ts`

```typescript
export interface WorkerCapabilities {
  // Add after supportsBatching
  /** Minimum hardware requirements */
  hardwareRequirements?: {
    minRamBytes?: number
    minVramBytes?: number
    binarySizeBytes?: number
    modelCacheSizeBytes?: number
  }
}
```

**Data Update:** `/bin/80-global-worker-catalog/src/data.ts`
```typescript
capabilities: {
  supportedFormats: ['gguf', 'safetensors'],
  maxContextLength: 32768,
  supportsStreaming: true,
  supportsBatching: false,
  hardwareRequirements: {  // ✅ ADD THIS
    minRamBytes: 8 * 1024 * 1024 * 1024,  // 8GB
    minVramBytes: 6 * 1024 * 1024 * 1024,  // 6GB for CUDA
    binarySizeBytes: 50 * 1024 * 1024,  // ~50MB
    modelCacheSizeBytes: 10 * 1024 * 1024 * 1024,  // 10GB for models
  },
}
```

---

## Recommended Type Structure

```typescript
export interface GWCWorker {
  // ━━━ Identity (GLOBAL) ━━━
  id: string
  implementation: WorkerImplementation
  version: string
  releasedAt: string  // ✅ ADD
  updatedAt: string   // ✅ ADD
  
  // ━━━ Metadata (GLOBAL) ━━━
  name: string
  description: string
  license: string
  coverImage?: string
  readmeUrl?: string
  category: 'text' | 'image' | 'audio' | 'video' | 'multimodal' | 'utility'  // ✅ ADD
  author: string  // ✅ ADD
  authorType: 'official' | 'community' | 'verified' | 'unverified'  // ✅ ADD
  authorUrl?: string  // ✅ ADD
  status: 'alpha' | 'beta' | 'stable' | 'deprecated' | 'archived'  // ✅ ADD
  tier: 'free' | 'premium' | 'marketplace'  // ✅ ADD
  priceInCents?: number  // ✅ ADD
  
  // ━━━ Build System (GLOBAL) ━━━
  buildSystem: BuildSystem
  source: SourceConfig
  variants: BuildVariant[]
  
  // ━━━ Capabilities (GLOBAL) ━━━
  capabilities: WorkerCapabilities  // Enhanced below
  
  // ━━━ Marketplace Compatibility (GLOBAL) ━━━
  marketplaceCompatibility: MarketplaceCompatibility
}

export interface WorkerCapabilities {
  supportedFormats: string[]
  maxContextLength?: number
  supportsStreaming: boolean
  supportsBatching: boolean
  hardwareRequirements?: {  // ✅ ADD
    minRamBytes?: number
    minVramBytes?: number
    binarySizeBytes?: number
    modelCacheSizeBytes?: number
  }
  performance?: {  // ✅ ADD
    tokensPerSecond?: number
    imagesPerMinute?: number
    benchmarkHardware?: string
  }
}

export interface BuildVariant {
  // ... existing fields ...
  binarySizeBytes: number  // ✅ ADD
  installNotes?: string  // ✅ ADD
}
```

---

**Created by:** TEAM-486  
**Next Steps:** Fix imageUrl mapping (critical), then add category and hardware requirements
