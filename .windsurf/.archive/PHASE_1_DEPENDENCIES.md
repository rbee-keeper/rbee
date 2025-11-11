# Phase 1: Dependency Setup & Architecture Verification

**Status**: ✅ Complete (TEAM-467)  
**Estimated Time**: 30 minutes  
**Dependencies**: None  
**Blocking**: Phase 2

---

## Objectives

1. ✅ Verify WASM SDK is built correctly
2. ✅ Add marketplace-node as dependency to frontend
3. ✅ Understand the type mappings between Rust and TypeScript
4. ✅ Verify the SDK works in both Node.js and Tauri contexts

---

## Step 1: Verify WASM Build

### Commands
```bash
cd /home/vince/Projects/rbee/bin/79_marketplace_core/marketplace-node

# Check if WASM is built
ls -la wasm/

# Expected files:
# - marketplace_sdk_bg.wasm
# - marketplace_sdk.js
# - marketplace_sdk.d.ts
# - package.json
```

### Verification
```bash
# Test the WASM module loads
node -e "import('@rbee/marketplace-node').then(m => console.log(Object.keys(m)))"

# Expected output: Array of exported functions including:
# - searchHuggingFaceModels
# - listHuggingFaceModels
# - getHuggingFaceModel
# - checkModelCompatibility
```

### If WASM Not Built
```bash
cd /home/vince/Projects/rbee/bin/79_marketplace_core/marketplace-node
pnpm run build

# This runs:
# 1. cd ../marketplace-sdk && wasm-pack build --target nodejs
# 2. tsc (compile TypeScript wrapper)
```

---

## Step 2: Add Dependency to Frontend

### ⚠️ CRITICAL: Import Boundaries

**ALLOWED**:
```typescript
// ✅ Frontend apps import marketplace-node
import { listHuggingFaceModels } from '@rbee/marketplace-node'
```

**FORBIDDEN**:
```typescript
// ❌ Frontend apps NEVER import marketplace-sdk directly
import { HuggingFaceClient } from '@rbee/marketplace-sdk'  // WRONG!
```

**Why?**
- `marketplace-sdk` is Rust/WASM - only for Node.js build scripts
- Frontend runtime doesn't need SDK - it loads static manifests
- Only `marketplace-node` should import `marketplace-sdk`

### Update package.json

**File**: `/home/vince/Projects/rbee/frontend/apps/marketplace/package.json`

```json
{
  "dependencies": {
    "@rbee/marketplace-node": "workspace:*",
    // ❌ DO NOT ADD: "@rbee/marketplace-sdk": "workspace:*"
    // ... other dependencies
  },
  "devDependencies": {
    // marketplace-node is only needed at BUILD TIME for manifest generation
    // but we put it in dependencies, not devDependencies, 
    // because pnpm workspace resolution is easier
  }
}
```

### Install Dependencies
```bash
cd /home/vince/Projects/rbee/frontend/apps/marketplace
pnpm install

# Verify it's linked
ls -la node_modules/@rbee/marketplace-node
```

---

## Step 3: Understand Type Mappings

### Rust → TypeScript Type Reference

#### Rust SDK Types (`marketplace-sdk/src/types.rs`)
```rust
pub struct Model {
    pub id: String,
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub image_url: Option<String>,
    pub tags: Vec<String>,
    pub downloads: f64,
    pub likes: f64,
    pub size: String,
    pub provider: ModelProvider,
    pub category: ModelCategory,
    pub siblings: Option<Vec<ModelFile>>,
}

pub struct ModelFile {
    pub filename: String,
    pub size: Option<f64>,
}
```

#### TypeScript Types (`marketplace-node/src/types.ts`)
```typescript
export interface Model {
  id: string
  name: string
  description: string
  author?: string
  image_url?: string
  tags: string[]
  downloads: number
  likes: number
  size: string
  provider: 'HuggingFace' | 'CivitAI'
  category: 'Llm' | 'Checkpoint' | 'Lora' | 'Unknown'
  siblings?: ModelFile[]
}

export interface ModelFile {
  filename: string
  size?: number  // in bytes
}
```

#### HuggingFace Raw API (`marketplace-node/src/huggingface.ts`)
```typescript
export interface HFModel {
  id: string
  author?: string
  tags?: string[]
  downloads?: number
  likes?: number
  siblings?: Array<{
    rfilename: string  // ⚠️ Note: HF uses "rfilename"
    size?: number
  }>
  pipeline_tag?: string
  cardData?: {
    license?: string
    // ... many more fields
  }
  safetensors?: {
    parameters?: {
      F32?: number
      // ...
    }
  }
}
```

### Key Differences

| Concept | Rust SDK | TypeScript | HuggingFace API |
|---------|----------|------------|-----------------|
| **Model ID** | `id: String` | `id: string` | `id: string` or `modelId: string` |
| **Author** | `author: Option<String>` | `author?: string` | `author?: string` |
| **File field** | `filename: String` | `filename: string` | `rfilename: string` ⚠️ |
| **Size** | `size: String` | `size: string` | Calculated from `safetensors.parameters` |

**IMPORTANT**: The Rust SDK normalizes `rfilename` → `filename`

---

## Step 4: Test SDK in Node.js Context

### Create Test Script

**File**: `/home/vince/Projects/rbee/frontend/apps/marketplace/scripts/test-sdk.ts`

```typescript
#!/usr/bin/env tsx
import { listHuggingFaceModels } from '@rbee/marketplace-node'

async function test() {
  console.log('Testing marketplace-node SDK...')
  
  try {
    const models = await listHuggingFaceModels({
      limit: 5,
      sort: 'downloads'
    })
    
    console.log(`✅ Fetched ${models.length} models`)
    console.log('\nFirst model:')
    console.log(JSON.stringify(models[0], null, 2))
    
    // Verify expected fields
    const model = models[0]
    const hasRequiredFields = 
      model.id && 
      model.name && 
      model.tags && 
      typeof model.downloads === 'number' &&
      typeof model.likes === 'number'
    
    if (hasRequiredFields) {
      console.log('\n✅ Model has all required fields')
    } else {
      console.log('\n❌ Model missing required fields')
    }
    
  } catch (error) {
    console.error('❌ SDK test failed:', error)
    process.exit(1)
  }
}

test()
```

### Run Test
```bash
cd /home/vince/Projects/rbee/frontend/apps/marketplace
tsx scripts/test-sdk.ts

# Expected output:
# ✅ Fetched 5 models
# First model:
# {
#   "id": "huggingface-sentence-transformers/all-MiniLM-L6-v2",
#   "name": "all-MiniLM-L6-v2",
#   "downloads": 138000000,
#   "likes": 4100,
#   ...
# }
# ✅ Model has all required fields
```

---

## Step 5: Verify Tauri Can Use Rust SDK

### Tauri Uses Native Rust (No WASM)

Tauri apps can directly use `marketplace-sdk` without WASM compilation.

**File**: Check if Tauri already imports it

```bash
# Check Tauri dependencies
cat /home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri/Cargo.toml | grep marketplace
```

### If Not Already Added

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri/Cargo.toml`

```toml
[dependencies]
marketplace-sdk = { path = "../../../../../bin/79_marketplace_core/marketplace-sdk" }
```

### Tauri Usage Example

```rust
use marketplace_sdk::huggingface::HuggingFaceClient;

#[tauri::command]
async fn search_huggingface_models(query: String) -> Result<Vec<Model>, String> {
    let client = HuggingFaceClient::new();
    client.search_models(&query, Some(10))
        .await
        .map_err(|e| e.to_string())
}
```

---

## Completion Checklist

- [ ] WASM SDK is built (`ls bin/79_marketplace_core/marketplace-node/wasm/`)
- [ ] Frontend has `@rbee/marketplace-node` dependency
- [ ] `pnpm install` ran successfully
- [ ] Test script runs and returns models
- [ ] Understand Rust ↔ TypeScript type mappings
- [ ] Verified `Model` type has all required fields
- [ ] Verified Tauri can use `marketplace-sdk` directly

---

## Troubleshooting

### Issue: WASM Won't Load in Node.js

**Error**: `Cannot find module '@rbee/marketplace-node'`

**Fix**:
```bash
cd /home/vince/Projects/rbee
pnpm install
cd frontend/apps/marketplace
pnpm install
```

### Issue: TypeScript Can't Find Types

**Error**: `Cannot find module '@rbee/marketplace-node' or its corresponding type declarations`

**Fix**:
Check that `marketplace-node/dist/index.d.ts` exists:
```bash
ls -la bin/79_marketplace_core/marketplace-node/dist/
```

If missing, rebuild:
```bash
cd bin/79_marketplace_core/marketplace-node
pnpm run build
```

### Issue: Rust Compilation Fails

**Error**: `wasm-pack build` fails

**Fix**:
```bash
# Install wasm-pack
cargo install wasm-pack

# Install wasm32 target
rustup target add wasm32-unknown-unknown

# Try again
cd bin/79_marketplace_core/marketplace-node
pnpm run build
```

---

## Next Phase

Once all checkboxes are complete, move to **Phase 2: Manifest Generation**.

**Status**: 🟡 → ✅ (update when complete)
