# Phase 5: Tauri Integration

**Status**: 🟡 Not Started  
**Estimated Time**: 1 hour  
**Dependencies**: Phase 4 Complete  
**Blocking**: None (Final Phase)

---

## Objectives

1. ✅ Verify Tauri can use marketplace-sdk directly (native Rust)
2. ✅ Add HuggingFace browsing to Tauri app
3. ✅ Test that both web and desktop use same SDK
4. ✅ Document the dual-use pattern
5. ✅ Ensure no WASM overhead in Tauri (use native)

---

## Architecture: Web vs Tauri

```
┌─────────────────────────────────────────────────────────┐
│ WEB (Next.js)                                            │
│                                                          │
│ Frontend → marketplace-node (WASM) → Build-time manifests│
│         → Runtime: Load static JSON manifests           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DESKTOP (Tauri)                                          │
│                                                          │
│ Frontend → Tauri Commands → marketplace-sdk (Native Rust)│
│         → Runtime: Direct API calls, no WASM            │
└─────────────────────────────────────────────────────────┘
```

**Key Difference**: 
- **Web**: Uses WASM at build time, serves static manifests at runtime
- **Tauri**: Uses native Rust directly, no WASM overhead

---

## Step 1: Add marketplace-sdk to Tauri

### Check Current Dependencies

```bash
cd /home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri
cat Cargo.toml | grep -A 20 "\[dependencies\]"
```

### Add Dependency if Missing

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...
marketplace-sdk = { path = "../../../../../bin/79_marketplace_core/marketplace-sdk" }
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["full"] }
```

### Build to Verify

```bash
cd /home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri
cargo check

# Expected: No errors
```

---

## Step 2: Create Tauri Commands

### Add HuggingFace Commands

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri/src/commands/marketplace.rs` (create if doesn't exist)

```rust
// TEAM-464: Marketplace commands for Tauri
// Uses marketplace-sdk natively (no WASM overhead)

use marketplace_sdk::huggingface::HuggingFaceClient;
use marketplace_sdk::types::{Model, ModelProvider};
use serde::{Deserialize, Serialize};

/// Search options for HuggingFace models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    pub limit: Option<u32>,
    pub sort: Option<String>,
}

/// Search HuggingFace models
#[tauri::command]
pub async fn search_huggingface_models(
    query: String,
    options: Option<SearchOptions>,
) -> Result<Vec<Model>, String> {
    let client = HuggingFaceClient::new();
    
    let opts = options.unwrap_or(SearchOptions {
        limit: Some(50),
        sort: Some("downloads".to_string()),
    });
    
    client
        .list_models(
            Some(query),
            opts.sort,
            None, // filter_tags
            opts.limit,
        )
        .await
        .map_err(|e| e.to_string())
}

/// List HuggingFace models (no search query)
#[tauri::command]
pub async fn list_huggingface_models(
    options: Option<SearchOptions>,
) -> Result<Vec<Model>, String> {
    let client = HuggingFaceClient::new();
    
    let opts = options.unwrap_or(SearchOptions {
        limit: Some(50),
        sort: Some("downloads".to_string()),
    });
    
    client
        .list_models(
            None, // No search query
            opts.sort,
            None, // filter_tags
            opts.limit,
        )
        .await
        .map_err(|e| e.to_string())
}

/// Get a specific HuggingFace model
#[tauri::command]
pub async fn get_huggingface_model(model_id: String) -> Result<Model, String> {
    let client = HuggingFaceClient::new();
    
    client
        .get_model(&model_id)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_list_huggingface_models() {
        let result = list_huggingface_models(Some(SearchOptions {
            limit: Some(5),
            sort: Some("downloads".to_string()),
        })).await;
        
        assert!(result.is_ok());
        let models = result.unwrap();
        assert!(!models.is_empty());
        assert!(models.len() <= 5);
    }
}
```

### Register Commands in main.rs

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri/src/main.rs`

```rust
mod commands;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // ... existing commands ...
            commands::marketplace::search_huggingface_models,
            commands::marketplace::list_huggingface_models,
            commands::marketplace::get_huggingface_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Create Module File

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src-tauri/src/commands/mod.rs`

```rust
pub mod marketplace;
```

---

## Step 3: Create Frontend Hook

### TypeScript Types

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src/hooks/useMarketplace.ts`

```typescript
import { invoke } from '@tauri-apps/api/tauri'

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
  category: string
}

export interface SearchOptions {
  limit?: number
  sort?: 'downloads' | 'likes' | 'recent'
}

export function useMarketplace() {
  const searchHuggingFaceModels = async (
    query: string,
    options?: SearchOptions
  ): Promise<Model[]> => {
    return invoke('search_huggingface_models', { query, options })
  }

  const listHuggingFaceModels = async (
    options?: SearchOptions
  ): Promise<Model[]> => {
    return invoke('list_huggingface_models', { options })
  }

  const getHuggingFaceModel = async (modelId: string): Promise<Model> => {
    return invoke('get_huggingface_model', { modelId })
  }

  return {
    searchHuggingFaceModels,
    listHuggingFaceModels,
    getHuggingFaceModel,
  }
}
```

---

## Step 4: Create Browse Page in Tauri

### HuggingFace Browse Component

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src/pages/HuggingFaceBrowse.tsx`

```typescript
import { useState, useEffect } from 'react'
import { useMarketplace, type Model } from '../hooks/useMarketplace'

export function HuggingFaceBrowse() {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)
  const [sort, setSort] = useState<'downloads' | 'likes'>('downloads')
  const { listHuggingFaceModels } = useMarketplace()

  useEffect(() => {
    async function loadModels() {
      setLoading(true)
      try {
        const data = await listHuggingFaceModels({ 
          limit: 100,
          sort 
        })
        setModels(data)
      } catch (error) {
        console.error('Failed to load models:', error)
      } finally {
        setLoading(false)
      }
    }
    
    loadModels()
  }, [sort])

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-4">HuggingFace Models</h1>
      
      <div className="mb-4">
        <label className="mr-2">Sort by:</label>
        <select 
          value={sort} 
          onChange={(e) => setSort(e.target.value as 'downloads' | 'likes')}
          className="border p-2 rounded"
        >
          <option value="downloads">Most Downloads</option>
          <option value="likes">Most Likes</option>
        </select>
      </div>

      {loading && <div>Loading...</div>}

      <div className="grid gap-4">
        {models.map((model) => (
          <div key={model.id} className="border p-4 rounded">
            <h3 className="font-semibold">{model.name}</h3>
            <p className="text-sm text-gray-600">{model.description}</p>
            <div className="flex gap-4 mt-2 text-sm">
              <span>👤 {model.author || 'Unknown'}</span>
              <span>⬇️ {model.downloads.toLocaleString()}</span>
              <span>❤️ {model.likes.toLocaleString()}</span>
            </div>
            <div className="flex gap-1 mt-2">
              {model.tags.slice(0, 5).map((tag) => (
                <span key={tag} className="text-xs bg-gray-200 px-2 py-1 rounded">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
```

### Add Route

**File**: `/home/vince/Projects/rbee/frontend/apps/rbee-desktop/src/App.tsx`

```typescript
import { HuggingFaceBrowse } from './pages/HuggingFaceBrowse'

function App() {
  return (
    <Router>
      <Routes>
        {/* ... existing routes ... */}
        <Route path="/marketplace/huggingface" element={<HuggingFaceBrowse />} />
      </Routes>
    </Router>
  )
}
```

---

## Step 5: Test Tauri Integration

### Build Tauri App

```bash
cd /home/vince/Projects/rbee/frontend/apps/rbee-desktop
pnpm tauri build --debug

# Or run in dev mode:
pnpm tauri dev
```

### Manual Test

1. Open Tauri app
2. Navigate to HuggingFace browse page
3. Verify models load
4. Change sort order
5. Verify models update

**Expected**:
```
✅ Models load from HuggingFace API
✅ No WASM loading (uses native Rust)
✅ Sorting works
✅ Downloads/likes display correctly
✅ Performance is fast (native code)
```

---

## Step 6: Compare Web vs Tauri

### Test Same SDK, Different Runtime

**Web (Next.js)**:
```typescript
// Uses WASM at BUILD time
import { listHuggingFaceModels } from '@rbee/marketplace-node'

// At build:
const models = await listHuggingFaceModels({ limit: 500 })
// Save to /public/manifests/hf-filter-small.json

// At runtime:
const manifest = await fetch('/manifests/hf-filter-small.json')
// No SDK calls at runtime!
```

**Tauri**:
```typescript
// Uses Native Rust at RUNTIME
import { invoke } from '@tauri-apps/api/tauri'

// At runtime:
const models = await invoke('list_huggingface_models', { 
  options: { limit: 100 }
})
// Direct SDK calls, no WASM!
```

### Verify Same Data Structure

Both should return the same `Model` type:

```typescript
interface Model {
  id: string
  name: string
  description: string
  author?: string
  downloads: number
  likes: number
  tags: string[]
  // ... etc
}
```

---

## Step 7: Document Dual-Use Pattern

### Create Documentation

**File**: `/home/vince/Projects/rbee/bin/79_marketplace_core/DUAL_USE_PATTERN.md`

```markdown
# Dual-Use Pattern: Web + Tauri

## Overview

The marketplace-sdk is designed to work in both web and desktop contexts:

- **Web (Next.js)**: Uses WASM wrapper at BUILD time, serves static manifests at RUNTIME
- **Desktop (Tauri)**: Uses native Rust directly at RUNTIME, no WASM needed

## Architecture

### Build Time (Web)
```
TypeScript Script → marketplace-node (WASM) → marketplace-sdk → HuggingFace API
                  ↓
            /public/manifests/*.json
```

### Runtime (Web)
```
Next.js → fetch(/manifests/*.json) → Display
(No SDK calls at runtime!)
```

### Runtime (Tauri)
```
React → invoke(tauri_command) → marketplace-sdk (Native) → HuggingFace API → Display
```

## Benefits

1. **Web**: Fast runtime (static JSON), good SEO, cheap hosting
2. **Tauri**: Real-time data, offline capability, full API access
3. **Both**: Same data types, same SDK logic, single source of truth

## When to Use Which

- **Web manifests**: For public catalog pages, SEO, static hosting
- **Tauri native**: For user's personal library, real-time updates, offline mode
```

---

## Completion Checklist

- [ ] Added marketplace-sdk to Tauri Cargo.toml
- [ ] Created Tauri commands for HuggingFace
- [ ] Registered commands in main.rs
- [ ] Created useMarketplace hook
- [ ] Created HuggingFaceBrowse page
- [ ] Tested in Tauri dev mode
- [ ] Verified models load correctly
- [ ] Verified no WASM overhead (check bundle size)
- [ ] Compared data structures between web and Tauri
- [ ] Documented dual-use pattern
- [ ] Both web and Tauri use same SDK ✅

---

## Performance Comparison

| Metric | Web (Manifest) | Tauri (Native) |
|--------|---------------|----------------|
| **Initial Load** | ~100ms | ~500ms |
| **Filter Change** | ~100ms | ~500ms |
| **Data Freshness** | Build time | Real-time |
| **Offline** | Yes (cached) | No |
| **API Calls** | 0 (runtime) | Every load |
| **Bundle Size** | +2KB (JSON) | +0KB (native) |

---

## Troubleshooting

### Issue: Tauri Command Not Found

**Error**: `Command search_huggingface_models not found`

**Fix**: Verify command is registered in `main.rs`:

```rust
.invoke_handler(tauri::generate_handler![
    commands::marketplace::search_huggingface_models,  // ← Must be here
])
```

### Issue: CORS Errors in Tauri

**Error**: `CORS policy blocked`

**Fix**: Tauri doesn't have CORS restrictions. This shouldn't happen.

If it does, check that you're using `invoke()`, not `fetch()`.

### Issue: Slow Performance

**Problem**: Models take >2s to load

**Fix**: 
- Check internet connection
- Reduce `limit` parameter
- Add caching layer
- Consider using manifests in Tauri too

---

## Next Steps

**Status**: 🟡 → ✅ (update when complete)

Once complete, the masterplan is DONE! 🎉

**Final Verification**:
- [ ] Web filtering works
- [ ] Tauri browsing works  
- [ ] Both use same SDK
- [ ] Documentation complete
- [ ] No hacky TypeScript code remains
- [ ] All manifests use WASM SDK
