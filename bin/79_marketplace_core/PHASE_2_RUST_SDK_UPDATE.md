# Phase 2: Update Rust SDK to Use Shared Filters

**TEAM-464: Eliminate duplication in marketplace-sdk**

## Changes Needed

### 1. Remove Duplicate Types from `civitai.rs`

**Lines 19-38:** Remove duplicate `CivitaiModelType` enum
```rust
// ❌ DELETE THIS (lines 19-38)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CivitaiModelType {
    Checkpoint,
    TextualInversion,
    Hypernetwork,
    // ... etc
}
```

### 2. Import Shared Types

**Add to imports (line 12):**
```rust
use artifacts_contract::{
    CivitaiStats, CivitaiCreator,
    // TEAM-464: Add shared filter types
    CivitaiFilters, CivitaiModelType, TimePeriod, BaseModel, CivitaiSort,
    NsfwLevel, NsfwFilter,
};
```

### 3. Update `list_models()` Signature

**Current (lines 285-292):**
```rust
pub async fn list_models(
    &self,
    limit: Option<i32>,
    page: Option<i32>,
    types: Option<Vec<&str>>,
    sort: Option<&str>,
    nsfw: Option<bool>,
    allow_commercial_use: Option<Vec<&str>>,
) -> Result<CivitaiListResponse>
```

**New:**
```rust
pub async fn list_models(
    &self,
    filters: &CivitaiFilters,
) -> Result<CivitaiListResponse>
```

### 4. Update `list_models()` Implementation

**Replace parameter building with:**
```rust
// Build query params from filters
let mut query_params: Vec<(&str, String)> = Vec::new();

// Limit and page
query_params.push(("limit", filters.limit.to_string()));
if let Some(page) = filters.page {
    query_params.push(("page", page.to_string()));
}

// Model types
if filters.model_type != CivitaiModelType::All {
    query_params.push(("types", filters.model_type.as_str().to_string()));
} else {
    // Default: Checkpoint and LORA
    query_params.push(("types", "Checkpoint".to_string()));
    query_params.push(("types", "LORA".to_string()));
}

// Sort
query_params.push(("sort", filters.sort.as_str().to_string()));

// Time period
if filters.time_period != TimePeriod::AllTime {
    query_params.push(("period", filters.time_period.as_str().to_string()));
}

// Base model
if filters.base_model != BaseModel::All {
    query_params.push(("baseModel", filters.base_model.as_str().to_string()));
}

// NSFW filtering
let nsfw_levels = filters.nsfw.max_level.allowed_levels();
for level in nsfw_levels {
    query_params.push(("nsfwLevel", level.as_number().to_string()));
}
```

### 5. Update `get_compatible_models()`

**Current (lines 466-476):**
```rust
pub(crate) async fn get_compatible_models(&self) -> Result<CivitaiListResponse> {
    self.list_models(
        Some(100),
        None,
        Some(vec!["Checkpoint", "LORA"]),
        Some("Most Downloaded"),
        None,
        None,
    )
    .await
}
```

**New:**
```rust
pub(crate) async fn get_compatible_models(&self) -> Result<CivitaiListResponse> {
    let filters = CivitaiFilters::default(); // Uses sensible defaults
    self.list_models(&filters).await
}
```

### 6. Update HuggingFace Client (Similar Pattern)

**File:** `huggingface.rs`

**Current:**
```rust
pub async fn list_models(
    &self,
    query: Option<String>,
    sort: Option<String>,
    filter_tags: Option<Vec<String>>,
    limit: Option<u32>,
) -> Result<Vec<Model>>
```

**New:**
```rust
pub async fn list_models(
    &self,
    filters: &HuggingFaceFilters,
) -> Result<Vec<Model>>
```

## Benefits

✅ **Single source of truth** - No more duplicate enums
✅ **Type-safe** - Compiler enforces correct filter usage
✅ **Simpler API** - One parameter instead of 6+
✅ **Consistent** - Same pattern for Civitai and HuggingFace
✅ **Extensible** - Easy to add new filters without changing signatures

## Migration Impact

### Breaking Changes
- ✅ **Acceptable** - Pre-1.0 software (v0.1.0)
- ✅ **Compiler will find all call sites** - No silent failures
- ✅ **Easy to fix** - Just wrap parameters in `CivitaiFilters {}`

### Call Sites to Update
1. `marketplace-sdk/src/civitai.rs` - `get_compatible_models()`
2. `marketplace-node/src/index.ts` - WASM bindings (auto-generated)
3. `rbee-keeper/src/tauri_commands.rs` - Tauri commands
4. Any tests

## Implementation Order

1. ✅ Update imports
2. ✅ Remove duplicate `CivitaiModelType`
3. ✅ Update `list_models()` signature
4. ✅ Update `list_models()` implementation
5. ✅ Update `get_compatible_models()`
6. ✅ Fix compilation errors
7. ✅ Update HuggingFace client
8. ✅ Test

---

**Status:** Ready to implement
**Next:** Start with civitai.rs updates
