# ModelFile Type Contract

**TEAM-463: Canonical ModelFile Type**  
**Date:** 2025-11-10  
**Status:** ✅ IMPLEMENTED

## Purpose

`ModelFile` represents a single file in a model repository (e.g., HuggingFace siblings).
It is used across marketplace SDK, catalog, and UI components.

## Source of Truth

**Location:** `bin/97_contracts/artifacts-contract/src/model/mod.rs`

```rust
/// Model file information from repository
/// 
/// TEAM-463: Canonical type for model files (siblings) in repositories.
/// Used by marketplace SDK, catalog, and UI components.
/// 
/// This represents a single file in a model repository (e.g., HuggingFace).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
#[serde(rename_all = "camelCase")]
pub struct ModelFile {
    /// File name (relative path in repo)
    pub filename: String,
    /// File size in bytes (optional, using f64 for TypeScript compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
}
```

## Type Flow

```
artifacts-contract (source of truth)
  ↓ re-exported by
marketplace-sdk
  ↓ generates TypeScript types for
keeper GUI (auto-generated bindings.ts)
  
Next.js marketplace (manual type definition)
  ⚠️ Must be kept in sync manually
```

## Usage

### Rust (Marketplace SDK)

```rust
use marketplace_sdk::ModelFile;

let file = ModelFile {
    filename: "model.safetensors".to_string(),
    size: Some(548227584.0),
};
```

### TypeScript (Keeper GUI - Auto-generated)

```typescript
import type { ModelFile } from './generated/bindings'

const file: ModelFile = {
  filename: 'model.safetensors',
  size: 548227584
}
```

### TypeScript (Next.js Marketplace - Manual)

```typescript
// ⚠️ MUST MATCH: artifacts-contract/src/model/mod.rs::ModelFile
interface ModelFile {
  filename: string
  size?: number | null
}
```

## Why This Matters

**Before TEAM-463:**
- `ModelFile` was defined in `marketplace-sdk/src/types.rs`
- Not a contract type, just an SDK type
- UI components had mismatched types (`rfilename` vs `filename`)

**After TEAM-463:**
- `ModelFile` is a **contract type** in `artifacts-contract`
- Single source of truth
- Re-exported by `marketplace-sdk`
- Clear documentation of type flow
- Manual TypeScript definitions reference the contract

## Verification

```bash
# Verify contract compiles
cargo check -p artifacts-contract

# Verify WASM compilation
cargo check -p artifacts-contract --target wasm32-unknown-unknown

# Verify marketplace SDK uses contract type
cargo check -p marketplace-sdk

# Verify keeper binary
cargo check --bin rbee-keeper
```

## Related Files

- **Contract:** `bin/97_contracts/artifacts-contract/src/model/mod.rs`
- **SDK:** `bin/79_marketplace_core/marketplace-sdk/src/types.rs` (re-exports)
- **UI Component:** `frontend/packages/rbee-ui/src/marketplace/molecules/ModelFilesList/ModelFilesList.tsx`
- **Template:** `frontend/packages/rbee-ui/src/marketplace/templates/ModelDetailPageTemplate/ModelDetailPageTemplate.tsx`

## Migration Notes

If you need to change `ModelFile`:

1. **Update the contract** in `artifacts-contract/src/model/mod.rs`
2. **Verify compilation** with `cargo check -p artifacts-contract`
3. **Update manual TypeScript** in `rbee-ui` if needed
4. **Regenerate bindings** for keeper GUI (automatic on build)
5. **Test both** keeper GUI and Next.js marketplace

## See Also

- `ALL_CONTRACTS_WASM_ENFORCED.md` - Contract crate architecture
- `artifacts-contract/README.md` - Artifact types overview
