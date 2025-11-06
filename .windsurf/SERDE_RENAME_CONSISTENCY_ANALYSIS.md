# Serde rename_all Consistency Analysis

**Date:** 2025-11-06  
**Team:** TEAM-421

## Summary

**Overall Codebase:**
- `lowercase`: 36 occurrences (60%)
- `camelCase`: 15 occurrences (25%)
- `snake_case`: 5 occurrences (8%)
- `SCREAMING_SNAKE_CASE`: 4 occurrences (7%)

**Canonical Source (artifacts-contract):**
- `camelCase`: 11 occurrences (58%) - **STRUCTS**
- `lowercase`: 7 occurrences (37%) - **ENUMS**
- `SCREAMING_SNAKE_CASE`: 1 occurrence (5%) - **Quantization enum**

## Pattern in artifacts-contract

### ✅ Consistent Pattern Found

**Enums** → `lowercase`
```rust
#[serde(rename_all = "lowercase")]
pub enum WorkerType { Cpu, Cuda, Metal }
// Serializes as: "cpu", "cuda", "metal"

#[serde(rename_all = "lowercase")]
pub enum Platform { Linux, Macos, Windows }
// Serializes as: "linux", "macos", "windows"

#[serde(rename_all = "lowercase")]
pub enum Architecture { X86_64, Aarch64 }
// Serializes as: "x86_64", "aarch64"

#[serde(rename_all = "lowercase")]
pub enum WorkerImplementation { Rust, Python, Cpp }
// Serializes as: "rust", "python", "cpp"

#[serde(rename_all = "lowercase")]
pub enum BuildSystem { Cargo, Make, Cmake }
// Serializes as: "cargo", "make", "cmake"

#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture { Llama, Mistral, ... }
// Serializes as: "llama", "mistral", ...

#[serde(rename_all = "lowercase")]
pub enum ModelFormat { SafeTensors, Gguf, ... }
// Serializes as: "safetensors", "gguf", ...
```

**Structs** → `camelCase`
```rust
#[serde(rename_all = "camelCase")]
pub struct WorkerBinary {
    pub worker_type: WorkerType,  // → "workerType"
    pub binary_path: String,       // → "binaryPath"
}

#[serde(rename_all = "camelCase")]
pub struct WorkerCatalogEntry {
    pub worker_type: WorkerType,      // → "workerType"
    pub pkgbuild_url: String,         // → "pkgbuildUrl"
    pub build_system: BuildSystem,    // → "buildSystem"
    pub binary_name: String,          // → "binaryName"
    pub install_path: String,         // → "installPath"
    pub supported_formats: Vec<...>,  // → "supportedFormats"
    pub max_context_length: u32,      // → "maxContextLength"
    pub supports_streaming: bool,     // → "supportsStreaming"
    pub supports_batching: bool,      // → "supportsBatching"
}

#[serde(rename_all = "camelCase")]
pub struct SourceInfo {
    #[serde(rename = "type")]
    pub source_type: String,  // → "type" (manual override)
}

#[serde(rename_all = "camelCase")]
pub struct BuildConfig { ... }

#[serde(rename_all = "camelCase")]
pub struct ModelEntry { ... }

#[serde(rename_all = "camelCase")]
pub struct LlmConfig { ... }

#[serde(rename_all = "camelCase")]
pub struct TokenizerConfig { ... }

#[serde(rename_all = "camelCase")]
pub struct ModelMetadata { ... }

#[serde(rename_all = "camelCase")]
pub struct ImageConfig { ... }
```

**Special Case** → `SCREAMING_SNAKE_CASE`
```rust
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Quantization {
    Fp16,    // → "FP16"
    Fp32,    // → "FP32"
    Q4_0,    // → "Q4_0"
    Q8_0,    // → "Q8_0"
}
// Rationale: Quantization values are industry-standard abbreviations
```

## Consistency Rules

### ✅ CONSISTENT in artifacts-contract

1. **Enums** = `lowercase`
   - Simple, readable enum values
   - Example: `"cpu"`, `"linux"`, `"rust"`

2. **Structs** = `camelCase`
   - JavaScript/TypeScript convention
   - Better for frontend consumption
   - Example: `"workerType"`, `"pkgbuildUrl"`

3. **Special Enums** = `SCREAMING_SNAKE_CASE`
   - Only for industry-standard abbreviations
   - Example: `"FP16"`, `"Q4_0"`

### ⚠️ INCONSISTENT in Other Crates

**contracts/api-types** (generated code):
- Mix of `lowercase`, `snake_case`, `SCREAMING_SNAKE_CASE`
- **Reason:** Auto-generated from templates

**audit-logging**:
- Uses `snake_case` for enums
- **Reason:** Internal logging, not exposed to frontend

**marketplace-sdk**:
- Mix of `camelCase` and `lowercase`
- **Reason:** Follows artifacts-contract pattern

## Recommendations

### ✅ Keep Current Pattern

**artifacts-contract** is the canonical source and has a **consistent, logical pattern**:
- Enums → `lowercase` (simple values)
- Structs → `camelCase` (JavaScript-friendly)
- Special → `SCREAMING_SNAKE_CASE` (industry standards)

### ✅ TypeScript Must Match

When creating TypeScript types for Rust structs/enums:

```typescript
// ✅ CORRECT - Matches Rust serialization
interface WorkerCatalogEntry {
  workerType: "cpu" | "cuda" | "metal";  // camelCase field, lowercase enum
  pkgbuildUrl: string;                   // camelCase
  buildSystem: "cargo" | "make";         // camelCase field, lowercase enum
}

// ❌ WRONG - Doesn't match Rust
interface WorkerCatalogEntry {
  worker_type: "cpu" | "cuda" | "metal";  // snake_case field
  pkgbuild_url: string;                   // snake_case
}
```

### ✅ Future TypeScript Generation

**TODO:** Generate TypeScript types from Rust instead of manual definitions:

```typescript
// Instead of manual types in Hono:
import type { WorkerCatalogEntry } from "./types";

// Use auto-generated types from marketplace-sdk:
import type { WorkerCatalogEntry } from "@rbee/marketplace-sdk/wasm";
```

This ensures:
- ✅ Field names always match
- ✅ Enum values always match
- ✅ No serialization errors
- ✅ Single source of truth

## Files Checked

### artifacts-contract (Canonical Source)
- `src/worker.rs` - ✅ Consistent
- `src/worker_catalog.rs` - ✅ Consistent
- `src/model/mod.rs` - ✅ Consistent
- `src/model/llm.rs` - ✅ Consistent
- `src/model/metadata.rs` - ✅ Consistent
- `src/model/image.rs` - ✅ Consistent

### Other Crates
- `contracts/api-types` - ⚠️ Generated code (mixed)
- `contracts/config-schema` - ✅ Uses `lowercase` for enums
- `marketplace-sdk` - ✅ Follows artifacts-contract
- `audit-logging` - ⚠️ Uses `snake_case` (internal only)
- `device-detection` - ✅ Uses `lowercase` for Backend enum

## Conclusion

**Status:** ✅ **CONSISTENT** in canonical source (artifacts-contract)

The pattern is:
1. **Enums** → `lowercase` (simple, readable)
2. **Structs** → `camelCase` (JavaScript-friendly)
3. **Special** → `SCREAMING_SNAKE_CASE` (industry standards only)

**Action Required:**
- ✅ Keep current pattern in artifacts-contract
- ✅ Ensure TypeScript matches Rust serialization
- 🔄 Future: Generate TypeScript from Rust (eliminate manual types)

**No changes needed** - The pattern is already consistent and logical!

---

**TEAM-421 Analysis Complete**
