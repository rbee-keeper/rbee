# TEAM-384: Catalog ID Sanitization Fix

**Status:** ✅ COMPLETE

**Date:** Nov 2, 2025

## Problem

Models downloaded successfully but didn't appear in `./rbee model ls`.

### Root Cause

The `FilesystemCatalog` used artifact IDs directly as directory names without sanitization. IDs containing `/` or `:` created nested directory structures that broke the listing logic.

**Example:**
- Model ID: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- Created directory: `~/.cache/rbee/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/`
- `list_ids()` found: `TheBloke/` (top-level directory)
- Tried to load: `~/.cache/rbee/models/TheBloke/metadata.json` ❌ (doesn't exist!)
- Result: Model not listed

### Why This Happened

1. **Catalog save:** Used unsanitized ID → created nested dirs
2. **Catalog list:** Only read top-level dirs → found parent dir
3. **Catalog load:** Tried to load metadata from parent dir → failed silently

## Solution

Added `sanitize_id()` method to `FilesystemCatalog` that replaces `/` and `:` with `-` before using IDs as directory names.

**After fix:**
- Model ID: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- Sanitized: `TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- Directory: `~/.cache/rbee/models/TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/`
- `list_ids()` finds: `TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/`
- Loads metadata: `~/.cache/rbee/models/TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/metadata.json` ✅

## Implementation

### File Changed

**bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs**

Added `sanitize_id()` method:

```rust
/// Sanitize artifact ID for filesystem use
///
/// Replaces / and : with - to prevent nested directories.
/// This ensures each artifact gets a single top-level directory.
///
/// # TEAM-384: Critical for catalog listing
///
/// Without sanitization, IDs like "org/model:file.gguf" create:
/// - catalog_dir/org/model:file.gguf/metadata.json (nested!)
///
/// But list_ids() only reads top-level dirs, so it finds "org/"
/// and tries to load catalog_dir/org/metadata.json (doesn't exist).
///
/// With sanitization, we get:
/// - catalog_dir/org-model-file.gguf/metadata.json (flat!)
fn sanitize_id(&self, id: &str) -> String {
    id.replace('/', "-").replace(':', "-")
}
```

Updated `metadata_path()` to use sanitized IDs:

```rust
fn metadata_path(&self, id: &str) -> PathBuf {
    let safe_id = self.sanitize_id(id);
    self.catalog_dir.join(safe_id).join("metadata.json")
}
```

## Migration

Existing models with nested directories were migrated:

```bash
# Old structure (broken)
~/.cache/rbee/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/metadata.json

# New structure (fixed)
~/.cache/rbee/models/TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/metadata.json
```

## Verification

```bash
# Before fix
$ ./rbee model ls
Found 0 model(s)
No models found. Download a model with: ./rbee model download <model-id>

# After fix
$ ./rbee model ls
Found 1 model(s)
📦 Models:
  • TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (...)
```

## Impact

- ✅ Model listing now works correctly
- ✅ All catalog operations use consistent sanitization
- ✅ No breaking changes to public API (internal implementation detail)
- ✅ Matches provisioner's sanitization pattern (consistency)

## Related Code

This fix aligns with the provisioner's existing sanitization:

```@/home/vince/Projects/llama-orch/bin/25_rbee_hive_crates/model-provisioner/src/provisioner.rs#76-79
fn model_dir(&self, model_id: &str) -> PathBuf {
    // Sanitize model ID for filesystem (replace / with -)
    let safe_id = model_id.replace('/', "-").replace(':', "-");
    self.cache_dir.join(safe_id)
}
```

Both catalog and provisioner now use the same sanitization logic.

## Lessons Learned

1. **Filesystem paths need sanitization:** Never use user input (IDs) directly as directory names
2. **Silent failures are dangerous:** The catalog failed silently when metadata wasn't found
3. **Debug logging is essential:** The eprintln!() statements in catalog.rs helped diagnose this
4. **Test with realistic data:** Unit tests used simple IDs like "test-1", missing this edge case

## Future Improvements

1. Add validation to reject IDs with filesystem-unsafe characters at the API boundary
2. Add integration tests with realistic HuggingFace IDs
3. Consider adding migration tool for existing broken catalogs
4. Add warning when old nested directories are detected
