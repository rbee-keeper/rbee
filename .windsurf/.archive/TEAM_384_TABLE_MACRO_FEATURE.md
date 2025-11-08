# TEAM-384: Table Formatting Support in n!() Macro

**Status:** ✅ COMPLETE

**Date:** Nov 2, 2025

## Feature

Added `table:` argument to the `n!()` macro for automatic table formatting of JSON arrays.

## Usage

```rust
use serde_json::json;

let models = json!([
    {"id": "model1", "size": "7B"},
    {"id": "model2", "size": "13B"}
]);

n!("model_list", table: models.as_array().unwrap());
```

**Output:**
```
id     │ size
───────┼─────
model1 │ 7B
model2 │ 13B
```

## Implementation

### 1. Added macro variant (narration-core/src/lib.rs)

```rust
// TEAM-384: Table formatting: n!("action", table: &[Value])
($action:expr, table: $table:expr) => {{
    let formatted_table = $crate::format_array_table($table);
    $crate::macro_emit($action, &formatted_table, None, None, env!("CARGO_CRATE_NAME"), stdext::function_name!());
}};
```

### 2. Updated model list to use table formatting (rbee-hive/src/job_router.rs)

**Before (manual formatting):**
```rust
n!("model_list_header", "\n📦 Models:");
for model in &models {
    use rbee_hive_artifact_catalog::Artifact;
    n!("model_list_entry", "  • {} ({})", model.id(), model.path().display());
}
```

**After (table formatting):**
```rust
let models_json: Vec<serde_json::Value> = models.iter().map(|m| {
    serde_json::json!({
        "id": m.id(),
        "name": m.name(),
        "size_gb": format!("{:.2}", m.size() as f64 / 1_000_000_000.0),
        "path": m.path().display().to_string(),
    })
}).collect();

n!("model_list_table", table: &models_json);
```

## Example Output

```bash
$ ./rbee model ls
📋 Listing models on hive 'localhost'
Found 1 model(s)
id                                                                          │ name                          │ path                                                                                                                  │ size_gb
────────────────────────────────────────────────────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────
TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf │ TinyLlama-1.1B-Chat-v1.0-GGUF │ /home/vince/.cache/rbee/models/TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/model.gguf │ 0.67
✅ Model list operation complete
```

## Benefits

1. **Cleaner code:** Single line instead of loop
2. **Consistent formatting:** Uses existing `format_array_table()` function
3. **Better UX:** Aligned columns, clear headers, visual separators
4. **Extensible:** Works with any JSON array data

## Files Changed

- `bin/99_shared_crates/narration-core/src/lib.rs` (+18 LOC)
  - Added `table:` macro variant
  - Updated documentation with table example

- `bin/20_rbee_hive/src/job_router.rs` (+14 LOC, -6 LOC)
  - Converted model list to use table formatting
  - Added size_gb column for better readability

## Future Use Cases

This pattern can be used for:
- Worker lists (`./rbee worker ls`)
- Hive lists (`./rbee hive ls`)
- Process lists (`./rbee worker ps`)
- Any structured data that benefits from tabular display

## Pattern

```rust
// 1. Convert data to JSON array
let data_json: Vec<serde_json::Value> = items.iter().map(|item| {
    serde_json::json!({
        "column1": item.field1,
        "column2": item.field2,
        // ... more columns
    })
}).collect();

// 2. Emit as table
n!("action_name", table: &data_json);
```

## Comparison with Manual Formatting

**Manual (old way):**
- 5-10 lines of code
- Inconsistent formatting
- Hard to maintain
- No column alignment

**Table macro (new way):**
- 1 line to emit
- Consistent formatting (uses `format_array_table()`)
- Easy to maintain (just update JSON structure)
- Automatic column alignment and separators

## Technical Details

- Uses existing `format_array_table()` from `narration-core/src/format.rs`
- Supports any JSON array structure
- Automatically calculates column widths
- Handles empty arrays gracefully
- Works with SSE streaming (each table is a single narration event)
