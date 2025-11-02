# TEAM-384: UI JSON Response Fix

**Status:** ✅ COMPLETE

**Date:** Nov 2, 2025

## Problem

After adding table formatting to the `n!()` macro, the UI showed "No models downloaded" even though the CLI correctly displayed the model. The UI and CLI use the same backend but were getting different outputs.

## Root Cause

The backend was emitting ONLY a table (for CLI users) but NOT the JSON response that the UI expects.

**UI expectation (from `rbee-hive-react/src/index.ts`):**
```typescript
// Find the JSON line (starts with '[' or '{')
// Backend emits narration lines first, then JSON on last line
const jsonLine = lines.reverse().find(line => {
  const trimmed = line.trim()
  return trimmed.startsWith('[') || trimmed.startsWith('{')
})

return jsonLine ? JSON.parse(jsonLine) : []
```

The UI searches through SSE stream lines (in reverse) looking for a line that starts with `[` or `{`, then parses it as JSON.

**What the backend was emitting:**
- ✅ Narration events (table, status messages)
- ❌ NO JSON array

**Result:** UI couldn't find JSON, returned empty array `[]`, showed "No models downloaded"

## Solution

Emit BOTH outputs:
1. **Table** (for CLI users) - human-readable
2. **JSON array** (for UI) - machine-readable

### Implementation

```rust
// TEAM-384: Convert to JSON for both table display and UI consumption
use rbee_hive_artifact_catalog::Artifact;
let models_json: Vec<serde_json::Value> = models.iter().map(|m| {
    serde_json::json!({
        "id": m.id(),
        "name": m.name(),
        "size": m.size(),
        "path": m.path().display().to_string(),
        "loaded": false, // TEAM-384: UI expects this field
    })
}).collect();

if models.is_empty() {
    n!("model_list_empty", "No models found. Download a model with: ./rbee model download <model-id>");
} else {
    // TEAM-384: Emit table for CLI users (human-readable)
    n!("model_list_table", table: &models_json);
}

// TEAM-384: Emit JSON for UI (machine-readable) - UI expects this on last line before [DONE]
// Use narration (not println) so it appears in SSE stream in correct order
n!("model_list_json", "{}", serde_json::to_string(&models_json)?);

n!("model_list_complete", "✅ Model list operation complete");
```

## SSE Stream Output Order

**Correct order (after fix):**
```
1. model_list_start: "📋 Listing models on hive 'localhost'"
2. model_list_result: "Found 1 model(s)"
3. model_list_table: <table with columns>
4. model_list_json: [{"id":"...","loaded":false,...}]  ← UI parses this!
5. model_list_complete: "✅ Model list operation complete"
6. [DONE]
```

The UI searches backwards from `[DONE]`, finds line 4 (starts with `[`), parses it as JSON.

## Why Not `println!()`?

**Attempted (wrong):**
```rust
println!("{}", serde_json::to_string(&models_json)?);
```

**Problem:** `println!()` writes to stdout immediately, BEFORE narration events are emitted through the SSE system. This causes the JSON to appear out of order:

```
[{"id":"..."}]  ← Appears FIRST (stdout)
📋 Listing models...  ← Appears SECOND (SSE)
```

The UI would see the JSON before the narration, but the search logic expects it AFTER the table.

**Correct (using narration):**
```rust
n!("model_list_json", "{}", serde_json::to_string(&models_json)?);
```

This ensures the JSON appears in the SSE stream in the correct order.

## Key Fields for UI

The UI expects these fields in the JSON:
- `id` - Model identifier
- `name` - Display name
- `size` - File size in bytes
- `path` - Full filesystem path
- `loaded` - Boolean indicating if model is loaded in RAM

## Benefits

- ✅ CLI users see nice table formatting
- ✅ UI gets structured JSON data
- ✅ Both use the same backend endpoint
- ✅ Single source of truth for model data
- ✅ Consistent data flow through job-server

## Files Changed

- `bin/20_rbee_hive/src/job_router.rs` (+8 LOC, -6 LOC)
  - Added JSON emission after table
  - Added `loaded: false` field for UI compatibility
  - Used narration (not println) for correct SSE ordering

## Verification

**CLI output:**
```bash
$ ./rbee model ls
Found 1 model(s)
id                                                                          │ loaded │ name                          │ path                                                                                                                  │ size
────────────────────────────────────────────────────────────────────────────┼────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────
TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf │ false  │ TinyLlama-1.1B-Chat-v1.0-GGUF │ /home/vince/.cache/rbee/models/TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF-tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/model.gguf │ 668788096
✅ Model list operation complete
```

**UI should now show:**
- Downloaded (1) tab with the model listed
- Model details: name, size, path
- Load and Delete buttons available

## Pattern for Other Operations

This pattern should be applied to other list operations:
1. **Worker list** - Emit table + JSON
2. **Hive list** - Emit table + JSON
3. **Process list** - Emit table + JSON

**Template:**
```rust
// 1. Convert to JSON
let items_json: Vec<serde_json::Value> = items.iter().map(|item| {
    serde_json::json!({
        "field1": item.field1,
        "field2": item.field2,
        // ... UI-expected fields
    })
}).collect();

// 2. Emit table for CLI
if !items.is_empty() {
    n!("action_table", table: &items_json);
}

// 3. Emit JSON for UI
n!("action_json", "{}", serde_json::to_string(&items_json)?);
```

## Lessons Learned

1. **UI and CLI have different needs:** CLI wants human-readable, UI wants machine-readable
2. **Emit both formats:** Don't choose one over the other
3. **Use narration for ordering:** Don't mix `println!()` and narration - it breaks SSE ordering
4. **Document UI expectations:** The UI's JSON parsing logic is critical to understand
5. **Test both interfaces:** Changes to backend output affect both CLI and UI
