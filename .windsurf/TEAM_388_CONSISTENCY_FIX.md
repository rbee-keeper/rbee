# TEAM-388: Consistency Fix - Delete → Remove

**Status:** ✅ FIXED  
**Date:** Nov 2, 2025

## Problem

Inconsistent naming between model and worker handlers:
- Model: `delete` / `rm`
- Worker: `remove` / `rm`

**Unix convention** is `rm` (remove), not `delete`.

## Solution

Renamed `ModelAction::Delete` to `ModelAction::Remove` for consistency.

### File: `bin/00_rbee_keeper/src/handlers/model.rs`

**Before:**
```rust
pub enum ModelAction {
    Download { ... },
    List,
    Get { ... },
    Delete { id: Option<String> },  // ❌ Inconsistent
    Preload { ... },
    Unpreload { ... },
}
```

**After:**
```rust
pub enum ModelAction {
    Download { ... },
    List,
    Get { ... },
    Remove { id: Option<String> },  // ✅ Consistent with worker
    Preload { ... },
    Unpreload { ... },
}
```

## CLI Commands

### Before
```bash
./rbee model delete <id>    # ❌ Inconsistent
./rbee worker remove <id>   # Different verb
```

### After
```bash
./rbee model remove <id>    # ✅ Consistent
./rbee worker remove <id>   # Same verb
```

**Alias still works:**
```bash
./rbee model rm <id>        # ✅ Unix convention
./rbee worker rm <id>       # ✅ Unix convention
```

## Consistency Table

| Command | Model | Worker | Status |
|---------|-------|--------|--------|
| List | `list` / `ls` | `list` / `ls` | ✅ Consistent |
| Get | `get` / `show` | `get` / `show` | ✅ Consistent |
| Download | `download` / `dl` | `download` / `install` | ⚠️ Different (intentional) |
| Remove | `remove` / `rm` | `remove` / `rm` | ✅ Consistent |

**Note:** Download vs Install difference is intentional:
- Models are "downloaded" from HuggingFace
- Workers are "installed" from catalog (download + build + install)

## Benefits

1. **Consistency:** Same verb for same action
2. **Unix Convention:** `rm` is the standard Unix command
3. **Predictability:** Users know what to expect
4. **Clarity:** "Remove" is clearer than "Delete"

---

**TEAM-388 CONSISTENCY FIX COMPLETE** - Model and worker handlers now use consistent naming.
