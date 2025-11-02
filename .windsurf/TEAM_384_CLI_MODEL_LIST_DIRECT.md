# TEAM-384: CLI Model List Direct to Hive

**Date:** Nov 2, 2025 2:00 PM  
**Status:** 🔧 IMPLEMENTATION READY

---

## Goal

Add a `--direct` flag to `rbee model list` command to bypass queen and connect directly to rbee-hive.

**Why:** Debugging the Model List timeout issue - we want to test if the job server works when called directly (without going through queen-rbee).

---

## Current Flow (Via Queen)

```
User: ./rbee model list

./rbee
  ↓
xtask rbee
  ↓
rbee-keeper (CLI)
  ↓
handle_model()
  ↓
submit_and_stream_job(queen_url, operation)
  ↓
POST http://localhost:7833/v1/jobs (queen-rbee)
  ↓
queen forwards to hive
  ↓
http://localhost:7835/v1/jobs (rbee-hive)
```

---

## New Flow (Direct to Hive)

```
User: ./rbee model list --direct

./rbee
  ↓
xtask rbee
  ↓
rbee-keeper (CLI)
  ↓
handle_model() [checks --direct flag]
  ↓
submit_and_stream_job_to_hive(hive_url, operation)
  ↓
POST http://localhost:7835/v1/jobs (rbee-hive DIRECTLY)
```

**Benefits:**
- Bypasses queen (one less hop)
- Tests if hive job server works in isolation
- Faster debugging
- Can diagnose if issue is in queen forwarding or hive execution

---

## Implementation

### Step 1: Add --direct Flag to CLI

**File:** `bin/00_rbee_keeper/src/cli/commands.rs`

```rust
/// Model management
Model {
    /// Hive alias to operate on (defaults to localhost)
    #[arg(long = "hive", default_value = "localhost")]
    hive_id: String,
    
    /// Connect directly to hive (skip queen)
    /// TEAM-384: For debugging - bypasses queen-rbee
    #[arg(long, default_value = "false")]
    direct: bool,
    
    #[command(subcommand)]
    action: ModelAction,
},
```

### Step 2: Update main.rs to Pass Flag

**File:** `bin/00_rbee_keeper/src/main.rs` (line 159)

```rust
Commands::Model { hive_id, direct, action } => {
    handle_model(hive_id, action, &queen_url, direct).await
}
```

### Step 3: Update handle_model to Support Direct Mode

**File:** `bin/00_rbee_keeper/src/handlers/model.rs`

```rust
use crate::job_client::{submit_and_stream_job, submit_and_stream_job_to_hive};
use crate::handlers::hive_jobs::get_hive_url;

pub async fn handle_model(
    hive_id: String, 
    action: ModelAction, 
    queen_url: &str,
    direct: bool,  // TEAM-384: New parameter
) -> Result<()> {
    // Build operation
    let operation = match &action {
        ModelAction::Download { model } => {
            Operation::ModelDownload(ModelDownloadRequest { 
                hive_id: hive_id.clone(), 
                model: model.clone() 
            })
        }
        ModelAction::List => Operation::ModelList(ModelListRequest { 
            hive_id: hive_id.clone() 
        }),
        ModelAction::Get { id } => Operation::ModelGet(ModelGetRequest { 
            hive_id: hive_id.clone(), 
            id: id.clone() 
        }),
        ModelAction::Delete { id } => {
            Operation::ModelDelete(ModelDeleteRequest { 
                hive_id: hive_id.clone(), 
                id: id.clone() 
            })
        }
    };
    
    // TEAM-384: Choose connection method based on --direct flag
    if direct {
        // Direct to hive (skip queen)
        let hive_url = get_hive_url(&hive_id);
        println!("🔍 [DEBUG] Connecting directly to hive: {}", hive_url);
        submit_and_stream_job_to_hive(&hive_url, operation).await
    } else {
        // Via queen (default)
        submit_and_stream_job(queen_url, operation).await
    }
}
```

### Step 4: Update lib.rs Export

**File:** `bin/00_rbee_keeper/src/lib.rs` (if needed)

No changes needed - signature stays the same externally.

---

## Usage

### Test Direct Connection

```bash
# Direct to hive (bypass queen)
./rbee model list --direct

# Or with explicit hive
./rbee model --hive localhost list --direct
```

### Compare Against Queen

```bash
# Via queen (default)
./rbee model list

# Direct to hive
./rbee model list --direct
```

### Expected Output (Success)

```
📋 Job submitted: ModelList
📡 Streaming results for ModelList
rbee_hive::job_router::route_operation route_job               
Executing operation: ModelList
rbee_hive::job_router::execute_operation model_list_start        
📋 Listing models on hive 'localhost'
rbee_hive::job_router::execute_operation model_list_result       
Found 0 model(s)
rbee_hive::job_router::execute_operation model_list_json         
[]
rbee_hive::job_router::execute_operation model_list_complete     
✅ Model list operation complete
[DONE]
✅ Complete: ModelList
```

### Expected Output (Failure - For Debugging)

```
📋 Job submitted: ModelList
📡 Streaming results for ModelList
[NO OUTPUT - HANGS FOR 30 SECONDS]
Error: Operation timed out after 30 seconds
```

---

## Files to Modify

1. **`bin/00_rbee_keeper/src/cli/commands.rs`**
   - Add `direct: bool` field to `Commands::Model`
   
2. **`bin/00_rbee_keeper/src/main.rs`**
   - Extract `direct` from pattern match
   - Pass to `handle_model()`
   
3. **`bin/00_rbee_keeper/src/handlers/model.rs`**
   - Add `direct: bool` parameter
   - Add import for `submit_and_stream_job_to_hive` and `get_hive_url`
   - Add if/else to choose connection method

---

## Testing Plan

### Test 1: Direct Connection Works

```bash
# Start hive (if not running)
./rbee hive start

# Test direct connection
./rbee model list --direct

# Expected: Empty array [] and [DONE]
```

### Test 2: Compare Direct vs Queen

```bash
# Via queen
time ./rbee model list

# Direct
time ./rbee model list --direct

# Compare:
# - Does direct work when queen fails?
# - Is direct faster?
# - Do they return same data?
```

### Test 3: With Downloaded Model

```bash
# Download a model
./rbee model download --model "meta-llama/Llama-3.2-1B"

# List via queen
./rbee model list

# List direct
./rbee model list --direct

# Both should show the same model
```

---

## Debugging Strategy

### If Direct Works But Queen Fails

**Problem:** Queen forwarding is broken  
**Fix:** Check `queen-rbee/src/hive_forwarder.rs`

### If Both Fail

**Problem:** Hive job execution is broken  
**Fix:** Add debug logging to `rbee-hive/src/job_router.rs`

### If Direct Hangs But Queen Works

**Problem:** URL resolution or network issue  
**Fix:** Check `get_hive_url()` returns correct URL

---

## Code Locations

### Job Client (Already Supports Both)

**File:** `bin/00_rbee_keeper/src/job_client.rs`

```rust
// Already exists! No changes needed

/// Submit job to queen (line 34)
pub async fn submit_and_stream_job(target_url: &str, operation: Operation) -> Result<()>

/// Submit job directly to hive (line 58)
pub async fn submit_and_stream_job_to_hive(hive_url: &str, operation: Operation) -> Result<()>
```

### Hive URL Resolution

**File:** `bin/00_rbee_keeper/src/handlers/hive_jobs.rs` (line 107)

```rust
pub fn get_hive_url(alias: &str) -> String {
    if alias == "localhost" {
        "http://localhost:7835".to_string()
    } else {
        // Future: Read from hives.conf
        panic!("Only localhost is supported")
    }
}
```

---

## Example Session

```bash
# Terminal 1: Start hive
./rbee hive start

# Terminal 2: Test model list
./rbee model list --direct

# Output:
# 📋 Job submitted: ModelList
# 📡 Streaming results for ModelList
# 🔍 [DEBUG] Connecting directly to hive: http://localhost:7835
# [... narration lines ...]
# []
# [DONE]
# ✅ Complete: ModelList
```

---

## Summary

**3 small changes:**
1. Add `direct: bool` flag to CLI
2. Pass flag through main.rs
3. Choose connection method in handle_model()

**Total LOC:** ~15 lines of code

**Benefit:** Can test if Model List works when bypassing queen, helping isolate the timeout bug.

---

**TEAM-384:** Ready to implement! This will help debug the Model List timeout issue.
