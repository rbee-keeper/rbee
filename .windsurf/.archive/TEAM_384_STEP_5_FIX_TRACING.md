# TEAM-384 Step 5: Fix Tracing Output

**Status:** ⏳ PENDING  
**Dependencies:** Step 4 complete (daemon-lifecycle fixed, everything compiles)  
**Estimated Time:** 5 minutes

---

## Goal

Initialize tracing subscriber in rbee-hive to suppress raw format output. Narration should only appear via SSE (formatted), not stdout (raw).

---

## The Problem

**Current output:**
```
rbee_hive::job_router::execute_operation model_list_start    
📋 Listing models on hive 'localhost'
rbee_hive::job_router::execute_operation model_list_result   
Found 0 model(s)
```

**Why this happens:**
- Tracing has a **default subscriber** that prints to stdout when no subscriber is initialized
- The default format is: `<target> <action> <human>`
- This is the raw tracing event format

**What we want:**
- Narration goes to SSE only (formatted)
- No stdout output from rbee-hive (it's a daemon)

---

## File to Modify

**File:** `bin/20_rbee_hive/src/main.rs`

**Line:** After startup narration (~82)

---

## Current Code

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", build::BUILD_RUST_CHANNEL);
        std::process::exit(0);
    }

    // TEAM-340: Migrated to n!() macro
    n!("startup", "🐝 Starting rbee-hive on port {}", args.port);

    // TEAM-261: Initialize job registry for dual-call pattern
    let job_registry: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());

    // ... rest of initialization ...
}
```

---

## New Code

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", build::BUILD_RUST_CHANNEL);
        std::process::exit(0);
    }

    // TEAM-384: Initialize tracing subscriber to suppress raw output
    // rbee-hive is a daemon - narration goes to SSE only, not stdout
    // This prevents the default tracing subscriber from printing raw events
    tracing_subscriber::fmt()
        .with_writer(std::io::sink())  // Discard all output
        .init();

    // TEAM-340: Migrated to n!() macro
    n!("startup", "🐝 Starting rbee-hive on port {}", args.port);

    // TEAM-261: Initialize job registry for dual-call pattern
    let job_registry: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());

    // ... rest of initialization ...
}
```

---

## Add Import

**File:** `bin/20_rbee_hive/src/main.rs`

**Add to imports (top of file):**

```rust
use tracing_subscriber;  // TEAM-384: For suppressing tracing output
```

---

## Alternative: Filter Narration Events

If you want to keep tracing for debugging but filter out narration:

```rust
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

// TEAM-384: Initialize tracing with narration filtered out
tracing_subscriber::registry()
    .with(EnvFilter::new("info,observability_narration_core=off"))
    .with(fmt::layer())
    .init();
```

**This allows:**
- ✅ Other tracing events (errors, warnings) go to stdout
- ❌ Narration events are filtered out (go to SSE only)

---

## Recommended Approach

**Use `sink()` writer** (simplest):

```rust
tracing_subscriber::fmt()
    .with_writer(std::io::sink())
    .init();
```

**Why?**
- rbee-hive is a daemon (no human watching stdout)
- Narration goes to SSE (clients see it)
- Errors/panics still go to stderr (Rust default)
- Simplest solution

---

## Verification

### Test Model List

```bash
# Terminal 1: Start rbee-hive
cargo run --bin rbee-hive

# Terminal 2: Test model list
./rbee model list
```

**Expected Output (Clean!):**

```
📋 Job submitted: model_list
⏱️  Streaming job results (timeout: 30s)
📡 Streaming results for model_list
📋 Listing models on hive 'localhost'
Found 0 model(s)
[]
✅ Model list operation complete
[DONE]
✅ Complete: model_list
```

**Should NOT see:**
```
rbee_hive::job_router::execute_operation model_list_start    
```

---

## What This Achieves

### Before (Raw Tracing Output)

```
# Terminal output when running ./rbee model list
2025-11-02T14:13:21.138753Z  INFO actor="rbee_keeper" action="job_submit" target=job_submit human=📋 Job submitted: model_list
rbee_hive::job_router::execute_operation model_list_start    
📋 Listing models on hive 'localhost'
rbee_hive::job_router::execute_operation model_list_result   
Found 0 model(s)
rbee_hive::job_router::execute_operation model_list_json     
[]
rbee_hive::job_router::execute_operation model_list_complete 
✅ Model list operation complete
[DONE]
2025-11-02T14:13:21.205877Z  INFO actor="rbee_keeper" action="job_complete" target=job_complete human=✅ Complete: model_list
```

**Problems:**
- ❌ Mixed formats (formatted + raw)
- ❌ Confusing output
- ❌ Raw tracing leaking to client

### After (Clean SSE Output)

```
# Terminal output when running ./rbee model list
📋 Job submitted: model_list
⏱️  Streaming job results (timeout: 30s)
📡 Streaming results for model_list
📋 Listing models on hive 'localhost'
Found 0 model(s)
[]
✅ Model list operation complete
[DONE]
✅ Complete: model_list
```

**Benefits:**
- ✅ Clean, formatted output
- ✅ Consistent format
- ✅ No raw tracing leaks
- ✅ Professional appearance

---

## Two-Tier Narration in Action

### Tier 1: Job Narration (Has job_id)

**Code:**
```rust
// Inside job execution (rbee-hive)
n!("model_list_start", "📋 Listing models...");
```

**Behavior:**
- ✅ Has `job_id` (from job-server context)
- ✅ Goes to SSE channel
- ✅ Client sees it via SSE stream
- ❌ Does NOT go to stdout (tracing suppressed)

**Client sees:**
```
📋 Listing models...
```

### Tier 2: Daemon Narration (No job_id)

**Code:**
```rust
// Daemon startup (rbee-hive main.rs)
n!("startup", "🐝 Starting rbee-hive on port 7835");
```

**Behavior:**
- ❌ No `job_id` (not in job context)
- ❌ SSE sink rejects it (no job_id to route to)
- ✅ Would go to tracing (but we suppressed it)
- ❌ Does NOT appear anywhere (daemon log)

**Note:** Startup narration won't appear with `sink()` writer. If you want daemon logs, use the filter approach instead.

---

## If You Want Daemon Logs

**Use filter instead of sink:**

```rust
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

// TEAM-384: Filter out narration events, keep other logs
tracing_subscriber::registry()
    .with(EnvFilter::new("info,observability_narration_core=off"))
    .with(fmt::layer())
    .init();
```

**This gives you:**
- ✅ Startup logs visible
- ✅ Error logs visible
- ❌ Narration events filtered out (go to SSE only)

---

## Summary of Changes

### Files Modified: 1

1. ✅ `bin/20_rbee_hive/src/main.rs` - Added tracing subscriber init

### Lines Added: 4

```rust
tracing_subscriber::fmt()
    .with_writer(std::io::sink())
    .init();
```

### Benefits

- ✅ Clean output (no raw tracing)
- ✅ Professional appearance
- ✅ SSE-only narration
- ✅ No confusion

---

## Next Step

**Step 6:** Verify everything works end-to-end

**File:** `TEAM_384_STEP_6_VERIFICATION.md`

---

**TEAM-384:** Clean output! No more raw tracing! 🎯
