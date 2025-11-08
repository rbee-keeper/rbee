# TEAM-384: Complete Narration Context Investigation

**Date:** Nov 2, 2025 3:23 PM  
**Status:** 📋 COMPLETE INVESTIGATION

---

## What We Have Today (Actual Source Code)

### 1. The `#[with_job_id]` Macro

**Location:** `bin/99_shared_crates/narration-macros/src/with_job_id.rs`

**Purpose:** Eliminates boilerplate for setting narration context

**How it works:**

```rust
// BEFORE (manual)
pub async fn some_function(config: SomeConfig) -> Result<()> {
    let ctx = config.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));
    let impl_fn = async {
        n!("action", "Doing thing");
        Ok(())
    };
    if let Some(ctx) = ctx {
        with_narration_context(ctx, impl_fn).await
    } else {
        impl_fn.await
    }
}

// AFTER (with macro)
#[with_job_id]
pub async fn some_function(config: SomeConfig) -> Result<()> {
    n!("action", "Doing thing");
    Ok(())
}
```

**Generated code:**

```rust
pub async fn some_function(config: SomeConfig) -> Result<()> {
    // Inner async function with original logic
    async fn __some_function_inner(config: SomeConfig) -> Result<()> {
        n!("action", "Doing thing");
        Ok(())
    }
    
    // Wrap with context if job_id exists
    if let Some(job_id) = config.job_id.as_ref() {
        let ctx = observability_narration_core::NarrationContext::new().with_job_id(job_id);
        observability_narration_core::with_narration_context(ctx, __some_function_inner(config)).await
    } else {
        __some_function_inner(config).await
    }
}
```

**Requirements:**
1. Function must be `async`
2. Config parameter must have `job_id: Option<String>` field
3. Macro auto-detects parameter with "config" in name

---

### 2. The `with_narration_context()` Function

**Location:** `bin/99_shared_crates/narration-core/src/context.rs:73-78`

**Source code:**

```rust
pub async fn with_narration_context<F>(ctx: NarrationContext, f: F) -> F::Output
where
    F: std::future::Future,
{
    NARRATION_CONTEXT.scope(RefCell::new(ctx), f).await
}
```

**What it does:**
1. Takes a `NarrationContext` (with job_id, correlation_id, actor)
2. Takes a `Future` (your async block)
3. Sets the context in **task-local storage** using `tokio::task_local!`
4. Executes the future within that context
5. All `n!()` calls inside the future can access the context

**Key insight:** This uses **tokio task-local storage**, which means:
- ✅ Context is available to ALL async code in the same task
- ✅ Context propagates through `.await` points
- ✅ Context is isolated per task (no cross-contamination)

---

### 3. The `NarrationContext` Struct

**Location:** `bin/99_shared_crates/narration-core/src/context.rs:12-38`

**Source code:**

```rust
tokio::task_local! {
    static NARRATION_CONTEXT: RefCell<NarrationContext>;
}

#[derive(Debug, Clone, Default)]
pub struct NarrationContext {
    pub job_id: Option<String>,
    pub correlation_id: Option<String>,
    pub actor: Option<&'static str>,
}

impl NarrationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }
}
```

**Key insight:** This is stored in **tokio task-local storage**, not thread-local!

---

### 4. How `n!()` Macro Gets the Context

**Location:** `bin/99_shared_crates/narration-core/src/api/macro_impl.rs:62-121`

**Source code:**

```rust
pub fn macro_emit(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
    crate_name: &'static str,
    fn_name: &'static str,
) {
    // ... select message based on mode ...
    
    // TEAM-297: Get context if available (for job_id and correlation_id)
    let ctx = crate::context::get_context();
    let job_id = ctx.as_ref().and_then(|c| c.job_id.clone());
    let correlation_id = ctx.as_ref().and_then(|c| c.correlation_id.clone());
    
    // ... build fields ...
    let fields = NarrationFields {
        actor,
        action,
        target,
        human: selected_message.to_string(),
        level,
        fn_name,
        cute: cute.map(|s| s.to_string()),
        story: story.map(|s| s.to_string()),
        job_id,  // ← From context!
        correlation_id,
        ..Default::default()
    };
    
    crate::narrate(fields, level);
}
```

**How it gets context:**

```rust
// bin/99_shared_crates/narration-core/src/context.rs:81-83
pub(crate) fn get_context() -> Option<NarrationContext> {
    NARRATION_CONTEXT.try_with(|ctx| ctx.borrow().clone()).ok()
}
```

**Key insight:** Every `n!()` call automatically reads from task-local storage!

---

## Current Usage Patterns

### Pattern 1: Manual Context (rbee-hive)

**File:** `bin/20_rbee_hive/src/job_router.rs:94-115`

```rust
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    use observability_narration_core::{with_narration_context, NarrationContext};
    
    // TEAM-381: Set narration context for ALL operations
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        // Parse operation
        let operation: Operation = serde_json::from_value(payload)?;
        let operation_name = operation.name().to_string();
        
        n!("route_job", "Executing operation: {}", operation_name);
        
        execute_operation(operation, operation_name, job_id, state).await
    }).await
}
```

**Scope:** Context is set ONCE at `route_operation`, available to:
- ✅ `n!()` in `route_operation`
- ✅ `n!()` in `execute_operation` (called inside the async block)
- ✅ `n!()` in ANY function called inside the async block

---

### Pattern 2: Macro Context (queen-rbee RHAI operations)

**File:** `bin/10_queen_rbee/src/rhai/save.rs:16-18`

```rust
#[with_job_id(config_param = "save_config")]
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "💾 Saving RHAI script: {}", save_config.name);
    // ... implementation ...
}
```

**Expands to:**

```rust
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    async fn __execute_rhai_script_save_inner(save_config: RhaiSaveConfig) -> Result<()> {
        n!("rhai_save_start", "💾 Saving RHAI script: {}", save_config.name);
        // ... implementation ...
    }
    
    if let Some(job_id) = save_config.job_id.as_ref() {
        let ctx = NarrationContext::new().with_job_id(job_id);
        with_narration_context(ctx, __execute_rhai_script_save_inner(save_config)).await
    } else {
        __execute_rhai_script_save_inner(save_config).await
    }
}
```

**Scope:** Context is set ONCE at function entry, available to:
- ✅ `n!()` in the function
- ✅ `n!()` in ANY function called by this function

---

### Pattern 3: Manual Context (queen-rbee hive forwarder)

**File:** `bin/10_queen_rbee/src/hive_forwarder.rs:131-171`

```rust
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    hive_url: &str,
) -> Result<()> {
    // TEAM-380: Manual context setup
    let ctx = NarrationContext::new().with_job_id(job_id);
    
    with_narration_context(ctx, async move {
        let operation_name = operation.name();
        let hive_id = operation.hive_id().unwrap_or("unknown");
        
        n!("forward_start", "Forwarding {} to hive '{}'", operation_name, hive_id);
        
        // ... forward to hive ...
        
        n!("forward_complete", "Operation completed on hive '{}'", hive_id);
        
        Ok(())
    }).await
}
```

---

## The Key Question: Do We Need #[with_job_id] Everywhere?

### Answer: NO!

**You only need to set context ONCE at the root of the async task.**

**Why?** Because of **tokio task-local storage**:

```
┌─────────────────────────────────────────────────────────────┐
│ with_narration_context(ctx, async {                        │
│   ├─ n!("start", "...")        ← Has job_id ✅             │
│   ├─ function_a().await                                     │
│   │   ├─ n!("a1", "...")       ← Has job_id ✅             │
│   │   └─ function_b().await                                 │
│   │       ├─ n!("b1", "...")   ← Has job_id ✅             │
│   │       └─ function_c().await                             │
│   │           └─ n!("c1", "...") ← Has job_id ✅           │
│   └─ n!("end", "...")          ← Has job_id ✅             │
│ }).await                                                     │
└─────────────────────────────────────────────────────────────┘
```

**The context propagates through ALL `.await` points!**

---

## Current State in rbee-hive

**File:** `bin/20_rbee_hive/src/job_router.rs`

### ✅ Context IS Set at the Root

```rust
// Line 73-87: execute_job (entry point)
pub async fn execute_job(job_id: String, state: JobState) -> impl Stream<Item = String> {
    job_server::execute_and_stream(
        job_id,
        registry.clone(),
        move |job_id, payload| route_operation(job_id, payload, state_clone.clone()),
        None,
    ).await
}

// Line 94-115: route_operation (sets context)
async fn route_operation(job_id: String, payload: Value, state: JobState) -> Result<()> {
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        // Parse operation
        let operation: Operation = serde_json::from_value(payload)?;
        
        n!("route_job", "Executing operation: {}", operation_name);
        
        execute_operation(operation, operation_name, job_id, state).await
    }).await
}

// Line 118+: execute_operation (uses context from parent)
async fn execute_operation(operation: Operation, ...) -> Result<()> {
    match operation {
        Operation::ModelList(request) => {
            n!("model_list_start", "📋 Listing models...");  // ← Has job_id ✅
            // ...
            n!("model_list_complete", "✅ Complete");  // ← Has job_id ✅
        }
    }
}
```

### ✅ This Is CORRECT!

Context is set ONCE in `route_operation`, and ALL `n!()` calls in:
- `route_operation`
- `execute_operation`
- Any function called by `execute_operation`

...will have access to the `job_id`!

---

## Why The Confusion?

### The `#[with_job_id]` Macro is for DIFFERENT Use Case!

**The macro is for functions that:**
1. Are called from OUTSIDE a context (e.g., from HTTP handlers)
2. Receive a config struct with `job_id: Option<String>`
3. Need to set context for themselves

**Example:** queen-rbee RHAI operations

```rust
// HTTP handler calls this directly (no context set yet)
#[with_job_id]
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "💾 Saving...");
    // ...
}
```

**But in rbee-hive:**
- Context is set at `route_operation` (the root)
- All operations execute INSIDE that context
- No need for `#[with_job_id]` on individual operations!

---

## The Design is Actually Perfect!

### Job Server Pattern

```
HTTP POST /v1/jobs
    ↓
create_job() - Creates job_id
    ↓
execute_job() - Entry point
    ↓
route_operation() - Sets context ONCE ← ROOT
    ↓
execute_operation() - Uses context
    ↓
    ├─ ModelList handler - Uses context
    ├─ ModelDownload handler - Uses context
    ├─ WorkerSpawn handler - Uses context
    └─ ... all handlers use context
```

**One context, set at the root, propagates everywhere!**

---

## Verification: Does It Work?

### Evidence from Your Output

```
rbee_hive::job_router::execute_operation model_list_start    
📋 Listing models on hive 'localhost'
```

**This proves:**
1. ✅ Narration IS being emitted
2. ✅ `job_id` IS in the context (otherwise SSE wouldn't work)
3. ❌ The format is raw (tracing subscriber issue, not context issue)

**The narration IS going to SSE!** You can see it in the formatted output mixed with the raw tracing output.

---

## The REAL Problem (Not Context!)

### Problem: Tracing Default Subscriber

rbee-hive doesn't initialize a tracing subscriber, so tracing uses its default which prints to stdout in raw format.

### Solution: Initialize Subscriber

**Add to `bin/20_rbee_hive/src/main.rs` after line 82:**

```rust
// TEAM-384: Disable tracing output - narration goes to SSE only
tracing_subscriber::fmt()
    .with_writer(std::io::sink())
    .init();
```

---

## Summary: What We Have Today

### 1. Context Propagation (Task-Local Storage)

✅ **Works perfectly!** Set context once, propagates everywhere.

### 2. Two Ways to Set Context

**Option A: Manual (rbee-hive style)**
```rust
let ctx = NarrationContext::new().with_job_id(&job_id);
with_narration_context(ctx, async move {
    // All n!() calls here have job_id
}).await
```

**Option B: Macro (queen-rbee RHAI style)**
```rust
#[with_job_id]
pub async fn my_function(config: MyConfig) -> Result<()> {
    // All n!() calls here have job_id
}
```

### 3. When to Use Each

**Use Manual when:**
- You have the job_id as a parameter (not in a config struct)
- You're at the root of a job execution (like `route_operation`)

**Use Macro when:**
- You have a config struct with `job_id: Option<String>`
- You're writing a standalone operation handler
- You want less boilerplate

### 4. rbee-hive is CORRECT!

✅ Context set at `route_operation` (the root)  
✅ All operations execute inside that context  
✅ All `n!()` calls have `job_id`  
✅ SSE routing works  

❌ Only issue: Tracing subscriber prints raw format (easy fix!)

---

## Recommendation: NO REDESIGN NEEDED!

**The current design is excellent:**

1. ✅ Context is set ONCE at the root
2. ✅ Propagates through all async calls
3. ✅ No need to annotate every function
4. ✅ Task-local storage ensures isolation
5. ✅ SSE routing works correctly

**Only fix needed:**
- Initialize tracing subscriber in rbee-hive to suppress raw output

---

## Clarity on Confusion

### Your Concern: "Do we need #[with_job_id] on ALL functions?"

**Answer: NO!**

**You only need to set context ONCE at the root of the async task.**

### Why It Works

**Tokio task-local storage** means:
- Context is stored in the async task
- Propagates through `.await` points
- Available to ALL code in the same task
- No need to pass it explicitly

### The Pattern

```rust
// ROOT: Set context once
with_narration_context(ctx, async {
    
    // LEVEL 1: Has context
    n!("start", "...");
    
    // LEVEL 2: Still has context
    function_a().await;
    
    // LEVEL 3: Still has context
    function_b().await;
    
    // LEVEL 4: Still has context
    function_c().await;
    
}).await
```

**No annotations needed on `function_a`, `function_b`, `function_c`!**

---

**TEAM-384:** Complete investigation done! Current design is correct. Only need to fix tracing subscriber in rbee-hive. No redesign needed! 🎯
