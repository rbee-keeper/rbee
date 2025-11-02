# Complete Code Flow: `rbee model list`

**Date:** Nov 2, 2025 3:08 PM  
**Traced by:** TEAM-384

---

## User Command

```bash
rbee model list
# Equivalent to: rbee model list --hive localhost
```

---

## Complete Flow (Step by Step)

### Step 1: CLI Entry Point

**File:** `bin/00_rbee_keeper/src/main.rs:159`

```rust
Commands::Model { hive_id, action } => handle_model(hive_id, action).await,
```

**Values:**
- `hive_id` = `"localhost"` (from `commands.rs:71` default)
- `action` = `ModelAction::List`

---

### Step 2: Model Handler

**File:** `bin/00_rbee_keeper/src/handlers/model.rs:63-105`

```rust
pub async fn handle_model(hive_id: String, action: ModelAction) -> Result<()> {
    // Build operation
    let operation = match &action {
        ModelAction::List => {
            Operation::ModelList(ModelListRequest { 
                hive_id: hive_id.clone()  // "localhost"
            })
        }
        // ... other actions
    };
    
    // TEAM-384: Connect directly to hive (no queen)
    let hive_url = get_hive_url(&hive_id);  // → "http://localhost:7835"
    submit_and_stream_job_to_hive(&hive_url, operation).await
}
```

**Output:**
- `operation` = `Operation::ModelList(ModelListRequest { hive_id: "localhost" })`
- `hive_url` = `"http://localhost:7835"`

---

### Step 3: Resolve Hive URL

**File:** `bin/00_rbee_keeper/src/handlers/hive_jobs.rs:122-137`

```rust
pub fn get_hive_url(alias: &str) -> String {
    use crate::ssh_resolver::resolve_ssh_config;
    
    if alias == "localhost" {
        return "http://localhost:7835".to_string();  // ← Takes this path
    }
    
    // For remote hives, resolve via SSH config
    match resolve_ssh_config(alias) {
        Ok(ssh) => format!("http://{}:7835", ssh.hostname),
        Err(_) => format!("http://{}:7835", alias)
    }
}
```

**Output:** `"http://localhost:7835"`

---

### Step 4: Submit Job (Job Client)

**File:** `bin/00_rbee_keeper/src/job_client.rs:58-60`

```rust
pub async fn submit_and_stream_job_to_hive(hive_url: &str, operation: Operation) -> Result<()> {
    submit_and_stream_job(hive_url, operation).await
}
```

**File:** `bin/00_rbee_keeper/src/job_client.rs:65-104`

```rust
async fn stream_job_results(
    queen_url: &str,  // Actually hive_url in this case
    operation: Operation,
    operation_name: &'static str,
    _hive_id: Option<String>,
) -> Result<()> {
    n!("job_stream", "📡 Streaming results for {}", operation_name);
    
    // TEAM-259: Use shared JobClient
    let job_client = JobClient::new(queen_url);  // "http://localhost:7835"
    
    job_client
        .submit_and_stream(operation, |line| {
            println!("{}", line);  // Print narration to stdout
            
            if line.contains("[DONE]") {
                n!("job_complete", "✅ Complete: {}", operation_name);
            }
            
            Ok(())
        })
        .await?;
    
    Ok(())
}
```

**What happens:**
1. Creates `JobClient` pointing to `http://localhost:7835`
2. Calls `submit_and_stream()` with operation and line handler

---

### Step 5: JobClient - POST /v1/jobs

**File:** `bin/99_shared_crates/job-client/src/lib.rs:86-112`

```rust
pub async fn submit_and_stream<F>(
    &self,
    operation: Operation,
    mut line_handler: F,
) -> Result<String>
where
    F: FnMut(&str) -> Result<()>,
{
    // 1. Serialize operation to JSON
    let payload = serde_json::to_value(&operation)?;
    
    // 2. POST to /v1/jobs endpoint
    let job_response: JobResponse = self
        .client
        .post(format!("{}/v1/jobs", self.base_url))  // POST http://localhost:7835/v1/jobs
        .json(&payload)  // Body: {"type":"ModelList","hive_id":"localhost"}
        .send()
        .await?
        .json()
        .await?;
    
    // 3. Extract job_id from response
    let job_id = job_response.job_id;  // e.g., "job-abc123"
    
    // 4. Connect to SSE stream
    let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
    // → http://localhost:7835/v1/jobs/job-abc123/stream
    
    // ... continues to streaming
}
```

**HTTP Request:**
```http
POST http://localhost:7835/v1/jobs
Content-Type: application/json

{
  "type": "ModelList",
  "hive_id": "localhost"
}
```

**HTTP Response:**
```json
{
  "job_id": "job-abc123",
  "sse_url": "/v1/jobs/job-abc123/stream"
}
```

---

### Step 6: Hive - Create Job (Job Server)

**File:** `bin/20_rbee_hive/src/http/jobs.rs:54-63`

```rust
pub async fn handle_create_job(
    State(state): State<HiveState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<JobResponse>, (StatusCode, String)> {
    // Delegate to router
    crate::job_router::create_job(state.into(), payload)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
```

**File:** `bin/20_rbee_hive/src/job_router.rs:54-70`

```rust
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();  // Generate UUID
    let sse_url = format!("/v1/jobs/{}/stream", job_id);
    
    state.registry.set_payload(&job_id, payload);  // Store operation
    
    // TEAM-200: Create job-specific SSE channel
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);
    
    Ok(JobResponse {
        job_id: job_id.clone(),
        sse_url,
    })
}
```

**What happens:**
1. Generate job ID: `"job-abc123"`
2. Store operation payload in registry
3. Create SSE channel for this job (MPSC channel with 1000 buffer)
4. Return `JobResponse` to client

---

### Step 7: JobClient - GET /v1/jobs/{job_id}/stream

**File:** `bin/99_shared_crates/job-client/src/lib.rs:114-180`

```rust
// 4. Connect to SSE stream
let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
let response = self
    .client
    .get(&stream_url)  // GET http://localhost:7835/v1/jobs/job-abc123/stream
    .send()
    .await?;

// 5. Stream bytes and process lines incrementally
let mut stream = response.bytes_stream();
let mut buffer = Vec::new();

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    buffer.extend_from_slice(&chunk);
    
    // Parse complete lines
    while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
        let line_bytes = buffer.drain(..=newline_pos).collect::<Vec<_>>();
        
        if let Ok(line) = std::str::from_utf8(&line_bytes) {
            let line = line.trim();
            
            // Strip "data:" or "data: " prefix (SSE format)
            let data = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
                .unwrap_or(line);
            
            if data.is_empty() {
                continue;
            }
            
            // Call handler for each line
            line_handler(data)?;  // Prints to stdout
            
            // Check for completion markers
            if completion_markers::is_completion_marker(data) {
                if data == completion_markers::DONE {
                    return Ok(job_id);  // ← Exit successfully
                }
                // ... handle ERROR/CANCELLED
            }
        }
    }
}
```

**HTTP Request:**
```http
GET http://localhost:7835/v1/jobs/job-abc123/stream
Accept: text/event-stream
```

---

### Step 8: Hive - Stream Job (Job Server)

**File:** `bin/20_rbee_hive/src/http/jobs.rs:103-155`

```rust
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<HiveState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Take the SSE receiver (can only be done once per job)
    let sse_rx_opt = sse_sink::take_job_receiver(&job_id);
    
    let registry = state.registry.clone();
    
    // Trigger job execution (spawns in background)
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;
    
    // Stream narration events
    let combined_stream = async_stream::stream! {
        let Some(mut sse_rx) = sse_rx_opt else {
            yield Ok(Event::default().data("ERROR: Job channel not found"));
            yield Ok(Event::default().data("[DONE]"));
            return;
        };
        
        // Stream ALL narration events from SSE channel
        while let Some(event) = sse_rx.recv().await {
            yield Ok(Event::default().data(&event.formatted));
        }
        
        // SSE channel closed - send completion signal
        let state = registry.get_job_state(&job_id);
        let signal = match state {
            Some(JobState::Failed(err)) => format!("[ERROR] {}", err),
            Some(JobState::Cancelled) => "[CANCELLED]".to_string(),
            _ => "[DONE]".to_string(),
        };
        yield Ok(Event::default().data(&signal));
        
        // Cleanup
        sse_sink::remove_job_channel(&job_id);
    };
    
    Sse::new(combined_stream)
}
```

**What happens:**
1. Takes SSE receiver from channel (MPSC receiver)
2. Spawns job execution in background
3. Streams narration events as they arrive
4. When channel closes, sends `[DONE]` marker
5. Cleans up channel

---

### Step 9: Hive - Execute Job

**File:** `bin/20_rbee_hive/src/job_router.rs:73-551`

```rust
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    // ... setup ...
    
    // Parse operation from payload
    let operation: Operation = serde_json::from_value(payload)?;
    
    // Match on operation type
    match operation {
        Operation::ModelList(request) => {
            let hive_id = request.hive_id.clone();
            
            // TEAM-268: Implemented model list
            n!("model_list_start", "📋 Listing models on hive '{}'", hive_id);
            
            let models = state.model_catalog.list();
            
            n!("model_list_result", "Found {} model(s)", models.len());
            
            // Emit JSON response for frontend consumption
            let json = serde_json::to_string(&models)
                .unwrap_or_else(|_| "[]".to_string());
            n!("model_list_json", "{}", json);
            
            n!("model_list_complete", "✅ Model list operation complete");
        }
        // ... other operations
    }
    
    // Mark job as complete
    state.registry.complete_job(&job_id);
}
```

**What happens:**
1. Deserializes operation from payload
2. Matches on `Operation::ModelList`
3. Calls `model_catalog.list()` to get models
4. Emits narration events via `n!()` macro
5. Marks job as complete

---

### Step 10: Narration System

**File:** `bin/99_shared_crates/narration-core/src/lib.rs:261-270`

```rust
#[macro_export]
macro_rules! n {
    ($action:expr, $msg:expr) => {{
        $crate::macro_emit(
            $action,
            $msg,
            None,
            None,
            env!("CARGO_CRATE_NAME"),
            stdext::function_name!()
        );
    }};
}
```

**What `n!()` does:**
1. Creates narration event with action and message
2. Captures crate name and function name automatically
3. Formats the event as text
4. Sends to SSE channel (if job_id is set in context)
5. Also prints to stdout (for debugging)

**Example narration events:**
```
📋 Listing models on hive 'localhost'
Found 2 model(s)
[{"id":"meta-llama/Llama-3.2-1B","size":"2.5GB"}]
✅ Model list operation complete
```

---

### Step 11: SSE Streaming

**SSE Format (Server → Client):**

```
data: 📋 Listing models on hive 'localhost'

data: Found 2 model(s)

data: [{"id":"meta-llama/Llama-3.2-1B","size":"2.5GB"}]

data: ✅ Model list operation complete

data: [DONE]

```

**Client Processing:**
1. Receives SSE chunks
2. Strips `data:` prefix
3. Calls line handler (prints to stdout)
4. Detects `[DONE]` marker
5. Returns successfully

---

### Step 12: Output to User

**Terminal Output:**

```bash
$ rbee model list

📡 Streaming results for model_list
📋 Listing models on hive 'localhost'
Found 2 model(s)
[
  {
    "id": "meta-llama/Llama-3.2-1B",
    "size": "2.5GB",
    "status": "downloaded"
  }
]
✅ Model list operation complete
✅ Complete: model_list
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ USER: rbee model list                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: CLI Entry (main.rs)                                    │
│   Commands::Model { hive_id: "localhost", action: List }       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Model Handler (handlers/model.rs)                      │
│   operation = Operation::ModelList(...)                        │
│   hive_url = get_hive_url("localhost")                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Resolve URL (handlers/hive_jobs.rs)                    │
│   "localhost" → "http://localhost:7835"                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Job Client (job_client.rs)                             │
│   JobClient::new("http://localhost:7835")                      │
│   submit_and_stream(operation, line_handler)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: POST /v1/jobs (job-client crate)                       │
│   HTTP POST http://localhost:7835/v1/jobs                      │
│   Body: {"type":"ModelList","hive_id":"localhost"}             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Create Job (rbee-hive/http/jobs.rs)                    │
│   job_id = "job-abc123"                                         │
│   Create SSE channel                                            │
│   Return JobResponse                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: GET /v1/jobs/{id}/stream (job-client crate)            │
│   HTTP GET http://localhost:7835/v1/jobs/job-abc123/stream     │
│   Accept: text/event-stream                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: Stream Job (rbee-hive/http/jobs.rs)                    │
│   Take SSE receiver                                             │
│   Spawn execute_job() in background                             │
│   Stream narration events                                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: Execute Job (rbee-hive/job_router.rs)                  │
│   Parse Operation::ModelList                                    │
│   Call model_catalog.list()                                     │
│   Emit narration via n!() macro                                 │
│   Mark job complete                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 10: Narration (narration-core)                            │
│   n!("model_list_start", "📋 Listing models...")               │
│   → Sends to SSE channel                                        │
│   → Prints to stdout                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 11: SSE Stream (HTTP)                                     │
│   data: 📋 Listing models on hive 'localhost'                  │
│   data: Found 2 model(s)                                        │
│   data: [{"id":"..."}]                                          │
│   data: ✅ Model list operation complete                       │
│   data: [DONE]                                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 12: Output (Terminal)                                     │
│   Prints each line to stdout                                    │
│   Detects [DONE] marker                                         │
│   Exits successfully                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### Job Client (rbee-keeper side)

**Crate:** `bin/99_shared_crates/job-client`

**Responsibilities:**
1. POST operation to `/v1/jobs`
2. Parse `JobResponse` (job_id, sse_url)
3. GET SSE stream
4. Parse SSE events (strip `data:` prefix)
5. Call line handler for each event
6. Detect `[DONE]` marker
7. Return job_id

### Job Server (rbee-hive side)

**Crate:** `bin/99_shared_crates/job-server`

**Responsibilities:**
1. Create job with UUID
2. Store operation payload
3. Create SSE channel (MPSC)
4. Return `JobResponse`
5. Execute job in background
6. Stream narration to SSE channel
7. Send `[DONE]` when complete
8. Cleanup channel

### Narration System

**Crate:** `bin/99_shared_crates/narration-core`

**Responsibilities:**
1. `n!()` macro for easy narration
2. Format events as text
3. Send to SSE channel (if job_id set)
4. Also print to stdout (debugging)
5. Thread-safe (uses global broadcaster)

---

## Contract Enforcement

### Jobs Contract

**Crate:** `bin/97_contracts/jobs-contract`

**Provides:**
- `JobResponse` struct (job_id, sse_url)
- `completion_markers::DONE` = `"[DONE]"`
- `completion_markers::ERROR_PREFIX` = `"[ERROR]"`
- `completion_markers::CANCELLED` = `"[CANCELLED]"`

**Used by:**
- ✅ job-client (parses JobResponse, checks markers)
- ✅ job-server (returns JobResponse, sends markers)
- ✅ rbee-hive (uses job-server)
- ✅ rbee-keeper (uses job-client)

---

## Summary

**Flow:** CLI → Model Handler → Job Client → HTTP POST → Hive Create Job → HTTP GET → Hive Stream Job → Execute Job → Narration → SSE Stream → Job Client → Terminal Output

**Pattern:** Job Client → Job Server (with SSE streaming)

**Contract:** `jobs-contract` ensures type safety

**Narration:** `n!()` macro sends events to SSE channel

**Completion:** `[DONE]` marker signals success

**Result:** User sees real-time narration and model list! 🎯

---

**TEAM-384:** Complete code flow traced from CLI to terminal output. Every step documented! 📋
