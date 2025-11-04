# TEAM-395: Phase 5 - Job & SSE Endpoints

**Team:** TEAM-395  
**Phase:** 5 - Job Submission & SSE Streaming  
**Duration:** 45 hours  
**Dependencies:** TEAM-393 (generation engine), TEAM-394 (HTTP infra)  
**Parallel Work:** None (needs both dependencies)

---

## 🎯 Mission

Implement job submission endpoint and Server-Sent Events (SSE) streaming for real-time progress updates. Enable clients to submit generation jobs and receive live progress.

---

## 📦 What You're Building

### Files to Create (4 files, ~550 LOC total)

1. **`src/http/jobs.rs`** (~150 LOC)
   - POST /v1/jobs endpoint
   - Job submission logic
   - Job ID generation

2. **`src/http/stream.rs`** (~150 LOC)
   - GET /v1/jobs/:id/stream endpoint
   - SSE connection management
   - Stream lifecycle

3. **`src/http/sse.rs`** (~150 LOC)
   - SSE event formatting
   - Event types
   - [DONE] marker

4. **`src/http/narration_channel.rs`** (~100 LOC)
   - Narration → SSE bridge
   - Progress event conversion
   - Channel management

---

## 📋 Task Breakdown

### Day 1: Study & Design (8 hours)

**Morning (4 hours):**
- [ ] Study TEAM-393's generation engine API (1 hour)
- [ ] Study TEAM-394's HTTP infrastructure (1 hour)
- [ ] Read `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/jobs.rs` (1 hour)
- [ ] Read `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/stream.rs` (1 hour)

**Afternoon (4 hours):**
- [ ] Design job submission flow (2 hours)
- [ ] Design SSE event types (1 hour)
- [ ] Design error handling (1 hour)

**Output:** Design document, API contracts

---

### Day 2: Job Submission Endpoint (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/jobs.rs` (30 min)
- [ ] Define request/response types (1 hour)
- [ ] Implement POST /v1/jobs handler (1.5 hours)
- [ ] Add job ID generation (1 hour)

**Afternoon (4 hours):**
- [ ] Integrate with generation engine (2 hours)
- [ ] Add error handling (1 hour)
- [ ] Write tests (1 hour)

**Output:** Job submission working, returns job_id

---

### Day 3: SSE Infrastructure (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/sse.rs` (30 min)
- [ ] Define SSE event types (1 hour)
- [ ] Implement event formatting (1.5 hours)
- [ ] Add [DONE] marker (1 hour)

**Afternoon (4 hours):**
- [ ] Create `src/http/stream.rs` (30 min)
- [ ] Implement GET /v1/jobs/:id/stream handler (2 hours)
- [ ] Add SSE headers (1 hour)
- [ ] Test SSE connection (30 min)

**Output:** SSE endpoint working, sends events

---

### Day 4: Progress Streaming (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/narration_channel.rs` (30 min)
- [ ] Bridge generation events → SSE (2 hours)
- [ ] Add progress event conversion (1.5 hours)

**Afternoon (4 hours):**
- [ ] Add completion event with image (2 hours)
- [ ] Add error event handling (1 hour)
- [ ] Test end-to-end flow (1 hour)

**Output:** Progress streaming working end-to-end

---

### Day 5: Polish & Testing (8 hours)

**Morning (4 hours):**
- [ ] Add client disconnection handling (1 hour)
- [ ] Add timeout handling (1 hour)
- [ ] Add job registry integration (1 hour)
- [ ] Fix edge cases (1 hour)

**Afternoon (4 hours):**
- [ ] Write integration tests (2 hours)
- [ ] Load testing (1 hour)
- [ ] Documentation (1 hour)

**Output:** Production-ready, all tests passing

---

### Day 6: Integration (5 hours)

**Morning (3 hours):**
- [ ] Update routes.rs with new endpoints (1 hour)
- [ ] End-to-end testing (1 hour)
- [ ] Bug fixes (1 hour)

**Afternoon (2 hours):**
- [ ] Final review (1 hour)
- [ ] Handoff preparation (1 hour)

**Output:** Ready for TEAM-397 integration

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] POST /v1/jobs accepts generation requests
- [ ] Returns job_id immediately
- [ ] GET /v1/jobs/:id/stream establishes SSE connection
- [ ] Progress events fire for each diffusion step
- [ ] Completion event includes base64 image
- [ ] [DONE] marker sent at end
- [ ] Client disconnection handled gracefully
- [ ] Multiple concurrent streams work
- [ ] All tests passing
- [ ] Clean compilation (0 warnings)
- [ ] Can handle 10 concurrent generation jobs

---

## 🧪 Testing Requirements

### Integration Test

```rust
#[tokio::test]
async fn test_job_submission_and_streaming() {
    let app = create_test_app().await;
    
    // Submit job
    let request = json!({
        "prompt": "a photo of a cat",
        "steps": 20,
        "seed": 42
    });
    
    let response = app
        .post("/v1/jobs")
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), 200);
    let job_id: String = response.json().await.unwrap();
    
    // Stream progress
    let mut stream = app
        .get(&format!("/v1/jobs/{}/stream", job_id))
        .send()
        .await
        .unwrap()
        .bytes_stream();
    
    let mut progress_count = 0;
    let mut got_completion = false;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let text = String::from_utf8(chunk.to_vec()).unwrap();
        
        if text.contains("\"event\":\"progress\"") {
            progress_count += 1;
        }
        if text.contains("\"event\":\"complete\"") {
            got_completion = true;
            assert!(text.contains("image_base64"));
        }
        if text.contains("[DONE]") {
            break;
        }
    }
    
    assert_eq!(progress_count, 20);
    assert!(got_completion);
}
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **LLM Worker Jobs** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/jobs.rs`
   - Focus: Job submission pattern

2. **LLM Worker Stream** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/stream.rs`
   - Focus: SSE streaming pattern

3. **TEAM-393's Generation Engine** (Your Dependency)
   - Path: `src/backend/generation_engine.rs`
   - Usage: `GenerationEngine::submit()`

4. **TEAM-394's HTTP Infra** (Your Foundation)
   - Path: `src/http/backend.rs`, `src/http/routes.rs`
   - Usage: AppState, router extension

---

## 🔧 Implementation Notes

### Job Submission

```rust
#[derive(Deserialize)]
pub struct JobRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub width: usize,
    pub height: usize,
}

#[derive(Serialize)]
pub struct JobResponse {
    pub job_id: String,
}

pub async fn submit_job(
    State(state): State<AppState>,
    Json(request): Json<JobRequest>,
) -> Result<Json<JobResponse>, StatusCode> {
    let job_id = generate_job_id();
    
    let gen_request = GenerationRequest {
        job_id: job_id.clone(),
        prompt: request.prompt,
        steps: request.steps,
        // ... other fields
    };
    
    let (tx, rx) = mpsc::channel(100);
    state.generation_engine.submit(gen_request, tx).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Store rx in job registry for streaming
    state.job_registry.insert(job_id.clone(), rx).await;
    
    Ok(Json(JobResponse { job_id }))
}

fn generate_job_id() -> String {
    format!("job_{}", uuid::Uuid::new_v4())
}
```

### SSE Streaming

```rust
pub async fn stream_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.job_registry.get(&job_id).await
        .expect("Job not found");
    
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            match event {
                GenerationResponse::Progress { step, total } => {
                    let data = json!({
                        "event": "progress",
                        "step": step,
                        "total": total
                    });
                    yield Ok(Event::default().data(data.to_string()));
                }
                GenerationResponse::Complete { image_base64 } => {
                    let data = json!({
                        "event": "complete",
                        "image_base64": image_base64
                    });
                    yield Ok(Event::default().data(data.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                GenerationResponse::Error { message } => {
                    let data = json!({
                        "event": "error",
                        "message": message
                    });
                    yield Ok(Event::default().data(data.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    };
    
    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

### SSE Event Format

```
data: {"event":"progress","step":1,"total":20}

data: {"event":"progress","step":2,"total":20}

data: {"event":"complete","image_base64":"iVBORw0KGgoAAAANS..."}

data: [DONE]
```

---

## 🚨 Common Pitfalls

1. **SSE Headers**
   - Must set: `Content-Type: text/event-stream`
   - Must set: `Cache-Control: no-cache`
   - Axum Sse helper does this automatically

2. **Channel Cleanup**
   - Problem: Job registry grows unbounded
   - Solution: Remove entry after [DONE]

3. **Client Disconnection**
   - Problem: Server keeps generating after client disconnects
   - Solution: Check stream health, cancel generation

4. **Large Images**
   - Problem: Base64 image too large for single SSE event
   - Solution: Use chunked transfer or separate download endpoint

---

## 🎯 Handoff to TEAM-397

**What TEAM-397 needs from you:**

### Files Created
- `src/http/jobs.rs` - Job submission
- `src/http/stream.rs` - SSE streaming
- `src/http/sse.rs` - SSE utilities
- `src/http/narration_channel.rs` - Event bridge

### APIs Exposed

```rust
// Job submission
POST /v1/jobs
Body: { "prompt": "...", "steps": 20, ... }
Response: { "job_id": "job_..." }

// SSE streaming
GET /v1/jobs/:id/stream
Response: text/event-stream
Events: progress, complete, error, [DONE]
```

### What Works
- Job submission returns job_id
- SSE streaming sends progress
- Completion event includes image
- Error handling
- Multiple concurrent jobs

### What TEAM-397 Will Do
- Wire into binaries
- Add job_router.rs integration
- End-to-end testing

---

## 📊 Progress Tracking

- [ ] Day 1: Design complete
- [ ] Day 2: Job submission working
- [ ] Day 3: SSE infrastructure working
- [ ] Day 4: Progress streaming working
- [ ] Day 5: Polish complete
- [ ] Day 6: Integration ready, handoff complete

---

**TEAM-395: You're building the real-time experience. Make it smooth and reliable.** ⚡
