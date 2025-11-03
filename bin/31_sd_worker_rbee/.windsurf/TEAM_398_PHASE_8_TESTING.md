# TEAM-398: Phase 8 - Testing Suite

**Team:** TEAM-398  
**Phase:** 8 - Comprehensive Testing  
**Duration:** 50 hours  
**Dependencies:** TEAM-397 (working end-to-end)  
**Parallel Work:** ✅ Can work parallel to TEAM-399 (UI)

---

## 🎯 Mission

Build comprehensive test coverage: unit tests for all modules, integration tests for pipelines, HTTP API tests, and performance benchmarks.

---

## 📦 What You're Building

### Test Files to Create (~1000 LOC total)

1. **Unit Tests** (6 files, ~400 LOC)
   - `src/backend/inference_test.rs`
   - `src/backend/scheduler_test.rs`
   - `src/backend/vae_test.rs`
   - `src/backend/generation_engine_test.rs`
   - `src/http/validation_test.rs`
   - `src/http/jobs_test.rs`

2. **Integration Tests** (4 files, ~400 LOC)
   - `tests/text_to_image_test.rs`
   - `tests/image_to_image_test.rs`
   - `tests/http_api_test.rs`
   - `tests/sse_streaming_test.rs`

3. **Benchmarks** (2 files, ~200 LOC)
   - `benches/inference_bench.rs`
   - `benches/http_bench.rs`

---

## 📋 Task Breakdown

### Week 1: Unit Tests (24 hours)

**Day 1: Backend Tests (8 hours)**
- [ ] Test inference pipeline (2 hours)
- [ ] Test schedulers (DDIM, Euler) (2 hours)
- [ ] Test VAE encode/decode (2 hours)
- [ ] Test CLIP encoding (2 hours)

**Day 2: Generation Engine Tests (8 hours)**
- [ ] Test request queue (2 hours)
- [ ] Test generation engine (2 hours)
- [ ] Test progress callbacks (2 hours)
- [ ] Test cancellation (2 hours)

**Day 3: HTTP Tests (8 hours)**
- [ ] Test validation logic (2 hours)
- [ ] Test job submission (2 hours)
- [ ] Test SSE formatting (2 hours)
- [ ] Test middleware (2 hours)

---

### Week 2: Integration Tests (16 hours)

**Day 4: Pipeline Tests (8 hours)**
- [ ] Test text-to-image end-to-end (3 hours)
- [ ] Test image-to-image (2 hours)
- [ ] Test inpainting (3 hours)

**Day 5: HTTP API Tests (8 hours)**
- [ ] Test job submission flow (2 hours)
- [ ] Test SSE streaming (3 hours)
- [ ] Test error scenarios (2 hours)
- [ ] Test concurrent requests (1 hour)

---

### Week 3: Benchmarks & Load Testing (10 hours)

**Day 6: Benchmarks (5 hours)**
- [ ] Inference speed benchmarks (2 hours)
- [ ] Memory usage benchmarks (1 hour)
- [ ] HTTP throughput benchmarks (2 hours)

**Day 7: Load Testing (5 hours)**
- [ ] Concurrent generation tests (2 hours)
- [ ] Stress testing (2 hours)
- [ ] Documentation (1 hour)

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] All unit tests passing (>50 tests)
- [ ] All integration tests passing (>10 tests)
- [ ] Code coverage >70%
- [ ] Benchmarks run successfully
- [ ] Load tests pass (10 concurrent jobs)
- [ ] Performance regression tests in place
- [ ] Test documentation complete
- [ ] CI/CD integration ready
- [ ] Clean compilation (0 warnings)

---

## 🧪 Test Examples

### Unit Test: Inference Pipeline

```rust
#[tokio::test]
async fn test_text_to_image_basic() {
    let device = Device::Cpu;
    let model = SDModel::load("sd-v1-5", &device).await.unwrap();
    let backend = CandleSDBackend::new(model, device).unwrap();
    
    let config = SamplingConfig {
        prompt: "a photo of a cat".to_string(),
        steps: 20,
        guidance_scale: 7.5,
        seed: Some(42),
        width: 512,
        height: 512,
        ..Default::default()
    };
    
    let image = backend.text_to_image(config).await.unwrap();
    
    assert_eq!(image.width(), 512);
    assert_eq!(image.height(), 512);
}

#[tokio::test]
async fn test_seed_reproducibility() {
    let backend = create_test_backend().await;
    
    let config = SamplingConfig {
        prompt: "a photo of a dog".to_string(),
        seed: Some(42),
        ..Default::default()
    };
    
    let image1 = backend.text_to_image(config.clone()).await.unwrap();
    let image2 = backend.text_to_image(config).await.unwrap();
    
    // Images should be identical with same seed
    assert_eq!(image_to_bytes(&image1), image_to_bytes(&image2));
}
```

### Integration Test: HTTP API

```rust
#[tokio::test]
async fn test_job_submission_and_streaming() {
    let app = create_test_app().await;
    
    // Submit job
    let request = json!({
        "prompt": "a photo of a cat",
        "steps": 20,
        "seed": 42,
        "width": 512,
        "height": 512
    });
    
    let response = app
        .post("/v1/jobs")
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), 200);
    
    let body: JobResponse = response.json().await.unwrap();
    let job_id = body.job_id;
    
    // Stream progress
    let mut stream = app
        .get(&format!("/v1/jobs/{}/stream", job_id))
        .send()
        .await
        .unwrap()
        .bytes_stream();
    
    let mut events = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let text = String::from_utf8(chunk.to_vec()).unwrap();
        
        if text.contains("[DONE]") {
            break;
        }
        
        if text.starts_with("data: ") {
            let json_str = text.strip_prefix("data: ").unwrap();
            if let Ok(event) = serde_json::from_str::<Value>(json_str) {
                events.push(event);
            }
        }
    }
    
    // Verify events
    let progress_events: Vec<_> = events.iter()
        .filter(|e| e["event"] == "progress")
        .collect();
    assert_eq!(progress_events.len(), 20);
    
    let complete_events: Vec<_> = events.iter()
        .filter(|e| e["event"] == "complete")
        .collect();
    assert_eq!(complete_events.len(), 1);
    assert!(complete_events[0]["image_base64"].is_string());
}
```

### Benchmark: Inference Speed

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_text_to_image(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let backend = runtime.block_on(create_test_backend());
    
    c.bench_function("text_to_image_20_steps", |b| {
        b.to_async(&runtime).iter(|| async {
            let config = SamplingConfig {
                prompt: black_box("a photo of a cat".to_string()),
                steps: 20,
                seed: Some(42),
                ..Default::default()
            };
            
            backend.text_to_image(config).await.unwrap()
        });
    });
}

criterion_group!(benches, benchmark_text_to_image);
criterion_main!(benches);
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **LLM Worker Tests** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/tests/`
   - Focus: Test patterns, fixtures

2. **Rust Testing Best Practices**
   - Unit tests with `#[test]`
   - Integration tests in `tests/`
   - Benchmarks with Criterion

3. **TEAM-397's Implementation** (What You're Testing)
   - All backend modules
   - All HTTP endpoints
   - All binaries

---

## 🔧 Test Organization

### Unit Tests (in src/)

```
src/
├── backend/
│   ├── inference.rs
│   ├── inference_test.rs  ← Unit tests
│   ├── scheduler.rs
│   └── scheduler_test.rs  ← Unit tests
└── http/
    ├── jobs.rs
    └── jobs_test.rs       ← Unit tests
```

### Integration Tests (in tests/)

```
tests/
├── text_to_image_test.rs
├── image_to_image_test.rs
├── http_api_test.rs
└── common/
    └── mod.rs  ← Shared test utilities
```

### Benchmarks (in benches/)

```
benches/
├── inference_bench.rs
└── http_bench.rs
```

---

## 🚨 Common Pitfalls

1. **Model Download in Tests**
   - Problem: Tests download models every time
   - Solution: Use cached models, mock backends

2. **Slow Tests**
   - Problem: Full inference takes minutes
   - Solution: Use small step counts (5-10) for tests

3. **Flaky Tests**
   - Problem: Random failures
   - Solution: Use fixed seeds, deterministic tests

4. **Resource Leaks**
   - Problem: Tests don't clean up
   - Solution: Use `Drop` trait, cleanup fixtures

---

## 🎯 Handoff to TEAM-401

**What TEAM-401 needs from you:**

### Files Created
- Unit tests for all modules
- Integration tests for pipelines
- HTTP API tests
- Benchmarks

### Test Coverage
- Backend: >80% coverage
- HTTP: >70% coverage
- Overall: >70% coverage

### What Works
- All tests passing
- Benchmarks baseline established
- Load tests validate 10 concurrent jobs
- CI/CD ready

### What TEAM-401 Will Do
- Performance optimization
- Documentation
- Deployment scripts

---

## 📊 Progress Tracking

- [ ] Week 1: Unit tests complete
- [ ] Week 2: Integration tests complete
- [ ] Week 3: Benchmarks and load tests complete
- [ ] All tests passing, ready for TEAM-401

---

**TEAM-398: You're ensuring quality. Test everything thoroughly.** ✅
