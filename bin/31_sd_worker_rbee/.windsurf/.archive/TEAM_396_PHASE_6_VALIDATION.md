# TEAM-396: Phase 6 - Validation & Middleware

**Team:** TEAM-396  
**Phase:** 6 - Request Validation & Middleware  
**Duration:** 40 hours  
**Dependencies:** TEAM-394 (HTTP infrastructure)  
**Parallel Work:** ✅ Can work parallel to TEAM-395

---

## 🎯 Mission

Implement request validation for all generation parameters and authentication middleware. Ensure all inputs are safe and valid before processing.

---

## 📦 What You're Building

### Files to Create (3 files, ~350 LOC total)

1. **`src/http/validation.rs`** (~200 LOC)
   - Request validation logic
   - Parameter bounds checking
   - Error messages

2. **`src/http/middleware/auth.rs`** (~100 LOC)
   - Bearer token authentication
   - Token validation

3. **`src/http/middleware/mod.rs`** (~50 LOC)
   - Middleware exports
   - Middleware utilities

---

## 📋 Task Breakdown

### Day 1: Study & Design (8 hours)

**Morning (4 hours):**
- [ ] Study TEAM-394's HTTP infrastructure (1 hour)
- [ ] Read `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/validation.rs` (2 hours)
- [ ] Study Axum middleware patterns (1 hour)

**Afternoon (4 hours):**
- [ ] Design validation rules (2 hours)
- [ ] Design error response format (1 hour)
- [ ] Design auth strategy (1 hour)

**Output:** Validation spec, auth design

---

### Day 2-3: Request Validation (16 hours)

**Day 2 Morning (4 hours):**
- [ ] Create `src/http/validation.rs` (30 min)
- [ ] Define validation error types (1 hour)
- [ ] Implement prompt validation (1.5 hours)
- [ ] Implement dimension validation (1 hour)

**Day 2 Afternoon (4 hours):**
- [ ] Implement steps validation (1 hour)
- [ ] Implement guidance scale validation (1 hour)
- [ ] Implement strength validation (1 hour)
- [ ] Implement seed validation (1 hour)

**Day 3 Morning (4 hours):**
- [ ] Implement base64 image validation (2 hours)
- [ ] Add comprehensive error messages (1 hour)
- [ ] Write unit tests (1 hour)

**Day 3 Afternoon (4 hours):**
- [ ] Test edge cases (2 hours)
- [ ] Integration with job endpoint (1 hour)
- [ ] Documentation (1 hour)

**Output:** Complete validation, tests passing

---

### Day 4: Authentication Middleware (8 hours)

**Morning (4 hours):**
- [ ] Create `src/http/middleware/auth.rs` (30 min)
- [ ] Implement bearer token extraction (1 hour)
- [ ] Implement token validation (1.5 hours)
- [ ] Add auth error responses (1 hour)

**Afternoon (4 hours):**
- [ ] Create `src/http/middleware/mod.rs` (30 min)
- [ ] Wire auth into router (1.5 hours)
- [ ] Write tests (1 hour)
- [ ] Test with curl/Postman (1 hour)

**Output:** Auth middleware working

---

### Day 5: Polish & Testing (8 hours)

**Morning (4 hours):**
- [ ] Add request ID validation (1 hour)
- [ ] Add rate limiting (optional) (2 hours)
- [ ] Fix edge cases (1 hour)

**Afternoon (4 hours):**
- [ ] Integration testing (2 hours)
- [ ] Load testing (1 hour)
- [ ] Documentation (1 hour)

**Output:** Production-ready validation and auth

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] Prompt length validated (max 77 tokens for CLIP)
- [ ] Image dimensions validated (multiples of 8, reasonable range)
- [ ] Steps validated (1-150 range)
- [ ] Guidance scale validated (1.0-20.0 range)
- [ ] Strength validated (0.0-1.0 for img2img)
- [ ] Seed validated (u64 range)
- [ ] Base64 images validated (format, size)
- [ ] Clear error messages for all validation failures
- [ ] Bearer token authentication works
- [ ] Unauthorized requests rejected with 401
- [ ] All tests passing
- [ ] Clean compilation (0 warnings)

---

## 🧪 Testing Requirements

### Unit Tests

```rust
#[test]
fn test_prompt_validation() {
    // Valid prompt
    assert!(validate_prompt("a photo of a cat").is_ok());
    
    // Empty prompt
    assert!(validate_prompt("").is_err());
    
    // Too long (>77 tokens)
    let long_prompt = "word ".repeat(100);
    assert!(validate_prompt(&long_prompt).is_err());
}

#[test]
fn test_dimension_validation() {
    // Valid dimensions
    assert!(validate_dimensions(512, 512).is_ok());
    assert!(validate_dimensions(768, 512).is_ok());
    
    // Not multiple of 8
    assert!(validate_dimensions(513, 512).is_err());
    
    // Too large
    assert!(validate_dimensions(4096, 4096).is_err());
    
    // Too small
    assert!(validate_dimensions(64, 64).is_err());
}

#[test]
fn test_steps_validation() {
    assert!(validate_steps(20).is_ok());
    assert!(validate_steps(0).is_err());
    assert!(validate_steps(200).is_err());
}
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **LLM Worker Validation** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/validation.rs`
   - Focus: Validation patterns, error handling

2. **LLM Worker Middleware** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/http/middleware/`
   - Focus: Auth pattern

3. **TEAM-394's HTTP Infra** (Your Foundation)
   - Path: `src/http/routes.rs`
   - Usage: Middleware integration

---

## 🔧 Implementation Notes

### Validation Pattern

```rust
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Prompt is empty")]
    EmptyPrompt,
    
    #[error("Prompt too long (max 77 tokens, got {0})")]
    PromptTooLong(usize),
    
    #[error("Invalid dimensions: {0}x{1} (must be multiples of 8)")]
    InvalidDimensions(usize, usize),
    
    #[error("Steps out of range: {0} (must be 1-150)")]
    InvalidSteps(usize),
    
    #[error("Guidance scale out of range: {0} (must be 1.0-20.0)")]
    InvalidGuidanceScale(f64),
}

pub fn validate_job_request(request: &JobRequest) -> Result<(), ValidationError> {
    validate_prompt(&request.prompt)?;
    validate_dimensions(request.width, request.height)?;
    validate_steps(request.steps)?;
    validate_guidance_scale(request.guidance_scale)?;
    
    if let Some(strength) = request.strength {
        validate_strength(strength)?;
    }
    
    Ok(())
}

pub fn validate_prompt(prompt: &str) -> Result<(), ValidationError> {
    if prompt.is_empty() {
        return Err(ValidationError::EmptyPrompt);
    }
    
    // Rough token count (CLIP tokenizer limit is 77)
    let token_count = prompt.split_whitespace().count();
    if token_count > 77 {
        return Err(ValidationError::PromptTooLong(token_count));
    }
    
    Ok(())
}

pub fn validate_dimensions(width: usize, height: usize) -> Result<(), ValidationError> {
    if width % 8 != 0 || height % 8 != 0 {
        return Err(ValidationError::InvalidDimensions(width, height));
    }
    
    if width < 128 || width > 2048 || height < 128 || height > 2048 {
        return Err(ValidationError::InvalidDimensions(width, height));
    }
    
    Ok(())
}

pub fn validate_steps(steps: usize) -> Result<(), ValidationError> {
    if steps < 1 || steps > 150 {
        return Err(ValidationError::InvalidSteps(steps));
    }
    Ok(())
}

pub fn validate_guidance_scale(scale: f64) -> Result<(), ValidationError> {
    if scale < 1.0 || scale > 20.0 {
        return Err(ValidationError::InvalidGuidanceScale(scale));
    }
    Ok(())
}
```

### Authentication Middleware

```rust
pub struct AuthMiddleware;

#[async_trait]
impl<S> Middleware<S> for AuthMiddleware
where
    S: Send + Sync,
{
    async fn handle(
        &self,
        req: Request<Body>,
        next: Next<S>,
    ) -> Result<Response, StatusCode> {
        // Extract bearer token
        let token = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "))
            .ok_or(StatusCode::UNAUTHORIZED)?;
        
        // Validate token
        if !is_valid_token(token) {
            return Err(StatusCode::UNAUTHORIZED);
        }
        
        // Continue to next middleware/handler
        Ok(next.run(req).await)
    }
}

fn is_valid_token(token: &str) -> bool {
    // TODO: Implement actual token validation
    // For now, accept any non-empty token
    !token.is_empty()
}
```

### Integration with Routes

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health_check))
        .route("/ready", get(ready::readiness_check))
        .route("/v1/jobs", post(jobs::submit_job))
        .route("/v1/jobs/:id/stream", get(stream::stream_job))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .layer(TraceLayer::new_for_http())
                .layer(AuthMiddleware) // Add auth here
                .layer(TimeoutLayer::new(Duration::from_secs(300)))
        )
        .with_state(state)
}
```

---

## 🚨 Common Pitfalls

1. **Token Count Estimation**
   - Problem: Word count ≠ token count
   - Solution: Use rough estimate, CLIP will truncate

2. **Dimension Validation**
   - Problem: Forgetting multiples of 8 requirement
   - Solution: Always check `% 8 == 0`

3. **Error Messages**
   - Problem: Vague "invalid input" messages
   - Solution: Specific messages with actual values

4. **Auth Bypass**
   - Problem: Some routes skip auth
   - Solution: Apply auth middleware globally, exclude health/ready

---

## 🎯 Handoff to TEAM-397

**What TEAM-397 needs from you:**

### Files Created
- `src/http/validation.rs` - Request validation
- `src/http/middleware/auth.rs` - Authentication
- `src/http/middleware/mod.rs` - Middleware exports

### APIs Exposed

```rust
// Validation
pub fn validate_job_request(request: &JobRequest) -> Result<(), ValidationError>;

// Middleware
pub struct AuthMiddleware;
```

### What Works
- All generation parameters validated
- Clear error messages
- Bearer token authentication
- Integration with routes

### What TEAM-397 Will Do
- Wire validation into job submission
- Configure auth tokens
- End-to-end testing

---

## 📊 Progress Tracking

- [ ] Day 1: Design complete
- [ ] Day 2: Validation core working
- [ ] Day 3: Validation complete, tests passing
- [ ] Day 4: Auth middleware working
- [ ] Day 5: Polish complete, ready for handoff

---

**TEAM-396: You're the gatekeeper. Keep the bad inputs out.** 🛡️
