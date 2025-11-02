# OpenAI-Compatible Request Flow Documentation

**Complete flow from OpenAI client to rbee worker execution and streaming**  
**Date:** November 2, 2025  
**Status:** ⚠️ DESIGN PHASE (Adapter is stub implementation)

---

## Overview

This directory contains documentation of the OpenAI-compatible API flow in rbee, showing how external OpenAI clients can use rbee without modification.

**Flow Summary:**
```
OpenAI Client
    ↓ POST /openai/v1/chat/completions
OpenAI Adapter (Part 1) ⚠️ STUB
    ↓ Translate to Operation::Infer
Queen Job System (Part 2)
    ↓ Route to scheduler
Scheduler (Part 3)
    ↓ Select worker
Worker Execution (Part 4)
    ↓ Stream tokens
Response Translation (Part 5) ⚠️ PLANNED
    ↓ OpenAI format
Client Receives Response
```

---

## ⚠️ Current Implementation Status

### What Exists

**✅ Implemented:**
- Native rbee API (`/v1/jobs`)
- Job-based architecture
- SSE streaming
- Scheduler (simple, first-available)
- Worker execution
- Token streaming

**⚠️ Stub (Design Phase):**
- OpenAI adapter module structure
- OpenAI type definitions
- Router configuration
- Handler stubs (return `NOT_IMPLEMENTED`)

**❌ Not Implemented:**
- Request translation (OpenAI → rbee)
- Response translation (rbee → OpenAI)
- Streaming format conversion
- Error code mapping
- Model name mapping

---

## Documentation Parts

### ⚠️ [Part 1: Adapter Entry Point](./OPENAI_FLOW_PART_1_ADAPTER_ENTRY.md) (STUB)

**Scope:** OpenAI Client → Adapter → Request Translation

**Key Topics:**
- Crate structure (`rbee-openai-adapter`)
- Router configuration
- Request types (OpenAI format)
- Planned translation logic
- Error mapping

**Status:** Design phase, handlers return `501 NOT_IMPLEMENTED`

**Key Files:**
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/lib.rs`
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/router.rs`
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs`
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/types.rs`

---

### ✅ Part 2: Queen Routing (IMPLEMENTED)

**Scope:** Operation::Infer → Job System → Scheduler

**Key Topics:**
- Job creation (same as native API)
- Operation routing (`job_router.rs`)
- Scheduler invocation
- Worker selection

**Status:** Fully implemented (used by native API)

**Key Files:**
- `bin/10_queen_rbee/src/job_router.rs`
- `bin/15_queen_rbee_crates/scheduler/src/simple.rs`

**See Also:** [JOB_FLOW_PART_2_QUEEN_RECEPTION.md](./JOB_FLOW_PART_2_QUEEN_RECEPTION.md)

---

### ✅ Part 3: Scheduler Execution (IMPLEMENTED)

**Scope:** Worker Selection → Job Execution → Token Streaming

**Key Topics:**
- Worker registry query
- First-available selection
- POST to worker `/v1/inference`
- SSE stream connection
- Token forwarding

**Status:** Fully implemented

**Key Files:**
- `bin/15_queen_rbee_crates/scheduler/src/simple.rs`
- `bin/15_queen_rbee_crates/scheduler/src/types.rs`

---

### ✅ Part 4: Worker Execution (IMPLEMENTED)

**Scope:** Worker Receives Request → Inference → Token Streaming

**Key Topics:**
- Worker HTTP endpoint
- Model loading
- Inference execution
- Token generation
- SSE streaming

**Status:** Fully implemented

**Key Files:**
- `bin/30_llm_worker_rbee/src/http/routes.rs`
- `bin/30_llm_worker_rbee/src/backend/inference.rs`

---

### ✅ [Part 2: Request Translation & Job Submission](./OPENAI_FLOW_PART_2_TRANSLATION.md)

**Scope:** OpenAI Request → rbee Operation → Queen Job System

**Key Topics:**
- Extract prompt from messages
- Map model names (OpenAI → rbee)
- Convert to Operation::Infer
- Submit to queen job system
- Adapter state management

**Status:** Fully documented (implementation required)

**Key Files:**
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/translation.rs` (NEW)
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/handlers.rs` (UPDATE)

---

### ✅ [Part 3: Response Translation & Streaming](./OPENAI_FLOW_PART_3_RESPONSE_STREAMING.md)

**Scope:** rbee SSE Events → OpenAI Format → Client

**Key Topics:**
- Non-streaming response handling
- Streaming response with SSE
- Event translation (rbee → OpenAI)
- Token counting
- Models endpoint implementation

**Status:** Fully documented (implementation required)

**Key Files:**
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/streaming.rs` (NEW)
- `bin/15_queen_rbee_crates/rbee-openai-adapter/src/tokens.rs` (NEW)

---

### ✅ [Part 4: Complete Implementation Guide](./OPENAI_FLOW_PART_4_IMPLEMENTATION_GUIDE.md)

**Scope:** Step-by-Step Implementation from Start to Production

**Key Topics:**
- Phase 1: Basic translation (M2)
- Phase 2: Streaming responses (M2)
- Phase 3: Full compatibility (M3)
- Testing strategy
- Deployment checklist

**Status:** Complete implementation guide

**Estimated Effort:** 2-3 weeks (M2-M3)

---

## Quick Reference

### API Endpoints

**OpenAI-Compatible (Planned):**
| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| POST | `/openai/v1/chat/completions` | Chat completions | ⚠️ Stub |
| GET | `/openai/v1/models` | List models | ⚠️ Stub |
| GET | `/openai/v1/models/{model}` | Get model | ⚠️ Stub |

**Native rbee (Implemented):**
| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| POST | `/v1/jobs` | Submit job | ✅ Implemented |
| GET | `/v1/jobs/{job_id}/stream` | SSE stream | ✅ Implemented |

---

### Request Translation

**OpenAI Format:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "stream": true,
  "max_tokens": 100,
  "temperature": 0.7
}
```

**rbee Format (After Translation):**
```json
{
  "operation": "infer",
  "hive_id": "localhost",
  "model": "tinyllama",
  "prompt": "system: You are helpful\nuser: Hello",
  "stream": true,
  "max_tokens": 100,
  "temperature": 0.7
}
```

---

### Response Translation

**rbee SSE Event:**
```json
{
  "action": "token",
  "actor": "llm-worker-rbee",
  "formatted": "Hello",
  "job_id": "job_abc123",
  "timestamp": "2025-11-02T17:00:00Z"
}
```

**OpenAI Chunk (Planned):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1730563200,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "delta": {"content": "Hello"},
    "finish_reason": null
  }]
}
```

---

## Architecture Comparison

### OpenAI API Flow (Planned)

```
OpenAI Client
    ↓
POST /openai/v1/chat/completions
    ↓
OpenAI Adapter
    ↓ translate request
    ↓
POST /v1/jobs (internal)
    ↓
Queen Job System
    ↓
Scheduler
    ↓
Worker
    ↓ tokens
    ↓
OpenAI Adapter
    ↓ translate response
    ↓
OpenAI Client
```

### Native rbee API Flow (Implemented)

```
rbee-keeper CLI
    ↓
POST /v1/jobs
    ↓
Queen Job System
    ↓
Scheduler
    ↓
Worker
    ↓ tokens
    ↓
rbee-keeper CLI
```

**Key Insight:** OpenAI adapter is a thin translation layer over native API

---

## Implementation Roadmap

### Phase 1: Basic Translation (M2)
- [ ] Implement request translation
- [ ] Implement non-streaming responses
- [ ] Basic error mapping
- [ ] Model name mapping

### Phase 2: Streaming (M2)
- [ ] SSE event translation
- [ ] OpenAI chunk format
- [ ] Streaming completion markers

### Phase 3: Full Compatibility (M3)
- [ ] All OpenAI parameters supported
- [ ] Function calling (if needed)
- [ ] Token counting
- [ ] Usage statistics

### Phase 4: Advanced Features (M4+)
- [ ] Multiple model support
- [ ] Rate limiting
- [ ] API key authentication
- [ ] Billing integration

---

## Key Design Decisions

### 1. Thin Adapter Layer

**Decision:** OpenAI adapter is a thin translation layer, not a reimplementation

**Benefits:**
- ✅ Reuses existing job system
- ✅ Reuses existing scheduler
- ✅ Reuses existing worker execution
- ✅ Minimal code duplication
- ✅ Single source of truth

---

### 2. Separate Crate

**Decision:** OpenAI adapter is a separate crate (`rbee-openai-adapter`)

**Benefits:**
- ✅ Optional dependency
- ✅ Can be disabled if not needed
- ✅ Clear separation of concerns
- ✅ Independent versioning

---

### 3. Message Concatenation

**Decision:** Concatenate OpenAI messages into single prompt

**Rationale:**
- rbee workers expect single prompt string
- Simple transformation
- Preserves conversation context

**Alternative Considered:**
- Native multi-message support (requires worker changes)

---

### 4. Model Name Mapping

**Decision:** Map OpenAI model names to rbee model IDs

**Options:**
1. **Hardcoded mapping** (simple, inflexible)
2. **Configuration file** (flexible, requires management)
3. **Database** (most flexible, adds complexity)

**Recommendation:** Start with configuration file

---

## Error Handling

### OpenAI Error Format

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid model specified",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### rbee Error Mapping

| rbee Error | OpenAI Error Type | HTTP Status |
|------------|-------------------|-------------|
| Model not found | `model_not_found` | 404 |
| Worker unavailable | `service_unavailable` | 503 |
| Invalid parameters | `invalid_request_error` | 400 |
| Timeout | `timeout_error` | 504 |
| Internal error | `internal_error` | 500 |

---

## Testing Strategy

### Unit Tests
- [ ] Request translation
- [ ] Response translation
- [ ] Error mapping
- [ ] Model name mapping

### Integration Tests
- [ ] End-to-end OpenAI flow
- [ ] Streaming responses
- [ ] Error scenarios
- [ ] Timeout handling

### Compatibility Tests
- [ ] Test with real OpenAI clients
- [ ] Test with OpenAI SDKs (Python, Node.js)
- [ ] Test with LangChain
- [ ] Test with LlamaIndex

---

## Performance Considerations

### Latency

**Additional overhead from OpenAI adapter:**
- Request translation: <1ms
- Response translation: <1ms per token
- Total: Negligible (<1% of inference time)

### Memory

**Per request:**
- Request object: ~1KB
- Response buffering (non-streaming): ~10KB
- Total: Minimal

---

## Security Considerations

### Authentication

**Planned:**
- API key authentication (OpenAI-compatible)
- Bearer token in `Authorization` header
- Key validation before translation

### Rate Limiting

**Planned:**
- Per-key rate limits
- Token-based limits
- Request-based limits

---

## Related Documentation

- [Job Flow Documentation](./README.md) — Native rbee API flow
- [Phase 3: Narration Usage](../analysis/PHASE_3_NARRATION_USAGE_PART_1.md)
- [Phase 4: Runtime Patterns](../analysis/PHASE_4_RUNTIME_PATTERNS.md)
- [Scheduler Documentation](../../bin/15_queen_rbee_crates/scheduler/README.md)

---

## References

### OpenAI API Documentation
- [Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Models API](https://platform.openai.com/docs/api-reference/models)
- [Streaming](https://platform.openai.com/docs/api-reference/streaming)

### Implementation Examples
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI Node.js SDK](https://github.com/openai/openai-node)

---

**Status:** ✅ ALL 4 PARTS COMPLETE  
**Maintainer:** TEAM-385+  
**Last Updated:** November 2, 2025  
**Total Documentation:** ~2,500 lines across 4 files  
**Implementation Target:** M2-M3 (Milestones 2-3)  
**Estimated Effort:** 2-3 weeks
