# Critical Reference Paths for SD Worker Development

**For:** All teams working on SD Worker (TEAM-391 through TEAM-401)  
**Status:** MANDATORY READING

---

## 🔥 MUST FOLLOW: Architecture Pattern

### Primary Pattern (MIRROR THIS STRUCTURE)
```
/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/
```

**Why:** The SD worker MUST mirror the LLM worker's structure for consistency.

**What to copy:**
- Module organization: `backend/`, `http/`, `bin/`
- Patterns: Request queue, generation engine, SSE streaming
- Naming conventions
- Error handling patterns
- Testing structure

**Key files to study:**
- `src/backend/mod.rs` - Backend module structure
- `src/backend/generation_engine.rs` - Background task pattern
- `src/backend/request_queue.rs` - MPSC queue pattern
- `src/http/mod.rs` - HTTP module structure
- `src/http/jobs.rs` - Job submission endpoint
- `src/http/stream.rs` - SSE streaming endpoint
- `src/http/sse.rs` - SSE utilities
- `src/bin/cpu.rs` - Binary entry point pattern

---

## 📚 Working Candle Examples (STUDY THESE)

### SD 1.5/2.1/XL/Turbo Implementation
```
/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/
```

**Critical file:**
- `main.rs` - Complete working implementation

**What you'll learn:**
- How to load CLIP text encoder
- How to load UNet diffusion model
- How to load VAE decoder
- How to set up scheduler (DDIM, Euler, etc.)
- Complete text-to-image pipeline
- Image-to-image transformation
- Inpainting with masks
- Prompt encoding with tokenizer
- Latent diffusion loop
- VAE decoding to images

**Key sections to study:**
- Lines 1-250: Argument parsing and model file paths
- Lines 250-400: Model loading from HuggingFace Hub
- Lines 400-600: Text encoding with CLIP
- Lines 600-800: Diffusion loop with scheduler
- Lines 800+: VAE decoding and image saving

### SD 3/3.5 Implementation (Future Reference)
```
/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion-3/
```

**What you'll learn:**
- MMDiT architecture (different from UNet)
- SD 3 Medium (2.5B params)
- SD 3.5 Large (8.1B params)
- SD 3.5 Turbo (4-step inference)

**Note:** Focus on SD 1.5/2.1/XL first. SD 3 is for future enhancement.

---

## 🔧 Shared Components (DO NOT DUPLICATE)

### Shared Worker Utilities
```
/home/vince/Projects/llama-orch/bin/32_shared_worker_rbee/
```

**Already available (created by TEAM-390):**
- `src/device.rs` - Device management (CPU/CUDA/Metal)
  - `init_cpu_device()`
  - `init_cuda_device(gpu_id)`
  - `init_metal_device(gpu_id)`
  - `verify_device(device)`

- `src/heartbeat.rs` - Heartbeat system
  - `send_heartbeat_to_queen()`
  - `start_heartbeat_task()`

**Usage in SD worker:**
```rust
// In sd-worker-rbee
use shared_worker_rbee::device;

let device = device::init_cpu_device()?;
device::verify_device(&device)?;
```

**Rule:** If you need device or heartbeat functionality, use the shared crate. DO NOT duplicate code.

**Future additions:** If you create utilities that could be shared between LLM and SD workers, add them to this crate.

---

## 📂 Directory Structure Comparison

### LLM Worker (Pattern to Follow)
```
bin/30_llm_worker_rbee/
├── src/
│   ├── backend/
│   │   ├── models/              # Model definitions
│   │   ├── generation_engine.rs # Background task
│   │   ├── request_queue.rs     # MPSC queue
│   │   ├── inference.rs         # Main inference
│   │   └── mod.rs
│   ├── http/
│   │   ├── jobs.rs              # Job submission
│   │   ├── stream.rs            # SSE streaming
│   │   ├── sse.rs               # SSE utilities
│   │   ├── validation.rs        # Request validation
│   │   ├── middleware/          # Auth, logging
│   │   └── mod.rs
│   ├── bin/
│   │   ├── cpu.rs               # CPU binary
│   │   ├── cuda.rs              # CUDA binary
│   │   └── metal.rs             # Metal binary
│   ├── lib.rs
│   ├── error.rs
│   └── ...
└── Cargo.toml
```

### SD Worker (MUST Match This Structure)
```
bin/31_sd_worker_rbee/
├── src/
│   ├── backend/
│   │   ├── models/              # SD model definitions (✅ DONE by TEAM-390)
│   │   ├── generation_engine.rs # Background task (TODO)
│   │   ├── request_queue.rs     # MPSC queue (TODO)
│   │   ├── inference.rs         # SD inference (TODO)
│   │   ├── clip.rs              # CLIP encoder (TODO)
│   │   ├── vae.rs               # VAE decoder (TODO)
│   │   ├── scheduler.rs         # Diffusion scheduler (TODO)
│   │   └── mod.rs               # (✅ DONE by TEAM-390)
│   ├── http/
│   │   ├── jobs.rs              # Job submission (TODO)
│   │   ├── stream.rs            # SSE streaming (TODO)
│   │   ├── sse.rs               # SSE utilities (TODO)
│   │   ├── validation.rs        # Request validation (TODO)
│   │   ├── middleware/          # Auth, logging (TODO)
│   │   └── mod.rs               # (placeholder)
│   ├── bin/
│   │   ├── cpu.rs               # (✅ DONE by TEAM-390)
│   │   ├── cuda.rs              # (✅ DONE by TEAM-390)
│   │   └── metal.rs             # (✅ DONE by TEAM-390)
│   ├── lib.rs                   # (✅ DONE by TEAM-390)
│   ├── error.rs                 # (✅ DONE by TEAM-390)
│   └── ...
└── Cargo.toml                   # (✅ DONE by TEAM-390)
```

---

## 🎯 How to Use These References

### For TEAM-391 (Planning)
1. Study LLM worker structure thoroughly
2. Map LLM worker modules to SD worker equivalents
3. Use Candle examples to estimate complexity
4. Create work packages that mirror LLM worker development

### For TEAM-392+ (Implementation)
1. **Before writing ANY code:**
   - Read the corresponding LLM worker file
   - Read the Candle example code
   - Understand the pattern

2. **While implementing:**
   - Follow LLM worker patterns
   - Adapt Candle example code
   - Use shared components where available

3. **After implementing:**
   - Verify structure matches LLM worker
   - Test with Candle examples as reference
   - Document any deviations

---

## 🚨 Common Mistakes to Avoid

### ❌ DON'T:
1. **Ignore LLM worker structure** - "I'll organize it differently"
2. **Reinvent device management** - "I'll write my own device init"
3. **Skip Candle examples** - "I'll figure it out myself"
4. **Create different patterns** - "HTTP endpoints should work differently"
5. **Duplicate shared code** - "I'll copy device.rs into SD worker"

### ✅ DO:
1. **Mirror LLM worker structure** - Same modules, same patterns
2. **Use shared components** - Device, heartbeat from shared crate
3. **Study Candle examples** - Working code is the best teacher
4. **Follow established patterns** - Request queue, generation engine, SSE
5. **Ask if unsure** - Better to ask than to diverge

---

## 📖 Reading Order (Recommended)

### Phase 1: Understanding (Before Planning)
1. Read LLM worker README: `bin/30_llm_worker_rbee/README.md`
2. Study LLM worker structure: `bin/30_llm_worker_rbee/src/`
3. Read Candle SD example: `reference/candle/candle-examples/examples/stable-diffusion/main.rs`
4. Review shared components: `bin/32_shared_worker_rbee/src/`

### Phase 2: Planning (TEAM-391)
1. Map LLM modules to SD equivalents
2. Identify what Candle examples provide
3. Plan work packages based on LLM worker phases
4. Document dependencies

### Phase 3: Implementation (TEAM-392+)
1. Read your team's instructions
2. Study relevant LLM worker files
3. Study relevant Candle example sections
4. Implement following the patterns
5. Test against Candle examples

---

## 🔗 Quick Links

| Purpose | Path |
|---------|------|
| **LLM Worker (Pattern)** | `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/` |
| **Candle SD Examples** | `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/` |
| **Candle SD3 Examples** | `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion-3/` |
| **Shared Components** | `/home/vince/Projects/llama-orch/bin/32_shared_worker_rbee/` |
| **SD Worker (Current)** | `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/` |

---

## ✅ Verification

Before implementing, verify you understand:
- [ ] LLM worker module structure
- [ ] Request queue pattern
- [ ] Generation engine pattern
- [ ] SSE streaming pattern
- [ ] How Candle loads SD models
- [ ] How Candle runs inference
- [ ] What shared components are available

**If you can't check all boxes, read the references again.**

---

**Remember: These references are not suggestions. They are requirements.**

**Follow the patterns. Use the examples. Don't reinvent the wheel.** 🎯
