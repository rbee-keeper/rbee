# TEAM-401: Phase 11 - Polish & Optimization

**Team:** TEAM-401  
**Phase:** 11 - Final Polish & Production Ready  
**Duration:** 50 hours  
**Dependencies:** TEAM-398 (testing), TEAM-400 (UI complete)  
**Parallel Work:** None (final phase)

---

## 🎯 Mission

Optimize performance, complete documentation, prepare deployment, and deliver production-ready SD worker. This is the final phase before release.

---

## 📦 What You're Delivering

### Major Deliverables (4 areas)

1. **Performance Optimization** (~20 hours)
   - Flash attention (CUDA)
   - FP16 precision
   - Model quantization
   - Memory optimization

2. **Documentation** (~15 hours)
   - API reference
   - User guide
   - Developer guide
   - Deployment guide

3. **Deployment** (~10 hours)
   - Dockerfile
   - Systemd service
   - Deployment scripts
   - Monitoring setup

4. **Final Polish** (~5 hours)
   - Bug fixes
   - UI polish
   - Error messages
   - Logging

---

## 📋 Task Breakdown

### Week 1: Performance Optimization (20 hours)

**Day 1: Flash Attention (8 hours)**
- [ ] Study flash attention requirements (1 hour)
- [ ] Add flash-attn feature flag (1 hour)
- [ ] Implement flash attention in UNet (3 hours)
- [ ] Benchmark improvements (2 hours)
- [ ] Document usage (1 hour)

**Day 2: FP16 & Quantization (8 hours)**
- [ ] Add FP16 support (CUDA/Metal) (3 hours)
- [ ] Test FP16 quality (2 hours)
- [ ] Add model quantization (2 hours)
- [ ] Benchmark memory usage (1 hour)

**Day 3: Memory Optimization (4 hours)**
- [ ] Profile memory usage (1 hour)
- [ ] Optimize tensor allocations (2 hours)
- [ ] Add model offloading (1 hour)

---

### Week 2: Documentation (15 hours)

**Day 4: API Documentation (5 hours)**
- [ ] Create docs/API.md (2 hours)
- [ ] Document all endpoints (2 hours)
- [ ] Add request/response examples (1 hour)

**Day 5: User Guide (5 hours)**
- [ ] Create docs/USER_GUIDE.md (2 hours)
- [ ] Add installation instructions (1 hour)
- [ ] Add usage examples (1 hour)
- [ ] Add troubleshooting (1 hour)

**Day 6: Developer Guide (5 hours)**
- [ ] Create docs/DEVELOPER_GUIDE.md (2 hours)
- [ ] Document architecture (1 hour)
- [ ] Add contribution guide (1 hour)
- [ ] Add testing guide (1 hour)

---

### Week 3: Deployment & Polish (15 hours)

**Day 7: Deployment (10 hours)**
- [ ] Create Dockerfile (3 hours)
- [ ] Create systemd service (2 hours)
- [ ] Create deployment scripts (2 hours)
- [ ] Add monitoring setup (2 hours)
- [ ] Test deployment (1 hour)

**Day 8: Final Polish (5 hours)**
- [ ] Fix remaining bugs (2 hours)
- [ ] Improve error messages (1 hour)
- [ ] Add logging (1 hour)
- [ ] Final testing (1 hour)

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] Flash attention working (CUDA, Ampere+ GPUs)
- [ ] FP16 precision working (CUDA/Metal)
- [ ] Memory usage optimized (<8GB VRAM for SD 1.5)
- [ ] All documentation complete
- [ ] Dockerfile builds successfully
- [ ] Systemd service works
- [ ] Deployment scripts tested
- [ ] Monitoring configured
- [ ] All bugs fixed
- [ ] Production ready
- [ ] Clean compilation (0 warnings)

---

## 🔧 Implementation Notes

### Flash Attention

```rust
// src/backend/inference.rs
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;

impl CandleSDBackend {
    fn forward_with_attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            // Use flash attention (2-4x faster, 50% less memory)
            flash_attn(query, key, value, None, false)
        }
        
        #[cfg(not(feature = "flash-attn"))]
        {
            // Standard attention
            self.standard_attention(query, key, value)
        }
    }
}
```

### FP16 Precision

```rust
// src/backend/inference.rs
impl CandleSDBackend {
    pub fn with_fp16(mut self, use_fp16: bool) -> Self {
        self.use_fp16 = use_fp16;
        self
    }
    
    fn maybe_convert_to_fp16(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.use_fp16 {
            tensor.to_dtype(DType::F16)
        } else {
            Ok(tensor.clone())
        }
    }
}
```

### Dockerfile

```dockerfile
# Dockerfile
FROM rust:1.75 as builder

# Install CUDA (for CUDA builds)
RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build with CUDA and flash attention
RUN cargo build --release --bin sd-worker-cuda --features cuda,flash-attn

FROM nvidia/cuda:12.0-runtime

COPY --from=builder /app/target/release/sd-worker-cuda /usr/local/bin/

# Download default model
RUN mkdir -p /models
ENV HF_HOME=/models

EXPOSE 8600

CMD ["sd-worker-cuda", "--model", "sd-v1-5", "--port", "8600"]
```

### Systemd Service

```ini
# /etc/systemd/system/sd-worker.service
[Unit]
Description=Stable Diffusion Worker
After=network.target

[Service]
Type=simple
User=sdworker
WorkingDirectory=/opt/sd-worker
ExecStart=/usr/local/bin/sd-worker-cuda \
    --model sd-v1-5 \
    --port 8600 \
    --queen-url http://localhost:8500
Restart=always
RestartSec=10

# Resource limits
MemoryMax=16G
CPUQuota=400%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sd-worker

[Install]
WantedBy=multi-user.target
```

### Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

echo "🚀 Deploying SD Worker..."

# Build
echo "📦 Building..."
cargo build --release --bin sd-worker-cuda --features cuda,flash-attn

# Install binary
echo "📥 Installing binary..."
sudo cp target/release/sd-worker-cuda /usr/local/bin/

# Install systemd service
echo "⚙️  Installing service..."
sudo cp scripts/sd-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sd-worker
sudo systemctl restart sd-worker

# Check status
echo "✅ Checking status..."
sudo systemctl status sd-worker

echo "🎉 Deployment complete!"
```

---

## 📚 Documentation Structure

### docs/API.md

```markdown
# SD Worker API Reference

## Endpoints

### POST /v1/jobs
Submit a generation job.

**Request:**
```json
{
  "prompt": "a photo of a cat",
  "negative_prompt": "blurry, bad quality",
  "steps": 20,
  "guidance_scale": 7.5,
  "seed": 42,
  "width": 512,
  "height": 512
}
```

**Response:**
```json
{
  "job_id": "job_abc123"
}
```

### GET /v1/jobs/:id/stream
Stream generation progress via SSE.

**Events:**
- `progress`: Step progress
- `complete`: Final image (base64)
- `error`: Error message
- `[DONE]`: Stream end marker
```

### docs/USER_GUIDE.md

```markdown
# SD Worker User Guide

## Installation

### CPU (Linux/macOS/Windows)
```bash
cargo install --path . --bin sd-worker-cpu --features cpu
```

### CUDA (Linux, NVIDIA GPU)
```bash
cargo install --path . --bin sd-worker-cuda --features cuda
```

### Metal (macOS, Apple Silicon)
```bash
cargo install --path . --bin sd-worker-metal --features metal
```

## Quick Start

1. Start the worker:
```bash
sd-worker-cpu --model sd-v1-5 --port 8600
```

2. Generate an image:
```bash
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a photo of a cat","steps":20}'
```

3. Stream progress:
```bash
curl -N http://localhost:8600/v1/jobs/job_abc123/stream
```

## Troubleshooting

### Model download fails
- Check internet connection
- Check HuggingFace Hub status
- Try manual download

### Out of memory
- Reduce image size (512x512 instead of 1024x1024)
- Use fewer steps (10-20 instead of 50)
- Enable FP16 (CUDA/Metal)
```

---

## 🚨 Common Pitfalls

1. **Flash Attention Requirements**
   - Requires Ampere or newer GPU (RTX 30xx, 40xx, A100)
   - Graceful fallback to standard attention

2. **FP16 Quality**
   - May reduce quality slightly
   - Test before deploying

3. **Docker GPU Access**
   - Requires nvidia-docker
   - Must pass GPU devices

4. **Systemd Permissions**
   - Service user needs GPU access
   - Add to `video` group

---

## 🎯 Final Deliverables

**Documentation:**
- [ ] docs/API.md - Complete API reference
- [ ] docs/USER_GUIDE.md - Installation and usage
- [ ] docs/DEVELOPER_GUIDE.md - Architecture and development
- [ ] docs/DEPLOYMENT.md - Production deployment
- [ ] README.md - Updated with all features

**Deployment:**
- [ ] Dockerfile - Multi-stage build
- [ ] docker-compose.yml - Easy deployment
- [ ] scripts/deploy.sh - Deployment script
- [ ] scripts/sd-worker.service - Systemd service
- [ ] scripts/monitor.sh - Health monitoring

**Performance:**
- [ ] Flash attention implemented
- [ ] FP16 support added
- [ ] Memory optimized
- [ ] Benchmarks documented

**Quality:**
- [ ] All bugs fixed
- [ ] Error messages improved
- [ ] Logging comprehensive
- [ ] Production tested

---

## 📊 Progress Tracking

- [ ] Week 1: Performance optimization complete
- [ ] Week 2: Documentation complete
- [ ] Week 3: Deployment ready, production ready

---

## 🎉 Project Complete!

**When you're done:**

1. ✅ All 11 phases complete (TEAM-391 through TEAM-401)
2. ✅ SD worker fully functional
3. ✅ Production ready
4. ✅ Documented
5. ✅ Deployed

**Congratulations! You've built a complete Stable Diffusion worker!** 🚀

---

**TEAM-401: You're bringing it home. Make it production-ready and ship it!** 🎯
