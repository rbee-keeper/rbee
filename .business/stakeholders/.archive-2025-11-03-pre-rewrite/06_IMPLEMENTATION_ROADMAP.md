# rbee: Implementation Roadmap

**Audience:** Technical leaders, project managers, investors  
**Date:** November 2, 2025

---

## Current Status

**Development Progress:** 68% complete (42/62 BDD scenarios passing)

**What's Working Now:**
- ✅ Multi-machine orchestration (SSH-based)
- ✅ Heterogeneous backends (CUDA, Metal, CPU)
- ✅ Model downloading (HuggingFace)
- ✅ SSE token streaming
- ✅ OpenAI-compatible API
- ✅ Worker spawning and lifecycle
- ✅ Basic job routing

**What's In Progress:**
- 🚧 Lifecycle management (daemon start/stop/status)
- 🚧 Cascading shutdown
- 🚧 Worker cancellation endpoint

---

## Milestone Overview

| Milestone | Target | Status | Focus |
|-----------|--------|--------|-------|
| **M0** | Q4 2025 | 🚧 68% | Core orchestration |
| **M1** | Q1 2026 | 📋 Planned | Production-ready |
| **M2** | Q2 2026 | 📋 Planned | Rhai scheduler + Web UI |
| **M3** | Q3 2026 | 📋 Planned | Multi-modal support |
| **M4** | Q4 2026 | 📋 Planned | Multi-GPU & distributed |
| **M5** | 2027 | 🔮 Future | GPU marketplace |

---

## M0: Core Orchestration (Q4 2025)

**Goal:** Complete core orchestration features for single-modality (text) inference

**Status:** 🚧 68% complete (42/62 BDD scenarios passing)

### Features

**✅ Complete:**
- Multi-machine orchestration (SSH-based)
- Heterogeneous backends (CUDA, Metal, CPU)
- Model catalog (SQLite)
- Model provisioning (HuggingFace)
- Worker spawning (llm-worker-rbee variants)
- SSE streaming (token-by-token)
- HTTP APIs (queen, hive, worker)
- OpenAI-compatible API (basic)
- Backend detection (CUDA, Metal, CPU)
- Auto-update system

**🚧 In Progress:**
- Lifecycle management (daemon start/stop/status)
- Cascading shutdown (queen → hives → workers)
- SSH configuration management
- Worker cancellation endpoint
- Remaining BDD scenarios (20 scenarios)

**📋 Not Started:**
- Web UI (basic dashboard)
- Monitoring (Prometheus metrics)

### Timeline

**Week 1-2 (Nov 3-16):**
- ✅ Complete lifecycle management
- ✅ Complete cascading shutdown
- ✅ Complete SSH configuration

**Week 3-4 (Nov 17-30):**
- ✅ Complete worker cancellation
- ✅ Pass remaining BDD scenarios (62/62)
- ✅ Basic monitoring (Prometheus)

**Week 5-6 (Dec 1-14):**
- ✅ Basic web UI (dashboard)
- ✅ Documentation updates
- ✅ Release M0

**Target:** December 15, 2025

---

## M1: Production-Ready (Q1 2026)

**Goal:** Production-ready pool management and reliability

**Status:** 📋 Planned

### Features

**Pool Management:**
- Worker health monitoring (30s heartbeat)
- Idle timeout enforcement (5 minutes)
- Automatic worker restart on failure
- Worker pool scaling (min/max workers)

**Reliability:**
- Graceful degradation (worker failures)
- Circuit breaker pattern (failing workers)
- Retry logic (transient failures)
- Dead letter queue (failed jobs)

**Monitoring:**
- Prometheus metrics (comprehensive)
- Grafana dashboards (included)
- Alert rules (worker failures, queue depth)
- Performance metrics (latency, throughput)

**Security:**
- TLS support (HTTPS)
- API key authentication (bearer tokens)
- Rate limiting (per-customer)
- IP whitelisting

**Documentation:**
- Production deployment guide
- Monitoring and alerting guide
- Troubleshooting guide
- API reference (complete)

### Timeline

**January 2026:**
- Week 1-2: Pool management features
- Week 3-4: Reliability features

**February 2026:**
- Week 1-2: Monitoring and alerting
- Week 3-4: Security features

**March 2026:**
- Week 1-2: Documentation
- Week 3-4: Testing and release

**Target:** March 31, 2026

---

## M2: Rhai Scheduler + Web UI (Q2 2026)

**Goal:** Intelligent orchestration with user-scriptable routing

**Status:** 📋 Planned

### Features

**Rhai Programmable Scheduler:**
- User-scriptable routing (Rhai scripts)
- Multi-tenancy support (customer tiers)
- Quota enforcement (daily limits)
- Priority scheduling (enterprise > pro > free)
- Cost optimization (route by GPU cost)
- Time-based routing (peak vs off-peak)

**Web UI (Full Dashboard):**
- Hive management (add, remove, status)
- Worker management (spawn, stop, monitor)
- Model management (download, delete, catalog)
- Scheduler editor (Rhai script editor)
- Real-time monitoring (GPU utilization, queue depth)
- Customer management (tiers, quotas, usage)

**API Enhancements:**
- Customer API keys (per-customer tokens)
- Usage tracking (tokens, requests, costs)
- Billing integration (usage export)
- Webhook support (job completion, failures)

**Examples & Templates:**
- Rhai script templates (multi-tenancy, cost optimization)
- Deployment examples (single-machine, multi-machine)
- Integration examples (Zed, Cursor, custom apps)

### Timeline

**April 2026:**
- Week 1-2: Rhai scheduler implementation
- Week 3-4: Multi-tenancy and quotas

**May 2026:**
- Week 1-2: Web UI (full dashboard)
- Week 3-4: Scheduler editor and templates

**June 2026:**
- Week 1-2: API enhancements
- Week 3-4: Testing and release

**Target:** June 30, 2026

---

## M3: Multi-Modal Support (Q3 2026)

**Goal:** Support text, images, audio, and video generation

**Status:** 📋 Planned

### Features

**Image Generation:**
- Stable Diffusion workers (SDXL, SD3)
- Image-to-image (img2img)
- Inpainting and outpainting
- ControlNet support
- LoRA support

**Audio Generation:**
- Text-to-Speech (TTS) workers
- Audio transcription (Whisper)
- Audio-to-audio (voice conversion)
- Music generation

**Video Generation:**
- Text-to-video workers
- Image-to-video (AnimateDiff)
- Video-to-video (style transfer)

**Multi-Modal Routing:**
- Capability-based routing (text, image, audio, video)
- Protocol-aware orchestration (SSE, JSON, Binary)
- Multi-modal Rhai scripts

**OpenAI Compatibility:**
- `/v1/images/generations` (DALL-E compatible)
- `/v1/audio/transcriptions` (Whisper compatible)
- `/v1/audio/speech` (TTS compatible)

### Timeline

**July 2026:**
- Week 1-2: Image generation workers (Stable Diffusion)
- Week 3-4: Audio workers (TTS, Whisper)

**August 2026:**
- Week 1-2: Video generation workers
- Week 3-4: Multi-modal routing

**September 2026:**
- Week 1-2: OpenAI compatibility (images, audio)
- Week 3-4: Testing and release

**Target:** September 30, 2026

---

## M4: Multi-GPU & Distributed (Q4 2026)

**Goal:** Distributed inference and multi-GPU support

**Status:** 📋 Planned

### Features

**Distributed Inference:**
- Tensor parallelism (split model across GPUs)
- Pipeline parallelism (split layers across GPUs)
- Worker groups (coordinator + shards)
- Transparent client interaction

**Multi-GPU Support:**
- Multi-GPU workers (single machine)
- GPU affinity (pin workers to GPUs)
- VRAM pooling (share VRAM across GPUs)

**Advanced Scheduling:**
- Cross-node load balancing
- GPU utilization optimization
- Latency-aware routing
- Geo-aware routing (EU vs US)

**Performance:**
- Batch inference (multiple requests)
- Request coalescing (same model)
- Speculative decoding
- KV cache optimization

### Timeline

**October 2026:**
- Week 1-2: Tensor parallelism
- Week 3-4: Pipeline parallelism

**November 2026:**
- Week 1-2: Worker groups
- Week 3-4: Multi-GPU support

**December 2026:**
- Week 1-2: Advanced scheduling
- Week 3-4: Testing and release

**Target:** December 31, 2026

---

## M5: GPU Marketplace (2027)

**Goal:** Global GPU marketplace (Airbnb for GPUs)

**Status:** 🔮 Future

### Features

**Provider Registration:**
- Provider onboarding (KYC, verification)
- GPU verification (proof of hardware)
- Pricing configuration (per-token, per-hour)
- Geo-tagging (EU, US, Asia)

**Customer Platform:**
- Unified API endpoint (marketplace)
- Provider selection (manual or automatic)
- SLA enforcement (uptime, latency)
- Billing integration (Stripe)

**Platform Features:**
- Task-based billing (pay per task)
- Federated placement (route to best provider)
- Provider reputation (ratings, reviews)
- Provider dashboard (earnings, utilization)
- Customer dashboard (usage, costs, SLA)

**Compliance:**
- GDPR enforcement (EU-only routing)
- Data residency verification
- Audit logging (platform-wide)
- Provider compliance checks

### Timeline

**Q1 2027:**
- Provider registration and verification
- Platform infrastructure

**Q2 2027:**
- Customer platform and API
- Billing integration

**Q3 2027:**
- SLA enforcement and monitoring
- Provider/customer dashboards

**Q4 2027:**
- Launch and scaling

**Target:** December 31, 2027

---

## Feature Dependency Graph

```
M0 (Core Orchestration)
  ↓
M1 (Production-Ready)
  ↓
M2 (Rhai Scheduler + Web UI)
  ↓
M3 (Multi-Modal)
  ↓
M4 (Multi-GPU & Distributed)
  ↓
M5 (GPU Marketplace)
```

**Critical path:** M0 → M1 → M2 (everything else depends on these)

---

## Resource Requirements

### M0 (Current)
- **Team:** 1 developer (AI-assisted)
- **Time:** 6 weeks
- **Cost:** ~$10K (labor)

### M1 (Production-Ready)
- **Team:** 1-2 developers
- **Time:** 3 months
- **Cost:** ~$30K-60K (labor)

### M2 (Rhai Scheduler + Web UI)
- **Team:** 2-3 developers (1 backend, 1-2 frontend)
- **Time:** 3 months
- **Cost:** ~$60K-90K (labor)

### M3 (Multi-Modal)
- **Team:** 2-3 developers (ML specialists)
- **Time:** 3 months
- **Cost:** ~$60K-90K (labor)

### M4 (Multi-GPU & Distributed)
- **Team:** 2-3 developers (distributed systems)
- **Time:** 3 months
- **Cost:** ~$60K-90K (labor)

### M5 (GPU Marketplace)
- **Team:** 5-10 developers (full-stack, platform)
- **Time:** 12 months
- **Cost:** ~$500K-1M (labor + infrastructure)

**Total (M0-M4):** ~$220K-340K over 18 months  
**Total (M0-M5):** ~$720K-1.34M over 30 months

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Candle framework limitations** | Medium | High | Fallback to llama.cpp adapters |
| **Multi-GPU complexity** | High | Medium | Start with tensor parallelism only |
| **Rhai performance** | Low | Medium | Benchmark and optimize early |
| **SSE streaming issues** | Low | High | Extensive testing in M0 |

### Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Low adoption (consumer)** | Medium | Medium | Focus on business use case |
| **Competition (Ollama)** | High | Medium | Differentiate on multi-machine |
| **GPL license concerns** | Low | High | Offer commercial license option |
| **Market timing** | Medium | High | Ship M1 by Q1 2026 |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Developer burnout** | Medium | High | Hire additional developers |
| **Scope creep** | High | Medium | Strict milestone boundaries |
| **Documentation lag** | High | Low | Document as you build |
| **Community support** | Medium | Medium | Build community early |

---

## Success Metrics

### M0 (Core Orchestration)
- ✅ 62/62 BDD scenarios passing
- ✅ 100 GitHub stars
- ✅ 10 community contributors
- ✅ 5 production deployments

### M1 (Production-Ready)
- ✅ 99.9% uptime (SLA)
- ✅ 500 GitHub stars
- ✅ 50 production deployments
- ✅ 10 business customers

### M2 (Rhai Scheduler + Web UI)
- ✅ 1,000 GitHub stars
- ✅ 100 production deployments
- ✅ 50 business customers
- ✅ 10 Rhai script templates

### M3 (Multi-Modal)
- ✅ 2,000 GitHub stars
- ✅ 500 production deployments
- ✅ 100 business customers
- ✅ Text + images + audio support

### M4 (Multi-GPU & Distributed)
- ✅ 5,000 GitHub stars
- ✅ 1,000 production deployments
- ✅ 200 business customers
- ✅ 405B model support

### M5 (GPU Marketplace)
- ✅ 10,000 GitHub stars
- ✅ 50 GPU providers
- ✅ 1,000 marketplace customers
- ✅ $1M+ annual platform revenue

---

## Go-to-Market Strategy

### M0-M1: Community Building (Q4 2025 - Q1 2026)
- **Focus:** Open source community
- **Channels:** GitHub, Reddit, Hacker News
- **Goal:** 1,000 users, 100 GitHub stars

### M2: Business Launch (Q2 2026)
- **Focus:** Business customers (self-hosted)
- **Channels:** LinkedIn, tech conferences, direct sales
- **Goal:** 50 business customers, $50K MRR

### M3: Multi-Modal Launch (Q3 2026)
- **Focus:** Content creators, AI platforms
- **Channels:** Product Hunt, tech blogs, partnerships
- **Goal:** 100 business customers, $100K MRR

### M4: Enterprise Launch (Q4 2026)
- **Focus:** Enterprise customers, large deployments
- **Channels:** Enterprise sales, partnerships
- **Goal:** 200 business customers, $200K MRR

### M5: Marketplace Launch (2027)
- **Focus:** GPU providers and customers
- **Channels:** PR, partnerships, conferences
- **Goal:** 50 providers, 1,000 customers, $200K MRR

---

## Key Decisions

### Decision 1: Rhai vs Lua (M2)
**Options:**
- Rhai (Rust-native, type-safe)
- Lua (mature, widely known)

**Recommendation:** Rhai
- ✅ Rust-native (no FFI overhead)
- ✅ Type-safe (fewer runtime errors)
- ✅ Smaller attack surface

### Decision 2: Web UI Framework (M2)
**Options:**
- React (mature, large ecosystem)
- Vue (simpler, faster)
- Svelte (fastest, smallest)

**Recommendation:** React
- ✅ Already using React in current UI
- ✅ Largest ecosystem
- ✅ Best TypeScript support

### Decision 3: Distributed Inference (M4)
**Options:**
- Tensor parallelism (split model)
- Pipeline parallelism (split layers)
- Both

**Recommendation:** Both
- ✅ Tensor parallelism for large models (405B)
- ✅ Pipeline parallelism for throughput
- ✅ Candle supports both

### Decision 4: Marketplace Platform (M5)
**Options:**
- Build from scratch
- Use existing platform (Stripe, etc.)

**Recommendation:** Build from scratch
- ✅ Full control over features
- ✅ Custom billing logic
- ✅ Better margins

---

## Next Steps

### Immediate (This Week)
1. Complete lifecycle management
2. Complete cascading shutdown
3. Pass 50/62 BDD scenarios

### Short-Term (This Month)
1. Complete M0 (62/62 scenarios)
2. Release M0 (December 15, 2025)
3. Start M1 planning

### Medium-Term (Next Quarter)
1. Complete M1 (production-ready)
2. Launch business use case
3. Get first 10 business customers

### Long-Term (Next Year)
1. Complete M2 (Rhai scheduler)
2. Complete M3 (multi-modal)
3. Complete M4 (multi-GPU)
4. Plan M5 (marketplace)

---

## Conclusion

rbee has a clear roadmap from current state (68% complete) to production-ready platform (M1) to multi-modal marketplace (M5).

**Key milestones:**
- **M0 (Q4 2025):** Core orchestration complete
- **M1 (Q1 2026):** Production-ready
- **M2 (Q2 2026):** Rhai scheduler + Web UI
- **M3 (Q3 2026):** Multi-modal support
- **M4 (Q4 2026):** Multi-GPU & distributed
- **M5 (2027):** GPU marketplace

**Total timeline:** 30 months (2.5 years)  
**Total cost:** $720K-1.34M  
**Expected revenue (Year 3):** $2.4M+

**ROI:** Positive by Year 2

---

**Read the use cases:**
- [Consumer Use Case](02_CONSUMER_USE_CASE.md)
- [Business Use Case](03_BUSINESS_USE_CASE.md)
- [Revenue Models](05_REVENUE_MODELS.md)
