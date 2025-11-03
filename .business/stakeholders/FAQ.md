# rbee: Frequently Asked Questions

**Date:** November 3, 2025  
**Version:** 3.0

---

## General Questions

### Q: What is rbee?
**A:** rbee (pronounced "are-bee") turns heterogeneous GPU infrastructure into a unified colony with one simple API. Use ALL your GPUs across ALL your machines together.

### Q: How is rbee pronounced?
**A:** "are-bee" (like the letter R + bee 🐝)

### Q: Is rbee bee-themed?
**A:** Yes! 🐝
- Queen = Orchestrator (makes routing decisions)
- Hives = Worker nodes (across machines)
- Workers = Executors (do inference on GPUs)
- Beekeeper = Manager (manages the colony)

---

## Consumer Questions

### Q: Do I need multiple computers?
**A:** No! rbee works great with ONE computer with multiple GPUs. But if you HAVE multiple computers, rbee lets you use them ALL together.

### Q: What if I only have a Mac?
**A:** Perfect! rbee supports Metal (Apple Silicon). Use your Mac's unified memory for large models.

### Q: Can I use rbee with Zed/Cursor/Continue.dev?
**A:** Yes! rbee is OpenAI-compatible. Just point your IDE to `http://localhost:7833/v1`.

### Q: Do I need to learn Rhai scripting?
**A:** No! The GUI works great for most users. Rhai is for power users who want custom routing.

### Q: What about Windows?
**A:** rbee supports Windows with NVIDIA GPUs (CUDA). Cross-platform support is built-in.

### Q: Is rbee free for consumers?
**A:** YES! 100% free forever. GPL-3.0 (binaries) + MIT (infrastructure). No catch, no limits.

---

## Business Questions

### Q: Can rbee handle 1000+ customers?
**A:** Yes! Multi-tenancy is built-in via Rhai scripts. Route by customer tier, quota, priority.

### Q: Is rbee GDPR compliant?
**A:** **Basic (free):** Simple audit logging (MIT license)  
**Full (€249):** Complete GDPR compliance - data lineage, right to erasure, automated reporting, 7-year retention.

### Q: What's the difference between free and premium?
**A:**

**Free (Self-Hosted):**
- ✅ Multi-tenancy (Rhai scripts)
- ✅ Basic audit logging
- ✅ Quota enforcement
- ✅ Keep 100% of revenue
- ❌ Advanced RHAI scheduler
- ❌ Deep telemetry
- ❌ Full GDPR compliance

**Premium (€129-499 lifetime):**
- ✅ Everything in free
- ✅ Advanced RHAI scheduling (40-60% better GPU utilization)
- ✅ Deep telemetry (Premium Worker)
- ✅ Full GDPR compliance (€249)

### Q: Why does Premium Worker require Premium Queen?
**A:** Premium Worker collects telemetry (GPU metrics, performance data). This telemetry is sent to Premium Queen for intelligent routing decisions.

**Without Premium Queen:** Telemetry has nowhere to go = useless data collection.

**With Premium Queen:** Telemetry enables 40-60% higher GPU utilization through smart routing.

**That's why Worker is only sold in bundles with Queen.**

---

## Premium Products Questions

### Q: Are premium products a subscription?
**A:** NO! Pay once, own forever. No recurring fees. Free updates.

**Example:** €499 (complete bundle) - pay once in 2026, use forever.

### Q: Can I upgrade later?
**A:** Yes! Buy Premium Queen (€129) now, add Premium Worker later (pay difference: €279 - €129 = €150).

### Q: Which bundle should I buy?

**If you want best GPU utilization:**
→ Queen + Worker (€279) - Most popular ⭐

**If you need GDPR compliance:**
→ Queen + Audit (€349) or Audit solo (€249)

**If you want everything:**
→ Complete Bundle (€499) - Best value ⭐⭐ - Save €58

**If you have existing monitoring:**
→ Premium Queen solo (€129) - Works with basic workers

### Q: When do premium products launch?
**A:** Q2 2026 (alongside M2 milestone - RHAI scheduler + Web UI)

---

## Technical Questions

### Q: What backends are supported?
**A:** 
- ✅ CUDA (NVIDIA GPUs - Windows/Linux)
- ✅ Metal (Apple Silicon - Mac M1/M2/M3)
- ✅ CPU (fallback - any machine)

**Unique advantage:** Mix them ALL in ONE colony!

### Q: Does rbee require Kubernetes?
**A:** NO! rbee uses SSH (like Ansible). Homelab-friendly. No K8s expertise required.

### Q: Can I use my own fine-tuned models?
**A:** YES! rbee supports custom models. Upload to any hive and use via API.

### Q: What ports does rbee use?
**A:**
- Queen: 7833 (default)
- Hive: 7835 (default)
- Workers: 9300+ (dynamic)

### Q: How does rbee compare to Ollama?
**A:**

**Ollama:** Great for ONE machine  
**rbee:** For ALL your machines together

| Feature | rbee | Ollama |
|---------|------|--------|
| Multi-machine | ✅ | ❌ |
| Heterogeneous | ✅ CUDA+Metal | ❌ CUDA or Metal |
| Custom routing | ✅ Rhai | ❌ |
| Multi-modal | ✅ (Q1 2026) | ❌ |

**Use Ollama if:** You have ONE machine  
**Use rbee if:** You have MULTIPLE machines

### Q: How does rbee compare to vLLM?
**A:**

**vLLM:** Library for ONE model  
**rbee:** Platform for MANY models across MANY machines

| Feature | rbee | vLLM |
|---------|------|------|
| Multi-machine orchestration | ✅ | ❌ |
| Custom routing | ✅ Rhai (no redeploy) | ❌ Python (redeploy) |
| Multi-tenancy | ✅ Built-in | ❌ Build yourself |
| GDPR | ✅ Built-in | ❌ Build yourself |

**Use vLLM if:** You need ONE model, maximum performance  
**Use rbee if:** You need MANY models, multi-machine orchestration

---

## Licensing Questions

### Q: What license is rbee?
**A:** Multi-license architecture:
- **User binaries:** GPL-3.0 (free forever, protects from forks)
- **Infrastructure/contracts:** MIT (allows premium to link without contamination)
- **Premium binaries:** Proprietary (closed source, revenue)

### Q: Can I use rbee commercially?
**A:** YES! GPL-3.0 and MIT both allow commercial use.

**Free version:** Keep 100% of revenue (self-hosted)  
**Premium version:** €129-499 lifetime

### Q: Can I modify rbee?
**A:** YES! GPL-3.0 and MIT are both open source. Modify as needed.

**But:** Premium binaries are proprietary (closed source).

---

## Timeline Questions

### Q: When is rbee production-ready?
**A:** M1 milestone (Q1 2026)

**Timeline:**
- M0 (Q4 2025): Core orchestration (text) - IN PROGRESS
- M1 (Q1 2026): Production-ready - **THIS IS IT**
- M2 (Q2 2026): RHAI + Web UI + Premium launch
- M3 (Q1 2026): Multi-modal (images, audio, video)

### Q: Can I use rbee now?
**A:** Yes! M0 (text inference) is working. Production-ready in Q1 2026.

### Q: When are premium products available?
**A:** Q2 2026 (with M2 milestone)

---

## Pricing Questions

### Q: How much does rbee cost?
**A:**

**Consumer:** $0 (free forever)

**Business:**
- Self-hosted: $0 (free, keep 100% of revenue)
- Premium: €129-499 (one-time, lifetime)

### Q: What are the premium prices?
**A:**
- Premium Queen: €129
- GDPR Auditing: €249
- Queen + Worker: €279 (save €29) ⭐
- Queen + Audit: €349 (save €29)
- Complete Bundle: €499 (save €58) ⭐⭐

### Q: Is €499 per year or lifetime?
**A:** LIFETIME! Pay once in 2026, use forever. No recurring fees.

### Q: How does €499 compare to alternatives?
**A:**
- **Together.ai:** $72K+/year (per-token pricing)
- **Build from scratch:** $500K-1.4M (Year 1)
- **rbee premium:** €499 (lifetime)

**rbee is 99.97% cheaper than building from scratch.**

---

## Support Questions

### Q: What support is available?
**A:**
- **Free version:** Community support (GitHub Discussions)
- **Premium version:** Community support + documentation
- **Enterprise:** Contact for custom support contracts

### Q: Where can I get help?
**A:**
- GitHub Discussions
- Documentation: `../../README.md`
- Architecture docs: `../../.arch/README.md`

---

## Roadmap Questions

### Q: What's coming next?
**A:**
- **Q4 2025 (M0):** Core orchestration (text inference)
- **Q1 2026 (M1):** Production-ready
- **Q2 2026 (M2):** RHAI scheduler + Web UI + Premium launch
- **Q1 2026 (M3):** Multi-modal (images, audio, video)
- **Q4 2026 (M4):** Multi-GPU & distributed
- **2027 (M5):** GPU marketplace

[Full roadmap → Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)

---

## Still Have Questions?

- **GitHub Discussions:** Ask the community
- **Documentation:** See main [README.md](../../README.md)
- **Business inquiries:** Contact via GitHub

---

**🐝 Welcome to the rbee colony!**
