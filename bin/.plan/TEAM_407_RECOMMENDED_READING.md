# TEAM-416: Recommended Reading List for Future Teams

**Created by:** TEAM-415  
**Date:** 2025-11-05  
**Purpose:** Essential documents for teams working on marketplace, filtering, and SEO  
**Status:** 📚 REFERENCE GUIDE

---

## 🎯 Quick Start (Read These First)

### 1. **TEAM_415_FIX_MARKETPLACE_PIPELINE.md** (CRITICAL)
**Location:** `/bin/.plan/TEAM_415_FIX_MARKETPLACE_PIPELINE.md`

**Why Read:** Explains the correct architecture for marketplace data flow

**Key Takeaways:**
- ✅ Correct: Next.js → @rbee/marketplace-node → HuggingFace API
- ❌ Wrong: Next.js → direct fetch → HuggingFace API
- marketplace-node is the single source of truth
- All HuggingFace calls MUST go through marketplace-node

**When to Read:** Before touching ANY marketplace code

---

### 2. **TEAM_413_MARKETPLACE_FILTERING_CHECKLIST.md** (NEXT PRIORITY)
**Location:** `/bin/.plan/TEAM_413_MARKETPLACE_FILTERING_CHECKLIST.md`

**Why Read:** Complete implementation plan for compatibility filtering

**Key Takeaways:**
- 5 phases: Compatibility Rules → Filter API → Filter SSG → Update UI → Testing
- "If we don't support it, it doesn't exist" philosophy
- Only compatible models get static pages (SEO optimization)
- SafeTensors only in Phase 1, GGUF in Phase 2

**When to Read:** Before implementing model compatibility filtering

---

### 3. **TEAM_414_MODEL_PAGE_SEO_STRATEGY.md** (SEO FOCUS)
**Location:** `/bin/.plan/TEAM_414_MODEL_PAGE_SEO_STRATEGY.md`

**Why Read:** Complete SEO strategy for model pages

**Key Takeaways:**
- 9 sections on every model page (hero, quick start, comparison, etc.)
- "Download rbee" CTA in 3 places
- Comparison tables (rbee vs Ollama vs LM Studio)
- FAQ section with rich snippets
- Expected: 5,000+ monthly downloads from SEO

**When to Read:** Before implementing model page UI

---

## 📊 Architecture & Strategy Documents

### 4. **ARCHITECTURE_EXPANSION_CHECKLIST.md** (TECHNICAL DEPTH)
**Location:** `/bin/30_llm_worker_rbee/docs/ARCHITECTURE_EXPANSION_CHECKLIST.md`

**Why Read:** Understand rbee's model architecture support strategy

**Key Takeaways:**
- Leverages candle-transformers library
- Phased rollout: Llama → Mistral → Phi → Qwen → Gemma
- GGUF support is CRITICAL for competitiveness
- Each architecture has specific implementation pattern

**When to Read:** Before adding new model architecture support

---

### 5. **TEAM_406_COMPETITIVE_RESEARCH.md** (MARKET POSITION)
**Location:** `/bin/.plan/TEAM_406_COMPETITIVE_RESEARCH.md`

**Why Read:** Understand rbee's competitive advantages and gaps

**Key Takeaways:**
- rbee advantages: Multi-machine, heterogeneous hardware, advanced filtering
- Ollama: 100+ models, GGUF-only, single-machine
- LM Studio: 50+ models, basic search, desktop-only
- GGUF support is the #1 competitive gap

**When to Read:** Before making product/feature decisions

---

### 6. **TEAM_406_FINAL_SUMMARY.md** (CONTEXT)
**Location:** `/bin/.plan/TEAM_406_FINAL_SUMMARY.md`

**Why Read:** Summary of all TEAM-406 decisions and research

**Key Takeaways:**
- Key decisions: CUDA-only, no CPU fallback, SafeTensors first
- Marketplace filtering strategy defined
- Architecture expansion roadmap
- Next steps for subsequent teams

**When to Read:** For overall project context

---

## 🛠️ Implementation Guides

### 7. **TEAM_415_IMPLEMENTATION_COMPLETE.md** (WHAT WAS DONE)
**Location:** `/bin/.plan/TEAM_415_IMPLEMENTATION_COMPLETE.md`

**Why Read:** See what TEAM-415 actually implemented

**Key Takeaways:**
- Created marketplace-node package (centralized HuggingFace client)
- Updated marketplace app to use marketplace-node
- Deleted old direct fetch code
- Fixed architecture pipeline

**When to Read:** To understand current state of marketplace codebase

---

## 📁 Code Structure Reference

### Key Files to Know

#### Marketplace Node Package
```
/frontend/packages/marketplace-node/
├── src/
│   ├── index.ts           # Main exports, conversion functions
│   ├── huggingface.ts     # HuggingFace API client
│   └── types.ts           # Shared TypeScript types
├── dist/                  # Built TypeScript
└── wasm/                  # WASM bindings (future)
```

#### Marketplace App
```
/frontend/apps/marketplace/
├── app/
│   ├── models/
│   │   ├── [slug]/page.tsx    # Model detail page (SSG)
│   │   └── page.tsx           # Model list page (SSG)
│   ├── search/page.tsx        # Dynamic search (client-side)
│   └── api/models/route.ts    # API route (fallback)
├── components/
│   └── ModelTableWithRouting.tsx  # Reusable table component
└── lib/
    └── slugify.ts         # URL slug utilities
```

#### Marketplace SDK (Rust)
```
/bin/99_shared_crates/marketplace-sdk/
├── src/
│   ├── lib.rs             # Main entry, WASM init
│   ├── huggingface.rs     # Native Rust HuggingFace client
│   └── types.rs           # Rust types
└── Cargo.toml             # Rust dependencies
```

---

## 🎓 Learning Path by Role

### Frontend Developer (Next.js/TypeScript)
1. **Start:** TEAM_415_FIX_MARKETPLACE_PIPELINE.md
2. **Then:** TEAM_414_MODEL_PAGE_SEO_STRATEGY.md
3. **Then:** TEAM_413_MARKETPLACE_FILTERING_CHECKLIST.md
4. **Reference:** Code structure above

**Focus:** Understand data flow, implement UI components, SEO optimization

---

### Backend Developer (Rust)
1. **Start:** ARCHITECTURE_EXPANSION_CHECKLIST.md
2. **Then:** TEAM_406_COMPETITIVE_RESEARCH.md
3. **Then:** marketplace-sdk source code
4. **Reference:** candle-transformers library

**Focus:** Add model architecture support, GGUF implementation

---

### Product Manager / Designer
1. **Start:** TEAM_406_COMPETITIVE_RESEARCH.md
2. **Then:** TEAM_414_MODEL_PAGE_SEO_STRATEGY.md
3. **Then:** TEAM_406_FINAL_SUMMARY.md

**Focus:** Competitive positioning, user experience, feature prioritization

---

### DevOps / Build Engineer
1. **Start:** TEAM_415_IMPLEMENTATION_COMPLETE.md
2. **Then:** TEAM_413_MARKETPLACE_FILTERING_CHECKLIST.md (Phase 3)
3. **Reference:** marketplace-node build scripts

**Focus:** SSG optimization, build performance, deployment

---

## 🚨 Common Pitfalls to Avoid

### 1. **Bypassing marketplace-node**
❌ **DON'T:** Call HuggingFace API directly from Next.js
✅ **DO:** Always use `@rbee/marketplace-node`

**Why:** Single source of truth, easier to add filtering, type safety

---

### 2. **Ignoring Compatibility Filtering**
❌ **DON'T:** Show all models from HuggingFace
✅ **DO:** Filter to only compatible models (SafeTensors, supported architectures)

**Why:** User confusion, SEO pollution, wasted static pages

---

### 3. **Hardcoding Model Lists**
❌ **DON'T:** Manually maintain list of supported models
✅ **DO:** Use compatibility checker with HuggingFace API

**Why:** Scales automatically, no manual updates needed

---

### 4. **Forgetting SEO**
❌ **DON'T:** Just show model info
✅ **DO:** Add download CTA, comparison tables, FAQ, use cases

**Why:** SEO traffic = free users, every model page is a landing page

---

### 5. **CPU Fallback**
❌ **DON'T:** Add CPU fallback for CUDA workers
✅ **DO:** Keep pure backend isolation (CUDA === CUDA only)

**Why:** 10x performance difference, architectural decision

---

## 📝 Quick Reference: Key Decisions

### Architecture
- **Data Flow:** Next.js → marketplace-node → HuggingFace API
- **Backend Isolation:** CUDA workers = CUDA only (no CPU fallback)
- **Model Format:** SafeTensors (Phase 1), GGUF (Phase 2)

### Filtering
- **Philosophy:** "If we don't support it, it doesn't exist"
- **Criteria:** Format (SafeTensors), Architecture (Llama, Mistral, etc.), Size (< 10GB)
- **Implementation:** Filter at API level + double-check at SSG

### SEO
- **Goal:** 5,000+ monthly downloads from SEO traffic
- **Strategy:** Every model page = landing page with CTA, comparison, FAQ
- **Metadata:** Title template, meta description, Open Graph, structured data

### Competitive Position
- **Advantages:** Multi-machine, heterogeneous hardware, advanced filtering
- **Gaps:** GGUF support (CRITICAL), model count (1,000 vs Ollama's 100+)
- **Opportunity:** Implement advanced filtering before LM Studio

---

## 🎯 Next Steps by Priority

### Priority 1: Fix Build Error (IMMEDIATE)
**Issue:** Model detail page crashes on nested objects in `config` field
**Solution:** Filter out or stringify nested objects before passing to React
**Estimated Time:** 30 minutes
**Document:** TEAM_415_IMPLEMENTATION_COMPLETE.md (Remaining Issue section)

---

### Priority 2: Implement Compatibility Filtering (HIGH)
**Goal:** Only show compatible models in marketplace
**Document:** TEAM_413_MARKETPLACE_FILTERING_CHECKLIST.md
**Estimated Time:** 10-15 hours (5 phases)
**Impact:** Better UX, faster builds, perfect SEO

---

### Priority 3: Implement SEO Strategy (HIGH)
**Goal:** Transform model pages into conversion funnels
**Document:** TEAM_414_MODEL_PAGE_SEO_STRATEGY.md
**Estimated Time:** 14-21 hours (5 phases)
**Impact:** 5,000+ monthly downloads from SEO

---

### Priority 4: Add GGUF Support (CRITICAL)
**Goal:** Support GGUF format for competitive parity
**Document:** ARCHITECTURE_EXPANSION_CHECKLIST.md
**Estimated Time:** 20-30 hours
**Impact:** Competitive with Ollama, 10x more models available

---

## 📚 Additional Resources

### External Documentation
- **HuggingFace API:** https://huggingface.co/docs/hub/api
- **Next.js SSG:** https://nextjs.org/docs/pages/building-your-application/rendering/static-site-generation
- **candle-transformers:** https://github.com/huggingface/candle/tree/main/candle-transformers

### Internal Specs
- `/bin/30_llm_worker_rbee/docs/` - Worker architecture docs
- `/bin/97_contracts/` - Shared contracts
- `/frontend/packages/rbee-ui/` - UI component library

---

## 🤝 Getting Help

### Questions About...

**Marketplace Architecture:**
- Read: TEAM_415_FIX_MARKETPLACE_PIPELINE.md
- Check: marketplace-node source code
- Ask: Frontend team lead

**Model Compatibility:**
- Read: TEAM_413_MARKETPLACE_FILTERING_CHECKLIST.md
- Check: ARCHITECTURE_EXPANSION_CHECKLIST.md
- Ask: Backend team lead

**SEO Strategy:**
- Read: TEAM_414_MODEL_PAGE_SEO_STRATEGY.md
- Check: Model page templates
- Ask: Product/Marketing team

**Competitive Positioning:**
- Read: TEAM_406_COMPETITIVE_RESEARCH.md
- Check: TEAM_406_FINAL_SUMMARY.md
- Ask: Product manager

---

## ✅ Checklist for New Team Members

Before starting work on marketplace:

- [ ] Read TEAM_415_FIX_MARKETPLACE_PIPELINE.md (understand architecture)
- [ ] Read TEAM_413_MARKETPLACE_FILTERING_CHECKLIST.md (understand filtering)
- [ ] Read TEAM_414_MODEL_PAGE_SEO_STRATEGY.md (understand SEO goals)
- [ ] Review marketplace-node source code
- [ ] Review marketplace app structure
- [ ] Understand "If we don't support it, it doesn't exist" philosophy
- [ ] Know the difference: SafeTensors (Phase 1) vs GGUF (Phase 2)
- [ ] Understand: CUDA === CUDA only (no CPU fallback)
- [ ] Review competitive research (rbee vs Ollama vs LM Studio)
- [ ] Know the priority: GGUF support is CRITICAL

---

**TEAM-416 - Recommended Reading List**  
**Total Documents:** 7 core + 3 reference  
**Estimated Reading Time:** 2-3 hours for core documents  
**Next:** Start with TEAM_415_FIX_MARKETPLACE_PIPELINE.md
