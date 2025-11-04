# Implementation Plans

**Last Updated:** 2025-11-04  
**Status:** Complete planning phase

---

## 📚 Document Index

**Read in this order:**

### 1. Start Here

- **README.md** ⭐ - This file, document index
- **QUICK_START.md** - For teams starting work
- **TEAM_CHECKLISTS.md** - Actionable task lists per team
- **FINAL_MASTER_PLAN.md** - Master roadmap
- **WOW_FACTOR_LAUNCH_MVP.md** - The killer demo
- **EXECUTIVE_SUMMARY.md** - High-level overview

### 2. Architecture & Design

- **BROWSER_TAB_SYSTEM.md** - Browser-like tabs (Zustand + dnd-kit)
- **MARKETPLACE_SYSTEM.md** - 3 marketplaces (HuggingFace, CivitAI, Workers)
- **WORKER_SPAWNING_3_STEPS.md** - 3-step spawning wizard UX
- **CATALOG_ARCHITECTURE_RESEARCH.md** - Backend crate analysis

### 3. Implementation

- **IMPLEMENTATION_PLAN_UPDATED.md** - Complete roadmap (37-55 days)

### 4. Business

- **LICENSE_STRATEGY.md** - Multi-license architecture for premium viability

---

## 🎯 Quick Summary

**Goal:** Launch MVP with dual-GPU demo showing LLM + SD running simultaneously

**Timeline:** 37-55 days (5-8 weeks)

**Key Features:**
- Browser-like tab system (multiple routes simultaneously)
- 3 marketplaces for browsing models/workers
- 3-step worker spawning wizard
- Split-screen worker UI (LLM chat + SD generation)

**Phases:**
1. Backend architecture (3-5 days)
2. Worker spawning backend (5-7 days)
3. Frontend infrastructure (7-10 days)
4. Marketplace system (6 days)
5. Worker spawning UX (3 days)
6. Dynamic worker UI (10-15 days)
7. Testing & polish (3-5 days)

---

## 🚀 The WOW Factor

**Demo:**
```
┌─────────────────────────────────────────────────────────┐
│  🐝 Bee Keeper                                          │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────┬──────────────────────┐        │
│  │ 💬 Chat (GPU 0)      │ 🎨 Images (GPU 1)    │        │
│  │ [LLM streaming]      │ [SD generating]      │        │
│  └──────────────────────┴──────────────────────┘        │
│  GPU 0: 87% ████████░░  GPU 1: 92% █████████░          │
└─────────────────────────────────────────────────────────┘
```

**BOTH running simultaneously on dual GPUs!**

---

## 🗑️ Deprecated Documents

**Rule Zero Applied:**

The following documents have been DELETED (outdated/superseded):
- ❌ COMPLETE_IMPLEMENTATION_PLAN.md - Superseded by IMPLEMENTATION_PLAN_UPDATED.md
- ❌ IMMEDIATE_NEXT_STEPS.md - Outdated, info merged into FINAL_MASTER_PLAN.md
- ❌ WORKER_SPAWNING_AND_UI_PLAN.md - Superseded by WORKER_SPAWNING_3_STEPS.md

**Single source of truth:** IMPLEMENTATION_PLAN_UPDATED.md

---

## 📊 Dependencies

**NPM packages:**
- zustand - State management for tabs
- @dnd-kit/core - Drag & drop for tab reordering
- @dnd-kit/sortable - Sortable tabs
- @dnd-kit/utilities - Utilities for dnd-kit

**Backend services:**
- bin/80-hono-worker-catalog (port 8502) - Worker metadata

---

## ⚠️ Important Notes

**Backend changes required:**
- Extend `ModelEntry` with `model_type` field (LLM vs SD)
- Extend `WorkerType` with SD variants (CpuSd, CudaSd, MetalSd)
- Implement `CivitAIVendor` for SD model downloads

**Frontend changes required:**
- Replace React Router `Routes` with Zustand tab system
- Update `App.tsx` to use `<TabBar />` and `<TabContent />`
- Update `KeeperSidebar` to use `addTab()` instead of `<Link>`

**Critical path:**
- Tab system blocks everything (need it to show multiple things)
- Marketplaces block spawning UX (need model browsing)
- Spawning UX blocks worker UI (need workers to spawn first)

---

## 🎯 Success Criteria

**MVP is ready when:**
- ✅ Can open multiple tabs
- ✅ Can browse models (HuggingFace + CivitAI)
- ✅ Can spawn LLM worker (3 steps)
- ✅ Can spawn SD worker (3 steps)
- ✅ Workers appear in tabs
- ✅ Split-screen demo works
- ✅ LLM chat works
- ✅ SD generation works
- ✅ Dual-GPU demo is impressive

**Then: LAUNCH!** 🚀
