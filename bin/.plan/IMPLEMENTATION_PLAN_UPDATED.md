# Complete Implementation Plan

**Date:** 2025-11-04  
**Status:** 📋 CURRENT - Single Source of Truth  
**Timeline:** 37-55 days (5-8 weeks)

---

## 📚 Related Documents

**See also:**
- BROWSER_TAB_SYSTEM.md - Tab architecture
- MARKETPLACE_SYSTEM.md - Marketplace templates
- WORKER_SPAWNING_3_STEPS.md - Spawning UX
- WOW_FACTOR_LAUNCH_MVP.md - Demo plan
- CATALOG_ARCHITECTURE_RESEARCH.md - Backend research
- EXECUTIVE_SUMMARY.md - High-level overview
- FINAL_MASTER_PLAN.md - Complete roadmap
- LICENSE_STRATEGY.md - Business licensing

---

## 📋 Updated Implementation Phases

### Phase 0: Critical Architecture (3-5 days) ✅ UNCHANGED

**Same as before:**
1. Extend ModelEntry
2. Extend WorkerType
3. Implement CivitAIVendor

### Phase 1: Worker Spawning Backend (5-7 days) ✅ UNCHANGED

**Same as before:**
1. Worker spawn validation
2. SD worker PKGBUILDs
3. End-to-end testing

### Phase 2: Frontend Infrastructure (7-10 days) 🆕 UPDATED

**2.1 Browser Tab System (5 days)**

Implement Zustand-based tab management:
- Install dependencies (zustand, @dnd-kit/core, @dnd-kit/sortable)
- Create Zustand tab store
- Create TabBar component with drag & drop
- Create TabContent renderer
- Update App.tsx to use tabs instead of Routes
- Add keyboard shortcuts
- Add tab persistence
- Test all layouts (single, split, grid)

**Verification:**
```tsx
// Can open multiple tabs
addTab({ type: 'home', title: 'Services', ... })
addTab({ type: 'queen', title: 'Queen', ... })

// Can reorder by dragging
// Can close tabs
// Tabs persist across reload
```

**2.2 Update Sidebar (2 days)**

Replace React Router Links with tab management:
- Update KeeperSidebar to use `useTabStore`
- Add Marketplaces section
- Update all navigation to open tabs
- Test navigation flow

**New Sidebar Structure:**
```
Main
  - Services
  - Queen

Marketplaces
  - HuggingFace 🤗
  - CivitAI 🎨
  - Workers 👷

Hives
  - localhost
  - remote-hive-1
  - remote-hive-2

System
  - Settings
  - Help
```

### Phase 3: Marketplace System (6 days) 🆕 NEW

**3.1 Marketplace Template (2 days)**

Create shared marketplace components:
- MarketplaceTemplate component
- MarketplaceCard component
- Filter components
- Search functionality
- Grid/List view toggle
- Test with mock data

**3.2 HuggingFace Marketplace (1 day)**

- Integrate with HuggingFace API
- Implement model search/filter
- Test download flow
- Show download progress

**3.3 CivitAI Marketplace (2 days)**

- Integrate with CivitAI API
- Handle preview images
- Implement NSFW toggle
- Test download flow

**3.4 Worker Catalog Marketplace (1 day)**

- Integrate with bin/80-hono-worker-catalog
- Show worker installation status
- Test installation flow
- Show build logs

**Verification:**
```bash
# Open marketplace tabs
Click: Sidebar → "HuggingFace" → Tab opens
Click: Sidebar → "CivitAI" → Tab opens
Click: Sidebar → "Workers" → Tab opens

# Search and download
Search: "llama"
Filter: 7B, Q4_K_M
Click: Download → Progress bar appears
Result: Model added to catalog
```

### Phase 4: Worker Spawning UX (3 days) 🆕 NEW

**4.1 3-Step Wizard (2 days)**

Create spawning wizard components:
- Step1_WorkerType (select LLM or SD)
- Step2_SelectModel (with marketplace links)
- Step3_SelectDevice (with availability check)
- SpawnWorkerWizard (main coordinator)
- Modal/dialog wrapper
- Integration with HivePage

**4.2 Worker Tab Auto-Open (1 day)**

- On successful spawn, create worker tab
- Auto-switch to split/grid layout
- Show worker interface (LLM chat or SD generation)

**Verification:**
```
1. Open hive page
2. Click "Spawn Worker"
3. Select: LLM Worker
4. Select: Llama-3.2-1B
5. Select: GPU 0
6. Worker spawns → Tab appears!
7. Split screen if 2+ workers
```

### Phase 5: Dynamic Worker UI (10-15 days) ✅ UNCHANGED

**Same as before:**
- LLM chat interface (3-4 days)
- SD generation interface (3-4 days)
- Integration & polish (3-4 days)

**But now with:**
- Auto-opens in tab when worker spawns
- Works in split-screen layout
- Multiple workers in grid layout

### Phase 6: Testing & Polish (3-5 days) ✅ UNCHANGED

**Same as before:**
- End-to-end testing
- UI polish
- Documentation

---

## 🗓️ Updated Timeline

| Phase | Duration | Dependencies | Status |
|-------|----------|--------------|--------|
| 0. Architecture | 3-5 days | None | ⏸️ Ready |
| 1. Worker Spawning Backend | 5-7 days | Phase 0 | ⏸️ Ready |
| 2. Frontend Infrastructure | 7-10 days | Phase 1 | ⏸️ Ready |
| 3. Marketplace System | 6 days | Phase 2 | ⏸️ Ready |
| 4. Worker Spawning UX | 3 days | Phase 2 | ⏸️ Ready |
| 5. Dynamic Worker UI | 10-15 days | Phase 4 | ⏸️ Ready |
| 6. Testing & Polish | 3-5 days | Phase 5 | ⏸️ Ready |
| **Total** | **37-55 days** | | **5-8 weeks** |

**Previous estimate:** 18-27 days  
**New estimate:** 37-55 days  
**Difference:** +19-28 days (added tab system + marketplaces + spawning UX)

---

## 🎯 New Dependencies

### NPM Packages

**Add to package.json:**
```json
{
  "dependencies": {
    "zustand": "^4.5.0",
    "@dnd-kit/core": "^6.1.0",
    "@dnd-kit/sortable": "^8.0.0",
    "@dnd-kit/utilities": "^3.2.2"
  }
}
```

### Backend Services

**Worker Catalog Service:**
- Location: `bin/80-hono-worker-catalog`
- Port: 8502
- Purpose: Worker metadata and PKGBUILDs
- Status: ✅ Exists, needs integration

---

## 🎨 New UI Components

**Created:**
```
src/
├── store/
│   └── tabStore.ts                 # Zustand tab management
├── components/
│   ├── TabBar.tsx                  # Browser-like tabs
│   ├── TabContent.tsx              # Tab content renderer
│   ├── MarketplaceTemplate.tsx     # Shared marketplace UI
│   ├── MarketplaceCard.tsx         # Marketplace item card
│   └── SpawnWorker/
│       ├── SpawnWorkerWizard.tsx   # Main wizard
│       ├── Step1_WorkerType.tsx    # Select type
│       ├── Step2_SelectModel.tsx   # Select model
│       └── Step3_SelectDevice.tsx  # Select device
└── pages/
    └── MarketplacePage.tsx         # Marketplace page
```

---

## 🔄 Migration Path

### From Old App.tsx (Routes)

**Before:**
```tsx
<Routes>
  <Route path="/" element={<ServicesPage />} />
  <Route path="/queen" element={<QueenPage />} />
  <Route path="/hive/:hiveId" element={<HivePage />} />
</Routes>
```

**After:**
```tsx
<div className="app-content">
  <TabBar />
  <TabContent />
</div>
```

### From Old Sidebar (Links)

**Before:**
```tsx
<Link to="/queen">Queen</Link>
```

**After:**
```tsx
<button onClick={() => addTab({
  type: 'queen',
  title: 'Queen',
  icon: '👑',
  route: '/queen'
})}>
  Queen
</button>
```

---

## 🎯 Updated Success Criteria

### Phase 2: Frontend Infrastructure

✅ Can open multiple tabs  
✅ Can reorder tabs by dragging  
✅ Can close tabs  
✅ Split-screen works automatically with 2 tabs  
✅ Grid layout works with 3-4 tabs  
✅ Tabs persist across reloads  
✅ Keyboard shortcuts work  
✅ Sidebar opens tabs instead of navigating  

### Phase 3: Marketplaces

✅ HuggingFace marketplace shows models  
✅ CivitAI marketplace shows models  
✅ Worker catalog shows workers  
✅ Search/filter works in all marketplaces  
✅ Download progress tracking works  
✅ Downloaded items are marked  
✅ Can browse while downloading  

### Phase 4: Worker Spawning

✅ Wizard opens from hive page  
✅ Step 1: Select worker type (2 options)  
✅ Step 2: Select model (with marketplace links)  
✅ Step 3: Select device (with availability)  
✅ Worker spawns successfully  
✅ Tab auto-opens for worker  
✅ Split-screen activates automatically  

### Phase 5: Dynamic Worker UI

✅ LLM chat works in tab  
✅ SD generation works in tab  
✅ Multiple workers work simultaneously  
✅ GPU utilization shown  
✅ Can interact with both workers at same time  

---

## 📊 Critical Path

**Sequence:**
```
Architecture → Backend → Tab System → Marketplaces
                                   ↓
                         Worker Spawning UX
                                   ↓
                         Dynamic Worker UI
                                   ↓
                         Testing & Launch!
```

**Blockers:**
- Tab system blocks everything (can't show multiple things without it)
- Marketplaces block spawning UX (need model browsing)
- Spawning UX blocks worker UI (need workers to spawn first)

---

## 🚀 Launch Readiness

**MVP Requirements:**
1. ✅ Tab system works
2. ✅ Can browse models (HuggingFace + CivitAI)
3. ✅ Can browse workers
4. ✅ Can spawn LLM worker (3 steps)
5. ✅ Can spawn SD worker (3 steps)
6. ✅ Workers appear in tabs
7. ✅ Split-screen demo works
8. ✅ LLM chat works
9. ✅ SD generation works
10. ✅ Dual-GPU demo is impressive

**When all 10 are checked: LAUNCH!** 🚀

---

## 📝 Documentation Updates

**Created:**
- ✅ BROWSER_TAB_SYSTEM.md - Tab architecture
- ✅ MARKETPLACE_SYSTEM.md - Marketplace design
- ✅ WORKER_SPAWNING_3_STEPS.md - Spawning UX
- ✅ WOW_FACTOR_LAUNCH_MVP.md - Demo plan
- ✅ IMPLEMENTATION_PLAN_UPDATED.md - This file

**Still valid:**
- ✅ CATALOG_ARCHITECTURE_RESEARCH.md - Backend research
- ✅ EXECUTIVE_SUMMARY.md - High-level overview (needs minor update)

---

**Everything is planned. Ready to execute!** 🚀
