# Checklist 05: Keeper UI (Tab System + Worker Spawning)

**Date:** 2025-11-04  
**Status:** 📋 READY TO START  
**Timeline:** 1 week  
**Dependencies:** CHECKLIST_04 (Tauri Protocol must be complete)

---

## 🎯 Goal

Add browser-like tab system and 3-step worker spawning wizard to Keeper. Enable users to manage multiple workers in separate tabs with drag-and-drop reordering.

---

## 📚 Reference Documents

**PRIMARY:**
- [BROWSER_TAB_SYSTEM.md](./BROWSER_TAB_SYSTEM.md) - Complete tab architecture (786 lines)
- [WORKER_SPAWNING_3_STEPS.md](./WORKER_SPAWNING_3_STEPS.md) - UX design (673 lines)

**SUPPORTING:**
- [COMPLETE_ONBOARDING_FLOW.md](./COMPLETE_ONBOARDING_FLOW.md) - Auto-run integration
- [MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md](./MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md) - Component patterns

---

## Phase 1: Tab System Foundation (Days 1-2)

### Day 1: State Management

- [ ] **Install dependencies** in `frontend/apps/keeper/`
  ```bash
  cd frontend/apps/keeper
  pnpm add zustand @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities
  ```
  **Verification:** Check `package.json` has all 4 dependencies

- [ ] **Create Zustand store** at `src/stores/tabStore.ts`
  ```typescript
  export interface Tab {
    id: string;
    title: string;
    type: 'dashboard' | 'worker' | 'spawn-wizard';
    icon?: string;
    data?: any;
  }
  
  export interface TabStore {
    tabs: Tab[];
    activeTabId: string | null;
    addTab: (tab: Omit<Tab, 'id'>) => void;
    closeTab: (id: string) => void;
    setActiveTab: (id: string) => void;
    updateTab: (id: string, updates: Partial<Tab>) => void;
    reorderTabs: (newOrder: string[]) => void;
  }
  ```
  **Implementation:** Follow BROWSER_TAB_SYSTEM.md lines 145-267
  **Verification:** `pnpm build` succeeds

- [ ] **Add Tab type definitions** at `src/types/tabs.ts`
  - Dashboard tab type
  - Worker tab type (workerId, status)
  - Spawn wizard tab type (currentStep, formData)
  **Verification:** TypeScript compiles without errors

- [ ] **Test store with unit tests** at `src/stores/__tests__/tabStore.test.ts`
  - Test addTab creates unique IDs
  - Test closeTab removes tab
  - Test reorderTabs updates order
  - Test activeTabId switches correctly
  **Verification:** `pnpm test` passes

### Day 2: Tab Bar Component

- [ ] **Create TabBar component** at `src/components/tabs/TabBar.tsx`
  ```tsx
  export function TabBar() {
    const { tabs, activeTabId, setActiveTab, closeTab, reorderTabs } = useTabStore();
    
    return (
      <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
        <SortableContext items={tabIds}>
          {tabs.map(tab => (
            <TabItem key={tab.id} tab={tab} />
          ))}
        </SortableContext>
      </DndContext>
    );
  }
  ```
  **Implementation:** Follow BROWSER_TAB_SYSTEM.md lines 270-428
  **Verification:** Renders in Storybook

- [ ] **Create TabItem component** at `src/components/tabs/TabItem.tsx`
  - Show tab icon + title
  - Active state styling
  - Close button (X)
  - Drag handle
  **Implementation:** Follow BROWSER_TAB_SYSTEM.md lines 430-531
  **Verification:** Drag-and-drop works

- [ ] **Add TabContent component** at `src/components/tabs/TabContent.tsx`
  ```tsx
  export function TabContent() {
    const { tabs, activeTabId } = useTabStore();
    const activeTab = tabs.find(t => t.id === activeTabId);
    
    if (!activeTab) return <div>No active tab</div>;
    
    switch (activeTab.type) {
      case 'dashboard': return <DashboardView />;
      case 'worker': return <WorkerView workerId={activeTab.data.workerId} />;
      case 'spawn-wizard': return <SpawnWorkerWizard />;
    }
  }
  ```
  **Verification:** Switching tabs shows correct content

- [ ] **Integrate into main layout** at `src/App.tsx`
  - Add TabBar to top of window
  - Add TabContent below
  - Set up default dashboard tab
  **Verification:** App launches with one tab

---

## Phase 2: Worker Spawning Wizard (Days 3-4)

### Day 3: Wizard Structure

- [ ] **Create wizard store** at `src/stores/spawnWizardStore.ts`
  ```typescript
  export interface SpawnWizardState {
    currentStep: 1 | 2 | 3;
    workerType: 'cpu-llm' | 'cuda-llm' | 'metal-llm' | null;
    selectedModel: ModelEntry | null;
    selectedDevice: DeviceInfo | null;
    goToStep: (step: 1 | 2 | 3) => void;
    reset: () => void;
  }
  ```
  **Implementation:** Follow WORKER_SPAWNING_3_STEPS.md lines 89-157
  **Verification:** Store compiles

- [ ] **Create Step1_WorkerType component** at `src/components/spawn-wizard/Step1_WorkerType.tsx`
  - Show 3 cards: CPU LLM, CUDA LLM, Metal LLM
  - Detect available worker binaries from Worker Catalog
  - Disable unavailable types
  - "Next" button (validates selection)
  **Implementation:** Follow WORKER_SPAWNING_3_STEPS.md lines 160-289
  **Verification:** Card selection updates store

- [ ] **Create Step2_SelectModel component** at `src/components/spawn-wizard/Step2_SelectModel.tsx`
  - Fetch models from Model Catalog API
  - Show ModelCard grid (from @rbee/marketplace-components)
  - Filter by compatibility (e.g., CUDA models for CUDA worker)
  - Search & sort
  - "Back" and "Next" buttons
  **Implementation:** Follow WORKER_SPAWNING_3_STEPS.md lines 292-421
  **Verification:** Model selection updates store

- [ ] **Create Step3_SelectDevice component** at `src/components/spawn-wizard/Step3_SelectDevice.tsx`
  - Fetch devices from Queen API (`/api/devices`)
  - Show device cards (GPU name, VRAM, status)
  - Single-select for device
  - "Back" and "Spawn" buttons
  **Implementation:** Follow WORKER_SPAWNING_3_STEPS.md lines 424-553
  **Verification:** Device selection updates store

### Day 4: Wizard Integration

- [ ] **Create SpawnWorkerWizard container** at `src/components/spawn-wizard/SpawnWorkerWizard.tsx`
  ```tsx
  export function SpawnWorkerWizard() {
    const { currentStep } = useSpawnWizardStore();
    
    return (
      <div className="h-full flex flex-col">
        <WizardProgress step={currentStep} />
        
        {currentStep === 1 && <Step1_WorkerType />}
        {currentStep === 2 && <Step2_SelectModel />}
        {currentStep === 3 && <Step3_SelectDevice />}
      </div>
    );
  }
  ```
  **Implementation:** Follow WORKER_SPAWNING_3_STEPS.md lines 556-638
  **Verification:** Wizard flow works end-to-end

- [ ] **Add WizardProgress component** at `src/components/spawn-wizard/WizardProgress.tsx`
  - Show step 1/2/3 indicators
  - Highlight current step
  - Show step titles: "Worker Type" → "Select Model" → "Select Device"
  **Verification:** Progress bar updates on step change

- [ ] **Handle spawn submission**
  - Call Queen API: `POST /api/jobs` with `Operation::WorkerSpawn`
  - Show loading state during spawn
  - On success: Create new worker tab, close wizard tab
  - On error: Show error message, stay on wizard
  **Implementation:** Follow WORKER_SPAWNING_3_STEPS.md lines 640-673
  **Verification:** Spawning creates new worker tab

- [ ] **Add "Spawn Worker" button to dashboard**
  - Button in dashboard view
  - Opens new spawn wizard tab
  - Only one wizard tab allowed at a time
  **Verification:** Clicking button opens wizard

---

## Phase 3: Worker Tab View (Day 5)

### Worker Tab Implementation

- [ ] **Create WorkerView component** at `src/components/worker/WorkerView.tsx`
  - Fetch worker status from Queen API
  - Show worker info card (model, device, status)
  - Show real-time logs (EventSource to Queen API)
  - Start/stop/restart buttons
  - Delete worker button
  **Implementation:** Use existing Keeper components, adapt for tabs
  **Verification:** Worker info displays correctly

- [ ] **Add worker status polling**
  - Poll Queen API every 5 seconds
  - Update tab title with status emoji (🟢 running, 🟡 starting, 🔴 stopped)
  - Update tab icon color
  **Verification:** Tab title updates in real-time

- [ ] **Implement worker actions**
  - Start: `POST /api/jobs` with `Operation::WorkerStart`
  - Stop: `POST /api/jobs` with `Operation::WorkerStop`
  - Delete: `POST /api/jobs` with `Operation::WorkerDelete`
  - On delete: Close worker tab
  **Verification:** All actions work correctly

- [ ] **Add error handling**
  - Handle worker not found (show error in tab)
  - Handle API errors (show retry button)
  - Handle worker crash (show restart option)
  **Verification:** Error states display correctly

---

## Phase 4: Dashboard Integration (Day 6)

### Dashboard Enhancements

- [ ] **Update dashboard to use tabs**
  - Show list of all workers
  - Each worker card has "Open in Tab" button
  - Clicking worker opens new tab (or focuses existing)
  - Show number of active tabs
  **Verification:** Dashboard shows all workers

- [ ] **Add tab shortcuts**
  - Keyboard shortcuts: Cmd+1 to Cmd+9 for tabs 1-9
  - Cmd+W to close current tab
  - Cmd+T to open new spawn wizard
  **Implementation:** Use Tauri's global shortcuts
  **Verification:** Shortcuts work

- [ ] **Add tab persistence**
  - Save tab state to localStorage
  - Restore tabs on app restart
  - Exclude wizard tabs from persistence
  **Verification:** Tabs restore after restart

- [ ] **Add max tabs limit**
  - Limit to 10 tabs max
  - Show warning when limit reached
  - Suggest closing inactive tabs
  **Verification:** Cannot open more than 10 tabs

---

## Phase 5: Auto-Run Flow Integration (Day 7)

### Protocol Handler Integration

- [ ] **Integrate wizard with protocol handler**
  - `rbee://spawn/worker/{workerType}/model/{modelId}/device/{deviceId}`
  - Open spawn wizard tab pre-filled
  - Skip to Step 3 (all values set)
  - Auto-click "Spawn" button
  **Implementation:** Extend protocol handler from CHECKLIST_04
  **Verification:** Protocol link opens pre-filled wizard

- [ ] **Handle auto-run errors**
  - If model not downloaded: Show download progress in wizard
  - If worker binary missing: Show error, suggest installation
  - If device unavailable: Show alternative devices
  **Verification:** Errors handled gracefully

- [ ] **Add notification system**
  - Show toast when worker spawned
  - Show toast when worker starts/stops
  - Show toast on errors
  - Use Tauri's notification API
  **Verification:** Notifications appear

- [ ] **Complete end-to-end test**
  - Click "Run with rbee" on marketplace.rbee.dev
  - Keeper opens with pre-filled wizard
  - Click "Spawn" button
  - New worker tab opens
  - Worker starts automatically
  - Tab title updates to "🟢 Running"
  **Verification:** Full flow works without manual input

---

## Phase 6: Testing & Polish (Day 7 afternoon)

### Testing

- [ ] **Unit tests for stores**
  - tabStore.test.ts
  - spawnWizardStore.test.ts
  **Verification:** `pnpm test` passes

- [ ] **Component tests**
  - TabBar.test.tsx (drag-and-drop)
  - SpawnWorkerWizard.test.tsx (step flow)
  - WorkerView.test.tsx (status updates)
  **Verification:** All component tests pass

- [ ] **Integration test**
  - Create test: Open app → Spawn worker → Close tab
  - Test tab reordering
  - Test keyboard shortcuts
  **Verification:** Manual testing passes

### Polish

- [ ] **Add loading states**
  - Loading spinner while fetching models
  - Loading spinner while spawning worker
  - Skeleton UI for worker view
  **Verification:** No blank screens during loading

- [ ] **Add empty states**
  - "No models found" message
  - "No devices available" message
  - "No workers running" in dashboard
  **Verification:** Empty states look good

- [ ] **Responsive design**
  - Tab bar works on small windows
  - Wizard fits in 1024x768 minimum
  - Worker view scrollable
  **Verification:** Test at different window sizes

- [ ] **Accessibility**
  - Tab keyboard navigation (Arrow keys)
  - Focus management in wizard
  - ARIA labels on buttons
  **Verification:** Screen reader compatible

---

## Success Criteria

### Week Complete When:

- [ ] Tab system works with drag-and-drop
- [ ] Spawn wizard completes all 3 steps
- [ ] Worker tabs show real-time status
- [ ] Auto-run flow works from marketplace.rbee.dev
- [ ] All tests passing
- [ ] No console errors
- [ ] Documentation complete

### Manual Testing Checklist:

**Tab System:**
- [ ] Can open multiple tabs
- [ ] Can close tabs (X button)
- [ ] Can reorder tabs (drag-and-drop)
- [ ] Can switch tabs (click)
- [ ] Active tab has visual indicator
- [ ] Tab titles update correctly

**Spawn Wizard:**
- [ ] Step 1: Can select worker type
- [ ] Step 2: Can select model
- [ ] Step 3: Can select device
- [ ] "Back" button works
- [ ] "Next" button validates
- [ ] "Spawn" button creates worker
- [ ] Wizard tab closes after spawn

**Worker Tab:**
- [ ] Worker info displays
- [ ] Status updates in real-time
- [ ] Logs stream correctly
- [ ] Start/stop buttons work
- [ ] Delete closes tab

**Auto-Run:**
- [ ] Protocol link opens Keeper
- [ ] Wizard pre-fills correctly
- [ ] Auto-spawn works
- [ ] New tab opens automatically

---

## Notes

### Key Principles

1. **Tab State Isolation** - Each tab has independent state
2. **Drag-and-Drop** - Use @dnd-kit, not custom implementation
3. **Real-Time Updates** - Use EventSource for worker status
4. **Error Recovery** - Every error should have clear action
5. **Keyboard First** - Support power users with shortcuts

### Common Pitfalls

- ❌ Forgetting to close wizard tab after spawn
- ❌ Not handling worker not found errors
- ❌ Creating multiple wizard tabs (only 1 allowed)
- ❌ Not updating tab title when worker status changes
- ✅ Test drag-and-drop on all browsers
- ✅ Test with 10+ tabs open
- ✅ Test auto-run flow end-to-end

### Performance Considerations

- Use `React.memo` for TabItem to prevent unnecessary re-renders
- Debounce tab reordering updates
- Lazy load worker logs (virtualized list)
- Use SWR or React Query for API caching

---

## Dependencies

**Must be complete before starting:**
- ✅ CHECKLIST_01 (Shared Components)
- ✅ CHECKLIST_02 (Marketplace SDK)
- ✅ CHECKLIST_04 (Tauri Protocol)

**Blocks:**
- CHECKLIST_06 (Launch Demo) - needs worker spawning working

---

## Files Created

**State Management:**
- `frontend/apps/keeper/src/stores/tabStore.ts`
- `frontend/apps/keeper/src/stores/spawnWizardStore.ts`
- `frontend/apps/keeper/src/types/tabs.ts`

**Tab System:**
- `frontend/apps/keeper/src/components/tabs/TabBar.tsx`
- `frontend/apps/keeper/src/components/tabs/TabItem.tsx`
- `frontend/apps/keeper/src/components/tabs/TabContent.tsx`

**Spawn Wizard:**
- `frontend/apps/keeper/src/components/spawn-wizard/SpawnWorkerWizard.tsx`
- `frontend/apps/keeper/src/components/spawn-wizard/Step1_WorkerType.tsx`
- `frontend/apps/keeper/src/components/spawn-wizard/Step2_SelectModel.tsx`
- `frontend/apps/keeper/src/components/spawn-wizard/Step3_SelectDevice.tsx`
- `frontend/apps/keeper/src/components/spawn-wizard/WizardProgress.tsx`

**Worker View:**
- `frontend/apps/keeper/src/components/worker/WorkerView.tsx`

**Tests:**
- `frontend/apps/keeper/src/stores/__tests__/tabStore.test.ts`
- `frontend/apps/keeper/src/stores/__tests__/spawnWizardStore.test.ts`
- `frontend/apps/keeper/src/components/tabs/__tests__/TabBar.test.tsx`
- `frontend/apps/keeper/src/components/spawn-wizard/__tests__/SpawnWorkerWizard.test.tsx`

---

## Next Steps

After completing this checklist:
1. Read CHECKLIST_06_LAUNCH_DEMO.md
2. Prepare "WOW FACTOR" demo
3. Record demo video
4. LAUNCH! 🚀

---

**Remember:** One checkbox at a time. Test everything. Document as you go.

**Let's build an amazing UI!** 🐝
