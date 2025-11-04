# Team Checklists - Complete Work Breakdown

**Date:** 2025-11-04  
**Status:** ✅ ACTIONABLE - Ready to Execute  
**Timeline:** 37-55 days (5-8 weeks)

---

## 🎯 Team Organization

**5 Teams, Working in Parallel Where Possible:**

1. **TEAM-BACKEND** - Rust backend (catalog, workers, spawning)
2. **TEAM-FRONTEND-INFRA** - Tab system, routing, state management
3. **TEAM-FRONTEND-UI** - Marketplaces, forms, wizards
4. **TEAM-WORKER-UI** - LLM chat + SD generation interfaces
5. **TEAM-INTEGRATION** - Testing, polish, launch prep

---

## 📅 Phase-Based Execution

### Phase 0: Backend Foundation (Days 1-5)

**TEAM-BACKEND** 🔧

**Catalog Architecture Updates:**
- [ ] Add `ModelType` enum to `model-catalog/src/types.rs`
  - [ ] Define `Llm` variant with architecture, parameter_count, context_length
  - [ ] Define `StableDiffusion` variant with SdVersion enum
  - [ ] Add `SdVersion` enum (V1_5, V2_1, XL, Turbo, etc.)
- [ ] Update `ModelEntry` struct
  - [ ] Add `model_type: ModelType` field
  - [ ] Add `vendor: String` field (huggingface, civitai, local)
  - [ ] Add `quantization: Option<String>` field
  - [ ] Add `metadata: HashMap<String, String>` field
- [ ] Add model compatibility methods
  - [ ] Implement `compatible_worker_types() -> Vec<WorkerType>`
  - [ ] Implement `is_compatible_with(worker_type: &WorkerType) -> bool`
- [ ] Update `ModelCatalog` serialization to handle new fields
- [ ] Write migration for existing models in catalog

**Worker Type Extension:**
- [ ] Add SD worker types to `worker-catalog/src/types.rs`
  - [ ] Add `CpuSd` variant
  - [ ] Add `CudaSd` variant
  - [ ] Add `MetalSd` variant
- [ ] Update `binary_name()` method for SD workers
  - [ ] `CpuSd` → "sd-worker-rbee-cpu"
  - [ ] `CudaSd` → "sd-worker-rbee-cuda"
  - [ ] `MetalSd` → "sd-worker-rbee-metal"
- [ ] Update `crate_name()` method
  - [ ] SD workers → "sd-worker-rbee"
- [ ] Update `build_features()` method
- [ ] Add `model_type_category()` method
  - [ ] Returns `ModelTypeCategory::Llm` or `::StableDiffusion`

**CivitAI Vendor:**
- [ ] Create `model-provisioner/src/civitai.rs`
- [ ] Define `CivitAIVendor` struct with `api_token` and `client`
- [ ] Implement `new(api_token: String) -> Result<Self>`
- [ ] Implement `from_env() -> Result<Self>` (reads CIVITAI_API_TOKEN)
- [ ] Implement `VendorSource` trait
  - [ ] `download()` - Stream download with progress
  - [ ] `supports()` - Check for "civitai:" prefix
  - [ ] `name()` - Return "CivitAI"
- [ ] Add ID parsing: `civitai:123456` or `civitai:123456:filename.safetensors`
- [ ] Add progress tracking integration
- [ ] Add cancellation support via CancellationToken
- [ ] Export in `model-provisioner/src/lib.rs`

**Testing:**
- [ ] `cargo test -p rbee-hive-model-catalog`
- [ ] `cargo test -p rbee-hive-worker-catalog`
- [ ] `CIVITAI_API_TOKEN=xxx cargo test -p rbee-hive-model-provisioner`
- [ ] Verify all existing tests still pass

**Deliverables:** ModelType system, SD workers, CivitAI vendor

---

### Phase 1: Worker Spawning Backend (Days 6-12)

**TEAM-BACKEND** 🔧

**Worker Spawn Validation:**
- [ ] Update `rbee-hive/src/operations/worker.rs`
- [ ] Add model compatibility check
  - [ ] Get model from catalog
  - [ ] Call `model.is_compatible_with(worker_type)`
  - [ ] Return error if incompatible with clear message
- [ ] Add worker binary lookup
  - [ ] `worker_catalog.find_by_type_and_platform(worker_type, platform)`
  - [ ] Return error if binary not found
- [ ] Build spawn command for LLM workers
  - [ ] Add `--worker-id`, `--model`, `--device` args
- [ ] Build spawn command for SD workers
  - [ ] Add `--worker-id`, `--model`, `--device` args
  - [ ] Add `--sd-version` arg from model metadata
- [ ] Spawn process and capture PID
- [ ] Register worker in worker registry
- [ ] Wait for health check (30 second timeout)
- [ ] Update worker status to Ready

**SD Worker PKGBUILDs:**
- [ ] Create `PKGBUILD.sd-worker-cpu` in `31_sd_worker_rbee/`
  - [ ] Set pkgname, version, arch, license
  - [ ] Define build() function with `cargo build --bin sd-worker-rbee-cpu --features cpu`
  - [ ] Define package() function with install path
- [ ] Create `PKGBUILD.sd-worker-cuda`
  - [ ] Build with `--features cuda`
- [ ] Create `PKGBUILD.sd-worker-metal`
  - [ ] Build with `--features metal`
- [ ] Test PKGBUILD execution locally

**End-to-End Testing:**
- [ ] Test LLM model download (HuggingFace)
  - [ ] `curl -X POST .../model_download -d '{"model_id":"TheBloke/Llama-2-7B-Chat-GGUF"}'`
  - [ ] Verify model appears in catalog with correct `model_type`
- [ ] Test SD model download (CivitAI)
  - [ ] `curl -X POST .../model_download -d '{"model_id":"civitai:123456"}'`
  - [ ] Verify model appears in catalog with correct `model_type`
- [ ] Test LLM worker spawn
  - [ ] Verify compatibility check works
  - [ ] Verify worker starts successfully
  - [ ] Verify health check passes
- [ ] Test SD worker spawn
  - [ ] Verify compatibility check works
  - [ ] Verify worker starts successfully
  - [ ] Verify health check passes
- [ ] Test incompatible model rejection
  - [ ] Try to spawn LLM worker with SD model → should fail
  - [ ] Try to spawn SD worker with LLM model → should fail

**Deliverables:** Worker spawning works end-to-end for both LLM and SD

---

### Phase 2: Frontend Infrastructure (Days 13-22)

**TEAM-FRONTEND-INFRA** ⚛️

**Tab System Setup:**
- [ ] Install dependencies
  - [ ] `pnpm add zustand @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities`
- [ ] Create `src/store/tabStore.ts`
  - [ ] Define `Tab` interface (id, type, title, icon, route, metadata, state)
  - [ ] Define `TabStore` interface (tabs, activeTabId, layout, actions)
  - [ ] Implement `addTab()` - Creates new tab, auto-layout
  - [ ] Implement `removeTab()` - Removes tab, updates active
  - [ ] Implement `setActiveTab()` - Changes active tab
  - [ ] Implement `updateTab()` - Updates tab properties
  - [ ] Implement `reorderTabs()` - Drag & drop reordering
  - [ ] Implement `closeAllTabs()`, `closeOtherTabs()`, `duplicateTab()`
  - [ ] Implement `setLayout()` - Manual layout control
  - [ ] Implement `autoLayout()` - Auto-detect best layout
  - [ ] Add Zustand persist middleware for localStorage

**Tab Bar Component:**
- [ ] Create `src/components/TabBar.tsx`
- [ ] Create `SortableTab` component
  - [ ] Use `useSortable` hook from dnd-kit
  - [ ] Show tab icon, title, dirty indicator
  - [ ] Show pin icon if pinned
  - [ ] Show close button if closeable
  - [ ] Handle click to activate
  - [ ] Handle right-click for context menu
- [ ] Create `TabBar` component
  - [ ] Wrap tabs in `DndContext` and `SortableContext`
  - [ ] Render all tabs from store
  - [ ] Handle drag end event
  - [ ] Add "New Tab" button
  - [ ] Add keyboard navigation support

**Tab Content Renderer:**
- [ ] Create `src/components/TabContent.tsx`
- [ ] Implement `renderTabContent(tab: Tab)` function
  - [ ] Switch on `tab.type`
  - [ ] Return correct page component for each type
  - [ ] Pass required props (hiveId, workerId, etc.)
- [ ] Implement layout rendering
  - [ ] Single layout: Full width, one tab
  - [ ] Split layout: 50/50, two tabs side-by-side
  - [ ] Grid layout: 2x2, up to 4 tabs
- [ ] Handle empty state (no tabs)

**App.tsx Update:**
- [ ] Remove React Router `<Routes>` component
- [ ] Add `<TabBar />` component
- [ ] Add `<TabContent />` component
- [ ] Initialize with home tab if no tabs exist
- [ ] Keep `<BrowserRouter>` wrapper (for future use)
- [ ] Keep narration and theme listeners

**Keyboard Shortcuts:**
- [ ] Implement `useKeyboardShortcuts` hook
  - [ ] Ctrl+T: New tab
  - [ ] Ctrl+W: Close tab
  - [ ] Ctrl+Tab: Next tab
  - [ ] Ctrl+Shift+Tab: Previous tab
  - [ ] Ctrl+1-9: Switch to tab N

**Tab Persistence:**
- [ ] Test localStorage persistence
  - [ ] Open tabs
  - [ ] Reload page
  - [ ] Verify tabs restored
  - [ ] Verify active tab restored
  - [ ] Verify tab order restored

**Deliverables:** Working tab system, can open/close/reorder tabs

---

**TEAM-FRONTEND-INFRA** ⚛️ (Continued)

**Sidebar Update:**
- [ ] Update `src/components/KeeperSidebar.tsx`
- [ ] Import `useTabStore` hook
- [ ] Remove all `<Link>` components
- [ ] Replace with `<button onClick={() => addTab({...})}>` for each nav item
- [ ] Update main navigation
  - [ ] Services → Opens home tab
  - [ ] Queen → Opens queen tab
- [ ] Add Marketplaces section
  - [ ] HuggingFace → Opens marketplace tab (type: huggingface)
  - [ ] CivitAI → Opens marketplace tab (type: civitai)
  - [ ] Workers → Opens marketplace tab (type: workers)
- [ ] Update hives section
  - [ ] Each hive → Opens hive tab
- [ ] Update system section
  - [ ] Settings, Help → Open respective tabs
- [ ] Remove `useLocation()` hook (no longer needed)
- [ ] Test all navigation items open tabs correctly

**Deliverables:** Sidebar opens tabs instead of navigating

---

### Phase 3: Marketplace System (Days 23-28)

**TEAM-FRONTEND-UI** 🎨

**Shared Marketplace Template:**
- [ ] Create `src/types/marketplace.ts`
  - [ ] Define `MarketplaceType` type
  - [ ] Define `MarketplaceItem` interface
  - [ ] Define `MarketplaceFilters` interface
  - [ ] Define `MarketplaceConfig` interface
- [ ] Create `src/components/MarketplaceTemplate.tsx`
  - [ ] Header with title, icon, search bar, view toggle
  - [ ] Sidebar with filters (collapsible)
  - [ ] Content area with grid/list view
  - [ ] Loading, error, empty states
  - [ ] Integrate with filter state
- [ ] Create `src/components/MarketplaceCard.tsx`
  - [ ] Image/icon display
  - [ ] Title, author, description
  - [ ] Tags (first 3)
  - [ ] Stats (downloads, likes, rating)
  - [ ] Status badge (available, downloading, downloaded)
  - [ ] Download progress bar
  - [ ] Action buttons (download, details)

**HuggingFace Marketplace:**
- [ ] Create `src/pages/MarketplacePage.tsx`
- [ ] Configure for HuggingFace
  - [ ] Title: "HuggingFace Models"
  - [ ] Icon: 🤗
  - [ ] API endpoint: `/api/marketplace/huggingface`
  - [ ] Filters: Architecture, Parameter Count, Quantization
- [ ] Implement `useHuggingFaceModels()` hook
  - [ ] Fetch from backend API
  - [ ] Transform to `MarketplaceItem[]`
  - [ ] Handle loading and errors
- [ ] Implement download handler
  - [ ] Call backend `model_download` operation
  - [ ] Show progress in card
  - [ ] Update status when complete
- [ ] Test search functionality
- [ ] Test filter functionality
- [ ] Test grid/list view toggle

**CivitAI Marketplace:**
- [ ] Configure for CivitAI
  - [ ] Title: "CivitAI Models"
  - [ ] Icon: 🎨
  - [ ] API endpoint: `/api/marketplace/civitai`
  - [ ] Filters: SD Version, Style, NSFW toggle
- [ ] Implement `useCivitAIModels()` hook
  - [ ] Fetch from backend API (proxied to CivitAI)
  - [ ] Transform to `MarketplaceItem[]`
  - [ ] Handle preview images
  - [ ] Handle loading and errors
- [ ] Implement download handler
  - [ ] Call backend `model_download` operation
  - [ ] Show progress in card
  - [ ] Update status when complete
- [ ] Test NSFW filter toggle
- [ ] Test preview image loading

**Worker Catalog Marketplace:**
- [ ] Configure for Workers
  - [ ] Title: "Worker Catalog"
  - [ ] Icon: 👷
  - [ ] API endpoint: `http://localhost:8502/workers`
  - [ ] Filters: Worker Type (LLM/SD), Device (CPU/CUDA/Metal)
- [ ] Implement `useWorkerCatalog()` hook
  - [ ] Fetch from worker-catalog service
  - [ ] Transform to `MarketplaceItem[]`
  - [ ] Handle loading and errors
- [ ] Implement install handler
  - [ ] Call backend `worker_install` operation
  - [ ] Show build progress
  - [ ] Show build logs
  - [ ] Update status when complete
- [ ] Test worker installation flow

**Deliverables:** 3 working marketplaces with download/install

---

### Phase 4: Worker Spawning UX (Days 23-31)

**TEAM-FRONTEND-UI** 🎨

**3-Step Wizard Components:**
- [ ] Create `src/components/SpawnWorker/` directory
- [ ] Create `Step1_WorkerType.tsx`
  - [ ] Two large cards: LLM Worker, SD Worker
  - [ ] Icons: 💬, 🎨
  - [ ] Descriptions of each type
  - [ ] Select button on each
  - [ ] Cancel button at bottom
- [ ] Create `Step2_SelectModel.tsx`
  - [ ] Show worker type icon and name
  - [ ] Search bar for filtering models
  - [ ] List of downloaded models (with checkmark)
  - [ ] List of available models (with download button)
  - [ ] "Browse Marketplace" links
  - [ ] Back and Cancel buttons
- [ ] Create `Step3_SelectDevice.tsx`
  - [ ] Show worker type and model name
  - [ ] List of available devices (CPU, GPU 0, GPU 1, etc.)
  - [ ] Show device info (name, VRAM, availability)
  - [ ] Warning if low VRAM
  - [ ] Tip about GPU performance
  - [ ] Back and Cancel buttons

**Wizard Coordinator:**
- [ ] Create `SpawnWorkerWizard.tsx`
- [ ] Manage wizard state (step, workerType, modelId, modelName)
- [ ] Handle step navigation (next, back, cancel)
- [ ] Handle worker type selection → Go to step 2
- [ ] Handle model selection → Go to step 3
- [ ] Handle device selection → Spawn worker
- [ ] Handle "Browse Marketplace" → Close wizard, open marketplace tab
- [ ] Implement `useSpawnWorker` hook
  - [ ] Call backend `worker_spawn` operation
  - [ ] Show loading overlay during spawn
  - [ ] Handle success: Close wizard, open worker tab
  - [ ] Handle errors: Show error message

**Integration with Hive Page:**
- [ ] Update `src/pages/HivePage.tsx`
- [ ] Add "Spawn Worker" button to header
- [ ] Add state for wizard visibility
- [ ] Render wizard in modal when visible
- [ ] Pass hiveId to wizard
- [ ] Handle modal backdrop click to close

**Worker Tab Auto-Open:**
- [ ] On successful spawn, create worker tab
  - [ ] Type: 'worker'
  - [ ] Title: Icon + Model name
  - [ ] Icon: 💬 for LLM, 🎨 for SD
  - [ ] Route: `/worker/{workerId}`
  - [ ] Metadata: workerId, workerType, modelId
- [ ] Auto-switch to split layout if 2+ workers
- [ ] Test tab appears immediately after spawn

**Deliverables:** Working 3-step spawning wizard, auto-opens tabs

---

### Phase 5: Dynamic Worker UI (Days 32-46)

**TEAM-WORKER-UI** 🖥️

**LLM Chat Interface:**
- [ ] Create `src/components/LlmChatInterface.tsx`
- [ ] Implement message state management
  - [ ] Array of messages (role, content)
  - [ ] Input state
  - [ ] Generating state
- [ ] Create `ChatMessage` component
  - [ ] User messages (right-aligned, blue)
  - [ ] Assistant messages (left-aligned, gray)
  - [ ] Markdown rendering
  - [ ] Copy button
- [ ] Create message input area
  - [ ] Textarea with auto-resize
  - [ ] Send button
  - [ ] Stop button (when generating)
  - [ ] Token counter
- [ ] Implement `useLlmInference` hook
  - [ ] `generate(prompt, onChunk)` - Stream response
  - [ ] `stop()` - Cancel generation
  - [ ] Track token count
  - [ ] Track generation state
- [ ] Connect to worker endpoint
  - [ ] POST to `/v1/jobs` with Infer operation
  - [ ] Stream from `/v1/jobs/{jobId}/stream`
  - [ ] Parse SSE events
  - [ ] Update message content on each chunk
- [ ] Add auto-scroll to latest message
- [ ] Add loading indicators
- [ ] Test with real LLM worker

**SD Generation Interface:**
- [ ] Create `src/components/SdGenerationInterface.tsx`
- [ ] Implement generation state
  - [ ] Prompt state
  - [ ] Parameters state (steps, size, guidance)
  - [ ] Generating state
  - [ ] Progress state (0-100)
  - [ ] Generated image (base64)
- [ ] Create generation form
  - [ ] Prompt textarea
  - [ ] Steps slider (10-50)
  - [ ] Size selector (512x512, 768x768, 1024x1024)
  - [ ] Guidance scale slider
  - [ ] Generate button
  - [ ] Stop button (when generating)
- [ ] Create progress display
  - [ ] Progress bar with percentage
  - [ ] Current step / total steps
- [ ] Create image preview area
  - [ ] Show generated image
  - [ ] Download button
  - [ ] Regenerate button
- [ ] Implement `useSdGeneration` hook
  - [ ] `generate(prompt, params)` - Start generation
  - [ ] `stop()` - Cancel generation
  - [ ] Track progress
  - [ ] Track generation state
- [ ] Connect to worker endpoint
  - [ ] POST to `/v1/jobs` with ImageGeneration operation
  - [ ] Stream from `/v1/jobs/{jobId}/stream`
  - [ ] Parse progress events
  - [ ] Display final image
- [ ] Add image download functionality
- [ ] Test with real SD worker

**Worker Tab Rendering:**
- [ ] Create `src/pages/WorkerPage.tsx`
- [ ] Get worker info from `workerId`
- [ ] Determine worker type (LLM or SD)
- [ ] Render appropriate interface
  - [ ] LLM → `<LlmChatInterface />`
  - [ ] SD → `<SdGenerationInterface />`
- [ ] Show worker status (header)
  - [ ] Worker ID
  - [ ] Model name
  - [ ] GPU assignment
  - [ ] Status indicator
- [ ] Add worker controls (header)
  - [ ] Stop worker button
  - [ ] Restart worker button
  - [ ] Worker settings

**Split-Screen Testing:**
- [ ] Spawn LLM worker → Verify chat interface appears
- [ ] Spawn SD worker → Verify split-screen activates
- [ ] Type in LLM chat → Verify streaming works
- [ ] Start SD generation → Verify progress updates
- [ ] Test both working simultaneously
- [ ] Verify GPU meters update
- [ ] Test closing workers removes tabs
- [ ] Test 3-4 workers in grid layout

**Deliverables:** Working LLM chat + SD generation, split-screen demo works

---

### Phase 6: Integration & Launch Prep (Days 47-55)

**TEAM-INTEGRATION** 🚀

**End-to-End Testing:**
- [ ] Test complete user flow
  - [ ] Open Keeper
  - [ ] Browse HuggingFace marketplace
  - [ ] Download LLM model
  - [ ] Browse CivitAI marketplace
  - [ ] Download SD model
  - [ ] Browse Worker catalog
  - [ ] Install LLM worker binary
  - [ ] Install SD worker binary
  - [ ] Navigate to localhost hive
  - [ ] Spawn LLM worker (3 steps)
  - [ ] Spawn SD worker (3 steps)
  - [ ] Chat with LLM
  - [ ] Generate SD image
  - [ ] Both work simultaneously
  - [ ] Close workers
  - [ ] Tabs close correctly
- [ ] Test error scenarios
  - [ ] Model download fails → Shows error
  - [ ] Worker spawn fails → Shows error
  - [ ] Incompatible model → Shows clear message
  - [ ] No device available → Shows warning
  - [ ] Worker crashes → Shows error
- [ ] Test edge cases
  - [ ] Open 10+ tabs → Still performant
  - [ ] Reorder tabs → Works smoothly
  - [ ] Close all tabs → Shows empty state
  - [ ] Reload page → Tabs persist

**UI Polish:**
- [ ] Review all animations
  - [ ] Tab open/close
  - [ ] Split-screen transition
  - [ ] Progress bars
  - [ ] Loading spinners
- [ ] Review all loading states
  - [ ] Marketplace loading
  - [ ] Model downloading
  - [ ] Worker spawning
  - [ ] Worker generating
- [ ] Review all error messages
  - [ ] Clear and actionable
  - [ ] Suggest next steps
  - [ ] No technical jargon
- [ ] Review all success messages
  - [ ] Celebrate completions
  - [ ] Guide to next action
- [ ] Accessibility review
  - [ ] Keyboard navigation works
  - [ ] Screen reader support
  - [ ] Focus indicators visible
  - [ ] Color contrast meets WCAG

**Performance Optimization:**
- [ ] Profile tab system performance
- [ ] Optimize re-renders in TabContent
- [ ] Lazy load worker interfaces
- [ ] Optimize SSE connection handling
- [ ] Test with 4 active workers
- [ ] Monitor memory usage
- [ ] Fix any performance issues

**Documentation:**
- [ ] Update README with new features
- [ ] Create user guide
  - [ ] How to browse marketplaces
  - [ ] How to download models
  - [ ] How to spawn workers
  - [ ] How to use LLM chat
  - [ ] How to generate images
- [ ] Create troubleshooting guide
  - [ ] Model won't download
  - [ ] Worker won't spawn
  - [ ] GPU not detected
  - [ ] Out of memory errors
- [ ] Update architecture docs
- [ ] Create API documentation

**Launch Preparation:**
- [ ] Record demo video (5 minutes)
  - [ ] Follow WOW_FACTOR_LAUNCH_MVP.md script
  - [ ] Show dual-GPU simultaneous work
  - [ ] Professional narration
  - [ ] High-quality recording
- [ ] Take screenshots
  - [ ] Marketplace browsing
  - [ ] Worker spawning wizard
  - [ ] Split-screen demo
  - [ ] Individual interfaces
- [ ] Write blog post
  - [ ] Problem: Wasted GPUs
  - [ ] Solution: Bee Keeper
  - [ ] Demo: Split-screen
  - [ ] Call to action
- [ ] Prepare social media posts
  - [ ] Twitter/X thread
  - [ ] Reddit posts (r/LocalLLaMA, r/StableDiffusion)
  - [ ] HackerNews submission
  - [ ] Product Hunt launch
- [ ] Test on clean machine
  - [ ] Fresh install
  - [ ] Follow setup guide
  - [ ] Verify everything works
- [ ] Practice demo 10+ times
  - [ ] Smooth narration
  - [ ] No mistakes
  - [ ] Confident delivery

**Launch Day:**
- [ ] Post demo video to YouTube
- [ ] Post to Twitter/X
- [ ] Post to Reddit
- [ ] Submit to HackerNews
- [ ] Submit to Product Hunt
- [ ] Monitor feedback
- [ ] Respond to questions
- [ ] Fix critical bugs immediately

**Deliverables:** Production-ready MVP, launched!

---

## 🎯 Critical Path Summary

**Sequential Dependencies:**

```
Phase 0 (Backend) → Phase 1 (Worker Spawning) → Required for all
                                                ↓
                    Phase 2 (Tab System) → Required for Phase 3, 4, 5
                                                ↓
        ┌──────────────────────┬────────────────┴──────────────┐
        ↓                      ↓                                ↓
Phase 3 (Marketplaces)  Phase 4 (Spawning UX)  Phase 5 (Worker UI)
        │                      │                                │
        └──────────────────────┴────────────────┬───────────────┘
                                                ↓
                              Phase 6 (Integration & Launch)
```

**Parallel Work Opportunities:**
- Phase 3 and 4 can run in parallel after Phase 2
- Phase 5 can start as soon as Phase 4 completes
- TEAM-BACKEND can help with integration testing in Phase 6

---

## 📊 Success Criteria Per Phase

### Phase 0 ✅
- [ ] `cargo test` passes for all catalog crates
- [ ] Can distinguish LLM from SD models
- [ ] Can download from CivitAI

### Phase 1 ✅
- [ ] Can spawn LLM worker via curl
- [ ] Can spawn SD worker via curl
- [ ] Incompatible models are rejected

### Phase 2 ✅
- [ ] Can open multiple tabs
- [ ] Can reorder tabs by dragging
- [ ] Tabs persist across reload
- [ ] Sidebar opens tabs

### Phase 3 ✅
- [ ] All 3 marketplaces show items
- [ ] Search/filter works
- [ ] Download shows progress

### Phase 4 ✅
- [ ] 3-step wizard works
- [ ] Worker spawns successfully
- [ ] Tab opens automatically

### Phase 5 ✅
- [ ] LLM chat streams responses
- [ ] SD generation shows progress
- [ ] Both work simultaneously

### Phase 6 ✅
- [ ] Demo video recorded
- [ ] All bugs fixed
- [ ] Launched on all platforms

---

## 📝 Daily Standup Format

**Each team reports:**
1. ✅ Completed yesterday
2. 🎯 Working on today
3. ⚠️ Blockers (if any)

**Example:**
```
TEAM-BACKEND (Day 3/5):
✅ ModelType enum implemented
✅ WorkerType extended with SD variants
🎯 Implementing CivitAI vendor
⚠️ Need CivitAI API token for testing
```

---

## 🚀 Ready to Execute!

**All work is:**
- ✅ Broken down into actionable tasks
- ✅ Organized by team
- ✅ Sequenced by dependencies
- ✅ Testable at each phase
- ✅ Ready to start immediately

**Next step:** Assign teams and begin Phase 0!
