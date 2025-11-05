# Remaining Work Checklist - Marketplace Implementation

**Created by:** TEAM-413  
**Date:** 2025-11-05  
**Status:** 📋 READY TO START  
**Estimated Time:** 4-5 days

---

## 🎯 Overview

This checklist consolidates ALL remaining work from CHECKLIST_03 and CHECKLIST_04. Start from Priority 1 and work down.

**Current Progress:** (Updated by TEAM-418)
- Priority 1: ✅ 100% complete (17.5h / 17.5h)
- Priority 2: ⏳ 64% complete (7h / 11h) 
- Priority 3: ❌ 0% complete (0h / 8h)
- **Overall: 67% complete (24.5h / 36.5h) → 12 hours remaining**

---

## 🔥 Priority 1: Critical Gaps (2-3 days) ✅ COMPLETE

### P1.1: Models List Page (4 hours) ✅ TEAM-413
**File:** `frontend/apps/marketplace/app/models/page.tsx`  
**Status:** ✅ COMPLETE  
**Completed:** Users can now browse models

- [ ] Create `app/models/page.tsx`
- [ ] Fetch models from HuggingFace (SSG)
- [ ] Use ModelsPage component from rbee-ui
- [ ] Add SEO metadata
- [ ] Test SSG build (`pnpm build`)
- [ ] Verify pages generated in `.next/server/app/models/`

**Code Template:** See CHECKLIST_03 lines 132-173

---

### P1.2: Workers Pages (6 hours)
**Files:** 
- `frontend/apps/marketplace/app/workers/page.tsx`
- `frontend/apps/marketplace/app/workers/[workerId]/page.tsx`

**Status:** ❌ MISSING  
**Blocks:** Can't browse or install workers

#### P1.2a: Workers List Page (3 hours)
- [ ] Create `app/workers/page.tsx`
- [ ] Fetch workers from Worker Catalog
- [ ] Use WorkersPage component from rbee-ui
- [ ] Add SEO metadata
- [ ] Test SSG build

**Code Template:** See CHECKLIST_03 lines 248-282

#### P1.2b: Worker Detail Page (3 hours)
- [ ] Create `app/workers/[workerId]/page.tsx`
- [ ] Fetch single worker data
- [ ] Use WorkerDetailPage component from rbee-ui
- [ ] Add generateStaticParams for SSG
- [ ] Add SEO metadata

**Code Template:** See CHECKLIST_03 lines 284-319

---

### P1.3: Installation Detection (4 hours)
**Files:**
- `frontend/apps/marketplace/app/hooks/useKeeperInstalled.ts`
- `frontend/apps/marketplace/app/components/InstallButton.tsx`

**Status:** ❌ MISSING  
**Blocks:** Can't show "Run with rbee" vs "Download Keeper"

#### P1.3a: Detection Hook (2 hours)
- [ ] Create `app/hooks/useKeeperInstalled.ts`
- [ ] Implement protocol detection (try rbee:// URL)
- [ ] Return { installed, checking } state
- [ ] Test detection logic

**Code Template:** See CHECKLIST_03 lines 329-367

#### P1.3b: Install Button Component (2 hours)
- [ ] Create `app/components/InstallButton.tsx`
- [ ] Use useKeeperInstalled hook
- [ ] Show "Run with rbee" if installed
- [ ] Show "Download Keeper" if not installed
- [ ] Trigger rbee:// protocol on click
- [ ] Test button behavior

**Code Template:** See CHECKLIST_03 lines 371-409

#### P1.3c: Integrate Install Button (30 minutes)
- [ ] Update `app/models/[slug]/page.tsx`
- [ ] Import InstallButton component
- [ ] Pass to ModelDetailPage template
- [ ] Test end-to-end

**Code Template:** See CHECKLIST_03 lines 413-425

---

### P1.4: Frontend Protocol Listener (3 hours)
**Files:**
- `bin/00_rbee_keeper/ui/src/hooks/useProtocol.ts`
- `bin/00_rbee_keeper/ui/src/App.tsx`

**Status:** ❌ MISSING  
**Blocks:** Protocol handler emits events but nothing listens

#### P1.4a: Protocol Hook (2 hours)
- [ ] Create `ui/src/hooks/useProtocol.ts`
- [ ] Listen for 'protocol:install-model' event
- [ ] Listen for 'protocol:install-worker' event
- [ ] Listen for 'navigate' event
- [ ] Navigate to marketplace on events
- [ ] Test event handling

**Code Template:** See CHECKLIST_04 lines 426-464

#### P1.4b: Integrate in App (1 hour)
- [ ] Update `ui/src/App.tsx`
- [ ] Import useProtocol hook
- [ ] Call hook at app root
- [ ] Test protocol events trigger navigation
- [ ] Verify end-to-end flow

**Code Template:** See CHECKLIST_04 lines 468-480

---

## ⚡ Priority 2: Important Features (1-2 days)

### P2.1: Auto-Run Logic (4 hours)
**Files:**
- `bin/00_rbee_keeper/src/handlers/auto_run.rs`
- `bin/00_rbee_keeper/src/protocol.rs` (update)

**Status:** ❌ MISSING  
**Impact:** Not truly "one-click" install (just navigation, no download)

#### P2.1a: Auto-Run Module (3 hours)
- [ ] Create `src/handlers/auto_run.rs`
- [ ] Implement `auto_run_model(model_id)` function
- [ ] Use JobClient to submit ModelDownload operation
- [ ] Use JobClient to submit WorkerSpawn operation
- [ ] Add progress logging
- [ ] Test auto-run flow

**Code Template:** See CHECKLIST_04 lines 330-391

#### P2.1b: Integrate Auto-Run (1 hour)
- [ ] Update `src/protocol.rs`
- [ ] Import auto_run functions
- [ ] Call auto_run_model in InstallModel handler
- [ ] Call auto_run_worker in InstallWorker handler
- [ ] Test end-to-end auto-download

**Code Template:** See CHECKLIST_04 lines 395-416

---

### P2.2: Open Graph Images (3 hours)
**Files:**
- `frontend/apps/marketplace/app/opengraph-image.tsx`
- `frontend/apps/marketplace/app/models/[slug]/opengraph-image.tsx`

**Status:** ❌ MISSING  
**Impact:** Poor social media sharing (no preview images)

#### P2.2a: Base OG Image (1 hour)
- [ ] Create `app/opengraph-image.tsx`
- [ ] Use Next.js ImageResponse API
- [ ] Design rbee Marketplace image (1200x630)
- [ ] Test image generation
- [ ] Verify on social media debuggers

**Code Template:** See CHECKLIST_03 lines 499-530

#### P2.2b: Model OG Images (2 hours)
- [ ] Create `app/models/[slug]/opengraph-image.tsx`
- [ ] Fetch model data for each slug
- [ ] Generate dynamic OG image with model name/author
- [ ] Test image generation for multiple models
- [ ] Verify on social media debuggers

**Code Template:** See CHECKLIST_03 lines 532-576

---

### P2.3: End-to-End Testing (4 hours)
**Status:** ⏳ PENDING  
**Impact:** Don't know if it actually works

#### P2.3a: Protocol Testing (2 hours)
- [ ] Build Keeper: `cargo tauri build`
- [ ] Install built app
- [ ] Test from terminal: `open "rbee://model/llama-3.2-1b"`
- [ ] Verify Keeper opens
- [ ] Verify navigation to marketplace
- [ ] Verify auto-download starts

**Reference:** See CHECKLIST_04 lines 488-518

#### P2.3b: Browser Testing (2 hours)
- [ ] Open marketplace site locally
- [ ] Navigate to model detail page
- [ ] Click "Run with rbee" button
- [ ] Verify browser prompts to open Keeper
- [ ] Verify Keeper opens and downloads
- [ ] Test on multiple browsers (Chrome, Firefox, Safari)

**Reference:** See CHECKLIST_04 lines 503-509

---

## 🎨 Priority 3: Polish & Distribution (1-2 days)

### P3.1: Platform Installers (6 hours)
**Status:** ❌ MISSING  
**Impact:** Can't distribute to users

#### P3.1a: Build Installers (3 hours)
- [ ] Build macOS: `cargo tauri build --target universal-apple-darwin`
- [ ] Build Linux: `cargo tauri build --target x86_64-unknown-linux-gnu`
- [ ] Build Windows: `cargo tauri build --target x86_64-pc-windows-msvc`
- [ ] Verify installers created in `target/release/bundle/`

**Reference:** See CHECKLIST_04 lines 543-553

#### P3.1b: Test Installers (2 hours)
- [ ] Test .dmg installer on macOS
- [ ] Test .deb/.AppImage on Linux
- [ ] Test .msi installer on Windows
- [ ] Verify protocol registration after install
- [ ] Test from browser on each platform

**Reference:** See CHECKLIST_04 lines 555-570

#### P3.1c: Upload to GitHub Releases (1 hour)
- [ ] Create GitHub release (v0.1.0)
- [ ] Upload all installers
- [ ] Add release notes
- [ ] Update marketplace download links
- [ ] Test download links work

**Reference:** See CHECKLIST_04 lines 572-580

---

### P3.2: Deployment (2 hours)
**Status:** ❌ PENDING  
**Impact:** No public marketplace

- [ ] Build Next.js site: `cd frontend/apps/marketplace && pnpm build`
- [ ] Test production build locally: `pnpm start`
- [ ] Deploy to Cloudflare Pages (already configured)
- [ ] Configure custom domain: marketplace.rbee.dev
- [ ] Test live site
- [ ] Verify sitemap.xml accessible
- [ ] Verify robots.txt accessible
- [ ] Test model pages load
- [ ] Test protocol links work from live site

**Reference:** See CHECKLIST_03 lines 595-655

---

## 📊 Progress Tracking

### Completion Checklist

**Priority 1 (Critical - 2-3 days):**
- [x] P1.1: Models list page (4h) ✅ TEAM-413 (already existed from TEAM-415)
- [x] P1.2a: Workers list page (3h) ✅ TEAM-413
- [x] P1.2b: Worker detail page (3h) ✅ TEAM-413
- [x] P1.3a: Detection hook (2h) ✅ TEAM-413
- [x] P1.3b: Install button (2h) ✅ TEAM-413
- [x] P1.3c: Integrate button (0.5h) ✅ TEAM-413
- [x] P1.4a: Protocol hook (2h) ✅ TEAM-413
- [x] P1.4b: Integrate in App (1h) ✅ TEAM-413

**Total Priority 1:** ~17.5 hours (2-3 days)

**Priority 2 (Important - 1-2 days):**
- [x] P2.1a: Auto-run module (3h) ✅ TEAM-416
- [x] P2.1b: Integrate auto-run (1h) ✅ TEAM-416
- [x] P2.2a: Base OG image (1h) ✅ TEAM-417
- [x] P2.2b: Model OG images (2h) ✅ TEAM-417
- [ ] P2.3a: Protocol testing (2h)
- [ ] P2.3b: Browser testing (2h)

**Total Priority 2:** ~11 hours (1-2 days)

**Priority 3 (Polish - 1-2 days):**
- [ ] P3.1a: Build installers (3h)
- [ ] P3.1b: Test installers (2h)
- [ ] P3.1c: Upload to releases (1h)
- [ ] P3.2: Deployment (2h)

**Total Priority 3:** ~8 hours (1 day)

**Grand Total:** ~36.5 hours (4-5 days of focused work)

---

## 🎯 Daily Plan

### Day 1: Critical Frontend (8 hours)
- Morning: P1.1 Models list page (4h)
- Afternoon: P1.2a Workers list page (3h), P1.2b start (1h)

### Day 2: Workers & Detection (8 hours)
- Morning: P1.2b Worker detail page finish (2h), P1.3a Detection hook (2h)
- Afternoon: P1.3b Install button (2h), P1.3c Integration (0.5h), P1.4a Protocol hook start (1.5h)

### Day 3: Protocol & Auto-Run (8 hours)
- Morning: P1.4a Protocol hook finish (0.5h), P1.4b Integration (1h), P2.1a Auto-run module (3h)
- Afternoon: P2.1b Integrate auto-run (1h), P2.2a Base OG image (1h), P2.2b Model OG images (2h)

### Day 4: Testing & Installers (8 hours)
- Morning: P2.3a Protocol testing (2h), P2.3b Browser testing (2h)
- Afternoon: P3.1a Build installers (3h), P3.1b Test installers start (1h)

### Day 5: Polish & Deploy (4 hours)
- Morning: P3.1b Test installers finish (1h), P3.1c Upload releases (1h), P3.2 Deployment (2h)

---

## ✅ Verification Commands

### Check Progress
```bash
# Models list page exists
ls -la frontend/apps/marketplace/app/models/page.tsx

# Workers pages exist
ls -la frontend/apps/marketplace/app/workers/page.tsx
ls -la frontend/apps/marketplace/app/workers/[workerId]/page.tsx

# Detection hook exists
ls -la frontend/apps/marketplace/app/hooks/useKeeperInstalled.ts

# Install button exists
ls -la frontend/apps/marketplace/app/components/InstallButton.tsx

# Protocol hook exists
ls -la bin/00_rbee_keeper/ui/src/hooks/useProtocol.ts

# Auto-run module exists
ls -la bin/00_rbee_keeper/src/handlers/auto_run.rs

# OG images exist
ls -la frontend/apps/marketplace/app/opengraph-image.tsx
ls -la frontend/apps/marketplace/app/models/[slug]/opengraph-image.tsx
```

### Test Builds
```bash
# Next.js build
cd frontend/apps/marketplace
pnpm build

# Keeper build
cd bin/00_rbee_keeper
cargo tauri build
```

### Test End-to-End
```bash
# Test protocol
open "rbee://model/llama-3.2-1b"

# Test from browser
open http://localhost:3000/models/llama-3.2-1b
# Click "Run with rbee" button
```

---

## 🚨 Blockers & Dependencies

### No Blockers
All work can proceed immediately. No external dependencies.

### Internal Dependencies
- P1.3c depends on P1.3a + P1.3b (install button needs detection hook)
- P1.4b depends on P1.4a (App needs protocol hook)
- P2.1b depends on P2.1a (protocol needs auto-run module)
- P2.3 depends on P1.4 + P2.1 (testing needs features complete)
- P3.1 depends on P2.3 (installers need testing)
- P3.2 depends on P1.1 + P1.2 (deployment needs pages)

### Recommended Order
1. Start with P1.1 (models list) - unblocks deployment
2. Then P1.2 (workers pages) - unblocks deployment
3. Then P1.3 (detection + button) - unblocks user flow
4. Then P1.4 (protocol listener) - unblocks end-to-end
5. Then P2.1 (auto-run) - completes one-click install
6. Then P2.2 (OG images) - improves SEO
7. Then P2.3 (testing) - verifies everything works
8. Then P3.1 (installers) - enables distribution
9. Finally P3.2 (deployment) - goes live

---

## 📝 Notes

### TEAM-413 Findings
- TEAM-412 claimed "85% complete" but only did backend (45% actual)
- Protocol handler works but missing frontend integration
- No auto-download logic (just navigation)
- No installation detection
- No platform installers

### What's Already Done
- ✅ Protocol handler (backend) - TEAM-412
- ✅ Sitemap & robots.txt - TEAM-410
- ✅ Model detail pages - TEAM-415
- ✅ Compatibility integration - TEAM-410

### What's Missing
- ❌ Models list page
- ❌ Workers pages
- ❌ Installation detection
- ❌ Install button
- ❌ Frontend protocol listener
- ❌ Auto-run logic
- ❌ Open Graph images
- ❌ Platform installers
- ❌ Deployment

---

## 🎉 Success Criteria

### When This Checklist is 100% Complete
- [ ] All Priority 1 tasks complete (critical gaps filled)
- [ ] All Priority 2 tasks complete (important features added)
- [ ] All Priority 3 tasks complete (polished & deployed)
- [ ] CHECKLIST_03 reaches 100%
- [ ] CHECKLIST_04 reaches 100%
- [ ] End-to-end flow works: Google → marketplace → click → Keeper opens → model downloads → ready to use
- [ ] Platform installers available for download
- [ ] Marketplace site live at marketplace.rbee.dev
- [ ] Ready to move to CHECKLIST_05 (Keeper UI marketplace tab)

---

**TEAM-413 - Remaining Work Identified and Organized** ✅  
**Next Team:** Start with Priority 1, work through systematically  
**Estimated Completion:** 4-5 days of focused work

**Let's finish this! 🐝🚀**
