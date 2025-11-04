# Checklist 06: Launch Demo (WOW FACTOR MVP)

**Timeline:** 3 days  
**Status:** 📋 NOT STARTED  
**Dependencies:** CHECKLIST_05 (Keeper UI must be complete)  
**TEAM-400:** ✅ Updated to reference corrected checklists

---

## 🎯 Goal

Prepare and execute the "WOW FACTOR" demo: Google search → Running model in under 60 seconds. Record professional demo video for launch.

**TEAM-400:** Demo showcases the complete flow from all previous checklists.

---

## 📚 Reference Documents

**PRIMARY:**
- [WOW_FACTOR_LAUNCH_MVP.md](./WOW_FACTOR_LAUNCH_MVP.md) - Complete demo script

**SUPPORTING:**
- [COMPLETE_ONBOARDING_FLOW.md](./COMPLETE_ONBOARDING_FLOW.md) - User journey
- [WORKER_SPAWNING_3_STEPS.md](./WORKER_SPAWNING_3_STEPS.md) - Spawning UX

---

## Phase 1: Demo Environment Setup (Day 1 Morning)

**TEAM-400:** Verify all components from previous checklists work together.

### 1.1 Verify Marketplace Site (CHECKLIST_03)

- [ ] **Test marketplace.rbee.dev is live**
  - Open in browser
  - Verify Llama 3.2 1B page exists
  - Verify "Run with rbee" button shows
  - Verify installation detection works
  **Verification:** Site loads, buttons work

- [ ] **Test SSG pages**
  - Check `/models` page loads
  - Check `/models/llama-3.2-1b` page loads
  - Check SEO metadata is correct
  **Verification:** All pages pre-rendered

### 1.2 Verify Protocol Handler (CHECKLIST_04)

- [ ] **Test rbee:// protocol registration**
  ```bash
  # macOS
  open "rbee://model/llama-3.2-1b"
  
  # Linux
  xdg-open "rbee://model/llama-3.2-1b"
  
  # Windows
  start "rbee://model/llama-3.2-1b"
  ```
  **Verification:** Keeper opens

- [ ] **Test protocol from browser**
  - Click "Run with rbee" on marketplace
  - Verify browser prompts to open Keeper
  - Verify Keeper opens and navigates to marketplace
  **Verification:** Protocol flow works

### 1.3 Verify Keeper UI (CHECKLIST_05)

- [ ] **Test marketplace page in Keeper**
  - Open Keeper
  - Click "Marketplace" in sidebar
  - Verify models list loads
  - Verify search works
  **Verification:** Marketplace page works

- [ ] **Test install flow**
  - Click "Install" on a model
  - Verify download starts
  - Verify progress shows
  - Verify worker spawns
  **Verification:** Full install flow works

### 1.4 Hardware Verification

- [ ] **Verify GPU setup**
  - Run `nvidia-smi` (or equivalent)
  - Confirm GPU 0 and GPU 1 detected
  - Check VRAM: Min 8GB each
  **Verification:** Both GPUs available

- [ ] **Verify models downloaded**
  - LLM: Llama 3.2 1B Instruct (~1GB)
  - Image: Flux.1 Schnell (~12GB) OR another model
  **Location:** `~/.cache/rbee/models/`
  **Verification:** Models exist

- [ ] **Verify worker binaries**
  - cuda-llm-worker (for Llama)
  - cuda-sd-worker (for Flux) OR use llm worker
  **Location:** `~/.cache/rbee/workers/`
  **Verification:** Binaries exist

- [ ] **Test hive start**
  ```bash
  rbee-keeper hive start
  ```
  **Verification:** Hive starts, port 9200 listening

---

## Phase 2: Demo Script Preparation (Day 1 Afternoon)

### 2.1 Demo Flow

**TEAM-400:** Updated to match actual architecture.

**Flow:**
1. **Start:** Open browser → marketplace.rbee.dev (3s)
2. **Browse:** Search "Llama 3.2 1B" (2s)
3. **Click:** "Run with rbee" button (1s)
4. **Protocol:** Browser prompts, user clicks "Open" (2s)
5. **Keeper Opens:** Navigates to marketplace, shows model (2s)
6. **Install:** Click "Install" button (1s)
7. **Download:** Progress bar shows download (5-10s)
8. **Spawn:** Worker spawns automatically (3s)
9. **Ready:** Model ready to use (1s)

**Total: ~20-30 seconds from Google to running model**

**Verification:** Can complete demo in <45 seconds

### 2.2 Visual Design

- [ ] **Set up color scheme**
  - GPU 0 Tab: Electric Blue (#00D9FF)
  - GPU 1 Tab: Neon Pink (#FF006E)
  - Background: Dark (#0A0A0F)
  **Implementation:** Update Keeper theme
  **Verification:** Colors look good

- [ ] **Create tab icons**
  - 💬 Chat icon for LLM tab
  - 🎨 Image icon for Image Gen tab
  **Verification:** Icons visible

### 2.3 Demo Content

- [ ] **Prepare chat prompt**
  - Test: "Write a haiku about AI"
  - Length: ~50 tokens
  - Should complete in 1-2 seconds
  **Verification:** Good output

- [ ] **Prepare image prompt** (if using image model)
  - Test: "A futuristic cityscape at sunset"
  - Should generate in 5-8 seconds
  **Verification:** Good image

- [ ] **Test simultaneous execution** (if dual GPU)
  - Start chat on GPU 0
  - Start image on GPU 1
  - Verify both complete
  **Verification:** No conflicts

### 2.4 Timing Rehearsal

- [ ] **Practice demo flow 3 times**
  - Time each step
  - Note any slowdowns
  - Optimize where possible
  **Verification:** Consistent timing

---

## Phase 3: Recording Setup (Day 2 Morning)

### 3.1 Screen Recording

- [ ] **Install recording software**
  - macOS: QuickTime or ScreenFlow
  - Linux: OBS Studio or SimpleScreenRecorder
  - Windows: OBS Studio
  **Verification:** Software works

- [ ] **Configure recording settings**
  - Resolution: 1920x1080
  - Frame rate: 60 FPS
  - Audio: System audio + microphone (optional)
  - Format: MP4 (H.264)
  **Verification:** Test recording plays smoothly

- [ ] **Set up browser window**
  - Chrome/Firefox clean profile
  - Window size: 1280x720
  - Zoom: 125% (readable)
  **Verification:** Text readable

- [ ] **Set up Keeper window**
  - Window size: Full screen or 1920x1080
  - Font size: Increase for readability
  - Hide personal info
  **Verification:** UI clear

### 3.2 Voiceover Script

- [ ] **Write voiceover script** (optional)
  ```
  "Watch as I go from Google search to running an AI model in under 60 seconds.
  
  I search for Llama 3.2, click 'Run with rbee', and boom—Keeper opens.
  
  The model downloads and spawns automatically. I type a prompt, and it generates instantly.
  
  Zero configuration. That's rbee."
  ```
  **Length:** ~20 seconds
  **Verification:** Matches timing

- [ ] **Practice voiceover**
  - Record test audio
  - Check quality
  - Speak clearly
  **Verification:** Audio clear

---

## Phase 4: Demo Recording (Day 2 Afternoon)

### 4.1 Recording Sessions

- [ ] **Record demo take 1**
  - Follow script
  - Note mistakes
  **Verification:** Recording completes

- [ ] **Review take 1**
  - Check for errors
  - Check audio sync
  - Note improvements
  **Verification:** Identify issues

- [ ] **Record demo take 2**
  - Fix issues
  - Aim for smooth execution
  **Verification:** Better than take 1

- [ ] **Record demo take 3** (if needed)
  - Polish details
  - Capture "wow factor"
  **Verification:** This is "the one"

### 4.2 Video Editing

- [ ] **Select best take**
  - Compare recordings
  - Choose smoothest
  **Verification:** Selected take is polished

- [ ] **Add intro card** (5 seconds)
  ```
  "rbee - Your Personal AI Infrastructure"
  "Watch: Google Search to Running Model in 60 Seconds"
  ```
  **Verification:** Intro looks professional

- [ ] **Add outro card** (5 seconds)
  ```
  "Get Started: rbee.dev"
  "Open Source • Self-Hosted • Privacy-First"
  ```
  **Verification:** Call-to-action clear

- [ ] **Add captions** (optional)
  - Transcribe voiceover
  - Add as subtitles
  **Verification:** Captions sync

- [ ] **Add music** (optional)
  - Royalty-free music
  - Low volume (background)
  **Verification:** Music doesn't overpower

---

## Phase 5: Launch Materials (Day 3)

### 5.1 Video Assets

- [ ] **Export final video**
  - Format: MP4 (H.264)
  - Resolution: 1920x1080
  - Frame rate: 60 FPS
  - Bitrate: 8 Mbps
  **Verification:** File size ~50-100 MB

- [ ] **Create thumbnail**
  - Show Keeper UI with model running
  - Text: "Google to Running Model in 60s"
  - Brand colors
  - Resolution: 1280x720
  **Verification:** Thumbnail eye-catching

- [ ] **Create GIF preview** (optional)
  - Extract 5-10 second clip
  - Show key moment
  - Under 5 MB
  **Verification:** GIF loops smoothly

### 5.2 Written Content

- [ ] **Write YouTube description**
  ```markdown
  # rbee - Your Personal AI Infrastructure
  
  Watch as I go from a Google search to running an AI model in under 60 seconds—with ZERO configuration.
  
  ## What You'll See:
  - Browse models on marketplace.rbee.dev
  - One-click install with rbee:// protocol
  - Automatic download and worker spawning
  - Model ready to use in seconds
  
  ## Features:
  - 🚀 Zero configuration
  - 🔒 Privacy-first (runs locally)
  - 🆓 Open source
  - 🎯 Simple workflow
  
  Get started: https://rbee.dev
  GitHub: https://github.com/veighnsche/llama-orch
  
  #AI #OpenSource #LocalAI #LLM #SelfHosted
  ```
  **Verification:** Description complete

- [ ] **Write Twitter/X thread**
  ```
  🚀 Introducing rbee: Your Personal AI Infrastructure
  
  Watch me go from Google search to running an AI model in under 60 seconds.
  
  No Docker. No config files. No cloud APIs.
  
  Just one click. 🧵👇
  
  [Video]
  
  1/ The problem: Running AI models locally is too complicated.
  
  You need to install dependencies, configure workers, download models, and manage everything manually.
  
  2/ The solution: rbee makes it as easy as installing a browser extension.
  
  Browse models → Click "Run with rbee" → Model downloads and spawns automatically.
  
  3/ How it works:
  - Marketplace site (marketplace.rbee.dev)
  - rbee:// protocol handler
  - Keeper desktop app
  - Automatic worker management
  
  4/ It's open source, privacy-first, and runs entirely on your hardware.
  
  No cloud. No tracking. No limits.
  
  Get started: https://rbee.dev
  GitHub: https://github.com/veighnsche/llama-orch
  ```
  **Verification:** Thread ready

- [ ] **Write Reddit post**
  ```markdown
  # rbee: Google Search to Running AI Model in 60 Seconds
  
  I built a system that makes running local AI models as easy as installing a browser extension.
  
  **Demo:** [YouTube link]
  
  ## The Problem
  Running AI models locally is complicated. You need Docker, config files, manual downloads, and lots of terminal commands.
  
  ## The Solution
  rbee simplifies everything:
  1. Browse models on marketplace.rbee.dev
  2. Click "Run with rbee"
  3. Keeper desktop app opens and handles everything
  4. Model ready in seconds
  
  ## How It Works
  - **Marketplace:** Next.js site with 1000+ pre-rendered model pages
  - **Protocol Handler:** rbee:// protocol (like mailto:)
  - **Keeper:** Tauri desktop app that manages workers
  - **Auto-Run:** Automatic download + spawn
  
  ## Tech Stack
  - Rust (backend, WASM SDK)
  - Next.js (marketplace)
  - Tauri (desktop app)
  - React (UI components)
  
  ## Open Source
  - GitHub: https://github.com/veighnsche/llama-orch
  - License: GPL-3.0
  - Contributions welcome!
  
  Feedback appreciated! 🙏
  ```
  **Verification:** Post ready

---

## Phase 6: Launch Checklist (Day 3 Afternoon)

### 6.1 Pre-Launch

- [ ] **Verify all components deployed**
  - [ ] marketplace.rbee.dev is live (CHECKLIST_03)
  - [ ] Keeper installers available (CHECKLIST_04)
  - [ ] Protocol handler works on all platforms
  - [ ] Marketplace page works in Keeper (CHECKLIST_05)

- [ ] **Test full flow one more time**
  - Fresh install of Keeper
  - Test from marketplace site
  - Verify protocol works
  - Verify install works
  **Verification:** Everything works

- [ ] **Prepare social media accounts**
  - Twitter/X account ready
  - Reddit account ready
  - YouTube channel ready
  - LinkedIn account ready (optional)

### 6.2 Launch Day

- [ ] **Upload video to YouTube**
  - Title: "rbee: Google Search to Running AI Model in 60 Seconds"
  - Description: (from 5.2)
  - Thumbnail: (from 5.1)
  - Tags: AI, OpenSource, LocalAI, LLM, SelfHosted
  **Verification:** Video published

- [ ] **Post on Twitter/X**
  - Post thread (from 5.2)
  - Include video link
  - Use hashtags
  **Verification:** Thread posted

- [ ] **Post on Reddit**
  - r/LocalLLaMA
  - r/selfhosted
  - r/opensource
  - Use post from 5.2
  **Verification:** Posts live

- [ ] **Post on Hacker News** (optional)
  - Title: "rbee: Google Search to Running AI Model in 60 Seconds"
  - URL: YouTube video or GitHub
  **Verification:** Posted

- [ ] **Post on LinkedIn** (optional)
  - Professional version of announcement
  - Focus on technical achievement
  **Verification:** Posted

### 6.3 Post-Launch

- [ ] **Monitor feedback**
  - Check comments on YouTube
  - Check replies on Twitter
  - Check Reddit comments
  - Respond to questions

- [ ] **Track metrics**
  - Video views
  - GitHub stars
  - Keeper downloads
  - Website traffic

- [ ] **Iterate based on feedback**
  - Note common questions
  - Note feature requests
  - Note bugs reported

---

## ✅ Success Criteria

### Must Have

- [ ] Demo video recorded and edited
- [ ] Video uploaded to YouTube
- [ ] Social media posts published
- [ ] All components working (marketplace, protocol, Keeper)
- [ ] Full flow tested end-to-end

### Nice to Have

- [ ] Professional voiceover
- [ ] Background music
- [ ] Captions
- [ ] Multiple platform posts (Twitter, Reddit, HN)
- [ ] Press coverage

---

## 🚀 Deliverables

1. **Demo Video:** 60-second professional recording
2. **YouTube Upload:** Published with description and thumbnail
3. **Social Media Posts:** Twitter thread, Reddit posts
4. **Launch Materials:** All written content ready
5. **Metrics Tracking:** Analytics set up

---

## 📝 Notes

### Key Principles

1. **SHOW, DON'T TELL** - Demo the actual product
2. **SIMPLE FLOW** - Google → Running model in 60s
3. **WOW FACTOR** - Make it look effortless
4. **PROFESSIONAL** - High-quality video and audio
5. **HONEST** - Show real product, no fake demos

### Common Pitfalls

- ❌ Don't fake the demo (use real product)
- ❌ Don't skip testing (verify everything works)
- ❌ Don't rush recording (take multiple takes)
- ❌ Don't forget call-to-action (link to rbee.dev)
- ✅ Test full flow multiple times
- ✅ Record multiple takes
- ✅ Edit professionally
- ✅ Launch on multiple platforms

### Demo Timing

**Target:** <60 seconds total
- Intro card: 5s
- Demo: 30-40s
- Outro card: 5s
- Total: 40-50s

**Actual demo flow:** ~20-30s
- Browse marketplace: 5s
- Click "Run with rbee": 2s
- Keeper opens: 2s
- Download + spawn: 10-15s
- Model ready: 1s

---

**Start with Phase 1, verify all components work!** ✅

**TEAM-400 🐝🎊**
