# Checklist 06: Launch Demo (WOW FACTOR MVP)

**Date:** 2025-11-04  
**Status:** 📋 READY TO START  
**Timeline:** 3 days  
**Dependencies:** CHECKLIST_05 (Keeper UI must be complete)

---

## 🎯 Goal

Prepare and execute the "WOW FACTOR" demo that showcases simultaneous LLM chat + image generation on two GPUs. Record professional demo video for launch.

---

## 📚 Reference Documents

**PRIMARY:**
- [WOW_FACTOR_LAUNCH_MVP.md](./WOW_FACTOR_LAUNCH_MVP.md) - Complete demo script (674 lines)

**SUPPORTING:**
- [COMPLETE_ONBOARDING_FLOW.md](./COMPLETE_ONBOARDING_FLOW.md) - User journey
- [WORKER_SPAWNING_3_STEPS.md](./WORKER_SPAWNING_3_STEPS.md) - Spawning UX

---

## Phase 1: Demo Environment Setup (Day 1 Morning)

### Hardware Verification

- [ ] **Verify dual GPU setup**
  - Run `nvidia-smi` or equivalent
  - Confirm GPU 0 and GPU 1 detected
  - Check VRAM: Min 8GB each
  **Verification:** Both GPUs listed with adequate VRAM

- [ ] **Verify models downloaded**
  - LLM: Llama 3.2 1B Instruct (GGUF format, ~1GB)
  - Image: Flux.1 Schnell (GGUF format, ~12GB)
  **Location:** `~/.cache/rbee/models/`
  **Verification:** `ls ~/.cache/rbee/models/` shows both models

- [ ] **Verify worker binaries installed**
  - cuda-llm-worker (for Llama)
  - cuda-sd-worker (for Flux) - if exists, otherwise use llm worker
  **Location:** `~/.cache/rbee/workers/`
  **Verification:** `ls ~/.cache/rbee/workers/` shows binaries

- [ ] **Test hive start**
  ```bash
  rbee-keeper hive start
  # or
  queen-rbee hive start localhost
  ```
  **Verification:** Hive starts without errors, port 8787 listening

### Software Verification

- [ ] **Test marketplace.rbee.dev is live**
  - Open in browser
  - Verify Llama 3.2 1B page exists
  - Verify Flux.1 Schnell page exists (if applicable)
  **Verification:** Both model pages load with "Run with rbee" button

- [ ] **Test protocol detection**
  - Click "Run with rbee" button
  - Verify Keeper opens (if installed)
  - Verify install modal shows (if not installed)
  **Verification:** Protocol handler works

- [ ] **Test auto-run flow end-to-end**
  - Close all workers
  - Click "Run with rbee" on Llama 3.2 1B
  - Verify Keeper opens with pre-filled wizard
  - Click "Spawn" button
  - Verify worker spawns and starts
  **Verification:** Worker running in <30 seconds

---

## Phase 2: Demo Script Preparation (Day 1 Afternoon)

### Visual Design

- [ ] **Set up color scheme** (from WOW_FACTOR_LAUNCH_MVP.md lines 232-285)
  - GPU 0 Tab: Electric Blue (#00D9FF)
  - GPU 1 Tab: Neon Pink (#FF006E)
  - Background: Dark (#0A0A0F)
  - Accent: Purple (#8B5CF6)
  **Implementation:** Update Keeper theme or CSS variables
  **Verification:** Colors match design doc

- [ ] **Create tab icons**
  - 💬 Chat icon for LLM tab
  - 🎨 Image icon for Image Gen tab
  - Icons visible in tab bar
  **Verification:** Icons display in tabs

- [ ] **Set up split-screen layout**
  - Two tabs side-by-side (if possible)
  - OR: Fast tab switching animation
  **Implementation:** May require custom layout component
  **Verification:** Layout looks clean

### Demo Content

- [ ] **Prepare chat prompts** (WOW_FACTOR_LAUNCH_MVP.md lines 287-370)
  - Test prompt: "Write a poem about AI"
  - Length: ~100 tokens
  - Should complete in 2-3 seconds on GPU
  **Verification:** Prompt generates good output

- [ ] **Prepare image prompts** (WOW_FACTOR_LAUNCH_MVP.md lines 372-428)
  - Test prompt: "A futuristic cityscape at sunset, cyberpunk style"
  - Should generate in 5-8 seconds on GPU
  **Verification:** Image quality is good

- [ ] **Test simultaneous execution**
  - Start chat generation on GPU 0
  - Immediately start image generation on GPU 1
  - Verify both complete successfully
  - Verify no GPU memory conflicts
  **Verification:** Both tasks complete without errors

### Timing Rehearsal

- [ ] **Practice demo flow** (WOW_FACTOR_LAUNCH_MVP.md lines 92-230)
  1. Open browser → marketplace.rbee.dev (3s)
  2. Search "Llama 3.2 1B" (2s)
  3. Click "Run with rbee" (1s)
  4. Keeper opens, wizard pre-filled (2s)
  5. Click "Spawn" → Worker tab opens (3s)
  6. Type chat prompt, press Enter (5s)
  7. Switch tabs, search "Flux Schnell" (3s)
  8. Click "Run with rbee", spawn on GPU 1 (5s)
  9. Type image prompt, press Enter (8s)
  10. Show both outputs side-by-side (5s)
  
  **Total: ~37 seconds from Google to two models running**
  
  **Verification:** Can complete demo in <45 seconds

---

## Phase 3: Recording Setup (Day 2 Morning)

### Screen Recording

- [ ] **Install recording software**
  - macOS: Use QuickTime or ScreenFlow
  - Linux: Use OBS Studio or SimpleScreenRecorder
  - Windows: Use OBS Studio
  **Verification:** Software installed and tested

- [ ] **Configure recording settings**
  - Resolution: 1920x1080 (minimum)
  - Frame rate: 60 FPS
  - Audio: System audio + microphone (optional)
  - Format: MP4 (H.264)
  **Verification:** Test recording plays back smoothly

- [ ] **Set up browser window**
  - Use Chrome/Firefox in clean profile (no extensions)
  - Window size: 1280x720 (fits nicely in recording)
  - Zoom: 125% (for readability)
  **Verification:** Text is readable in recording

- [ ] **Set up Keeper window**
  - Window size: Full screen or 1920x1080
  - Font size: Increase for readability
  - Hide personal info (paths, usernames)
  **Verification:** UI is clear in recording

### Voiceover Script

- [ ] **Write voiceover script** (optional, recommended)
  ```
  "Watch as I go from Google search to running two AI models in under 60 seconds.
  
  I search for Llama 3.2, click 'Run with rbee', and boom—Keeper opens.
  
  The model spawns on GPU 0, I type a prompt, and it generates instantly.
  
  Now I search for Flux, click run, spawn on GPU 1, and generate an image.
  
  Two models, two GPUs, zero configuration. That's rbee."
  ```
  **Length:** ~30 seconds
  **Verification:** Script matches demo timing

- [ ] **Practice voiceover**
  - Record test audio
  - Check audio quality (no background noise)
  - Speak clearly and at moderate pace
  **Verification:** Audio is clear and professional

---

## Phase 4: Demo Recording (Day 2 Afternoon)

### Recording Sessions

- [ ] **Record demo take 1**
  - Follow script exactly
  - Don't worry about perfection
  - Note any mistakes or improvements
  **Verification:** Recording completes without crashes

- [ ] **Review take 1**
  - Check for errors (wrong clicks, slow loading)
  - Check audio sync (if using voiceover)
  - Note timestamp of any issues
  **Verification:** Identify improvements needed

- [ ] **Record demo take 2**
  - Fix issues from take 1
  - Aim for smooth execution
  - Keep energy high
  **Verification:** Better than take 1

- [ ] **Record demo take 3** (if needed)
  - Polish final details
  - Ensure perfect timing
  - Capture "wow factor" moment
  **Verification:** This is "the one"

### Video Editing

- [ ] **Select best take**
  - Compare all recordings
  - Choose smoothest execution
  - May combine best parts from multiple takes
  **Verification:** Selected take is polished

- [ ] **Add intro card** (5 seconds)
  ```
  "rbee - Your Personal AI Infrastructure"
  "Watch: Google Search to Running Model in 60 Seconds"
  ```
  **Tool:** Use Canva or video editor
  **Verification:** Intro looks professional

- [ ] **Add outro card** (5 seconds)
  ```
  "Get Started: rbee.dev"
  "Open Source • Self-Hosted • Privacy-First"
  ```
  **Verification:** Call-to-action is clear

- [ ] **Add captions** (optional but recommended)
  - Transcribe voiceover
  - Add as subtitles
  - Use readable font (24pt+)
  **Tool:** YouTube auto-caption or manual
  **Verification:** Captions sync with audio

- [ ] **Add music** (optional)
  - Use royalty-free music (YouTube Audio Library)
  - Volume: Low (background only)
  - Genre: Tech/Electronic
  **Verification:** Music doesn't overpower voice

---

## Phase 5: Launch Materials (Day 3 Morning)

### Video Assets

- [ ] **Export final video**
  - Format: MP4 (H.264)
  - Resolution: 1920x1080
  - Frame rate: 60 FPS
  - Bitrate: 8 Mbps (high quality)
  **Verification:** File size ~50-100 MB for 60s video

- [ ] **Create thumbnail**
  - Show split-screen: Chat + Image Gen
  - Add text overlay: "2 GPUs • 1 Minute • 0 Config"
  - Use brand colors (blue + pink)
  - Resolution: 1280x720 (YouTube standard)
  **Tool:** Canva or Photoshop
  **Verification:** Thumbnail is eye-catching

- [ ] **Create GIF preview** (optional)
  - Extract 5-10 second clip
  - Show key moment (model spawning)
  - Resolution: 800x450
  - Under 5 MB file size
  **Tool:** ezgif.com or ffmpeg
  **Verification:** GIF loops smoothly

### Written Content

- [ ] **Write YouTube description**
  ```markdown
  # rbee - Your Personal AI Infrastructure
  
  Watch as I go from a Google search to running TWO AI models (LLM + Image Gen)
  on two GPUs in under 60 seconds—with ZERO configuration.
  
  ## What You Just Saw:
  ✅ SEO-optimized marketplace (marketplace.rbee.dev)
  ✅ One-click model deployment
  ✅ Automatic worker spawning
  ✅ Multi-GPU support
  ✅ Real-time generation
  
  ## Get Started:
  🌐 Website: rbee.dev
  📦 Install: [link]
  📖 Docs: [link]
  💻 GitHub: [link]
  
  ## Tech Stack:
  - Rust backend (queen-rbee, rbee-hive)
  - Tauri desktop app (Bee Keeper)
  - Next.js marketplace (SSG for SEO)
  - Protocol handler (rbee://)
  
  ## Timestamps:
  0:00 - Google search
  0:05 - Click "Run with rbee"
  0:10 - Model spawns on GPU 0
  0:15 - Chat generation
  0:25 - Spawn image model on GPU 1
  0:35 - Image generation
  0:45 - Both models running simultaneously
  
  #AI #MachineLearning #LLM #StableDiffusion #SelfHosted
  ```
  **Verification:** Description is clear and informative

- [ ] **Write Reddit post** (for r/LocalLLaMA, r/StableDiffusion)
  ```markdown
  # I built a system to run AI models in 60 seconds from Google search
  
  **TLDR:** marketplace.rbee.dev → Click "Run with rbee" → Model spawns automatically
  
  ## The Problem:
  Running local AI models is too complex. You need to:
  - Find and download models
  - Install dependencies
  - Configure GPU settings
  - Write inference scripts
  
  ## My Solution: rbee
  1. SEO-optimized marketplace (every model gets a page)
  2. One-click deployment via custom URL protocol (rbee://)
  3. Automatic worker spawning (no config needed)
  4. Multi-GPU support out of the box
  
  ## Demo Video: [link]
  
  Watch me run Llama 3.2 1B + Flux Schnell on two GPUs in <60 seconds.
  
  ## Open Source:
  - GitHub: [link]
  - License: GPL-3.0 (user binaries), MIT (libraries)
  - Docs: [link]
  
  Happy to answer questions! 🐝
  ```
  **Verification:** Post is engaging and informative

- [ ] **Write Twitter thread**
  ```
  🧵 I built a system to run AI models in 60 seconds from Google search.
  
  No Docker. No pip install. No config files.
  
  Just click a button. Watch the demo 👇
  
  [1/6]
  
  ---
  
  The problem: Running local AI is TOO HARD.
  
  Download models ✋
  Install deps ✋  
  Configure GPUs ✋
  Write scripts ✋
  
  It takes HOURS to run your first model.
  
  [2/6]
  
  ---
  
  The solution: rbee
  
  ✅ marketplace.rbee.dev (SEO for every model)
  ✅ Custom URL protocol (rbee://)
  ✅ One-click spawning
  ✅ Multi-GPU out of the box
  
  [3/6]
  
  ---
  
  Watch: I go from Google search to running TWO models on TWO GPUs in 60 seconds.
  
  [VIDEO]
  
  Llama 3.2 1B + Flux Schnell. Zero configuration.
  
  [4/6]
  
  ---
  
  How it works:
  
  1. Next.js marketplace (SSG for SEO)
  2. Tauri desktop app (Bee Keeper)
  3. Protocol handler (rbee://)
  4. Rust backend (queen-rbee, rbee-hive)
  
  [5/6]
  
  ---
  
  Open source. MIT + GPL-3.0.
  
  🌐 rbee.dev
  💻 github.com/[username]/rbee
  📖 docs.rbee.dev
  
  Star the repo if this is cool! 🐝⭐
  
  [6/6]
  ```
  **Verification:** Thread is concise and engaging

---

## Phase 6: Launch Execution (Day 3 Afternoon)

### Platform Publishing

- [ ] **Upload to YouTube**
  - Title: "Run Any AI Model in 60 Seconds | rbee Demo"
  - Description: [from above]
  - Thumbnail: [custom thumbnail]
  - Tags: AI, LLM, Stable Diffusion, Self-Hosted, Open Source
  - Category: Science & Technology
  - Visibility: Public
  **Verification:** Video is live and playable

- [ ] **Post to Reddit**
  - r/LocalLLaMA (high priority)
  - r/StableDiffusion (high priority)
  - r/selfhosted (medium priority)
  - r/programming (medium priority)
  **Timing:** Post during US morning hours (8-10am EST)
  **Verification:** Posts are live, no rule violations

- [ ] **Post to Twitter**
  - Tweet thread (6 tweets)
  - Tag relevant accounts (@huggingface, @karpathy, etc.)
  - Use hashtags: #AI #LLM #OpenSource
  **Verification:** Thread is live

- [ ] **Post to Hacker News**
  - Title: "Show HN: Run AI models in 60 seconds from Google search"
  - URL: YouTube video or rbee.dev
  **Timing:** Post Tuesday-Thursday, 9-11am EST
  **Verification:** Post is live

- [ ] **Post to LinkedIn** (optional)
  - Shorter, more professional version
  - Focus on technical achievement
  - Tag company/tech leaders
  **Verification:** Post is live

### Website Update

- [ ] **Update rbee.dev homepage**
  - Add demo video embed
  - Add "Watch the Demo" CTA
  - Update hero text: "Run AI Models in 60 Seconds"
  **Verification:** Video plays on homepage

- [ ] **Add testimonials section** (optional)
  - Collect early user feedback
  - Add quotes with avatars
  - Link to full reviews
  **Verification:** Section looks good

- [ ] **Add press kit** (optional)
  - High-res logo (SVG + PNG)
  - Screenshots
  - Demo video (MP4 download)
  - Fact sheet (PDF)
  **Location:** rbee.dev/press
  **Verification:** All assets downloadable

---

## Phase 7: Post-Launch Monitoring (Ongoing)

### Metrics Tracking

- [ ] **Set up analytics**
  - Google Analytics on marketplace.rbee.dev
  - YouTube Analytics for video
  - GitHub stars tracking
  **Verification:** Analytics collecting data

- [ ] **Monitor social media**
  - Reddit comments (respond within 1 hour)
  - Twitter mentions (engage with everyone)
  - Hacker News comments (professional responses)
  **Verification:** No questions go unanswered

- [ ] **Track conversion metrics**
  - marketplace.rbee.dev → Install clicks
  - Video views → Website visits
  - GitHub stars → Issues opened
  **Tool:** Google Analytics + GitHub Insights
  **Verification:** Funnel is tracked

### Response Plan

- [ ] **Prepare FAQ responses**
  - "Does it work on Mac/Windows?" → Yes, cross-platform
  - "Is it secure?" → Self-hosted, your data never leaves
  - "How does it compare to Ollama?" → [comparison doc]
  - "Can I use custom models?" → Yes, any GGUF model
  **Verification:** Responses saved for quick copy-paste

- [ ] **Handle bug reports**
  - Triage within 2 hours
  - Fix critical bugs within 24 hours
  - Communicate timeline transparently
  **Tool:** GitHub Issues
  **Verification:** All issues acknowledged

- [ ] **Collect feature requests**
  - Create "Feature Requests" GitHub Discussion
  - Vote on most-requested features
  - Roadmap updated weekly
  **Verification:** Users feel heard

---

## Success Criteria

### Launch Week Goals:

- [ ] Demo video: >1,000 views (YouTube)
- [ ] Website: >500 unique visitors (rbee.dev)
- [ ] GitHub: >100 stars
- [ ] Reddit: >100 upvotes on r/LocalLLaMA
- [ ] Hacker News: Front page (top 10)
- [ ] Twitter: >50 retweets
- [ ] Downloads: >50 Keeper installs

### Quality Benchmarks:

- [ ] Demo video: <60 seconds total
- [ ] Time from Google → Running model: <45 seconds
- [ ] Zero errors during demo
- [ ] Professional audio/video quality
- [ ] Clear call-to-action
- [ ] Positive comment sentiment (>80%)

### User Feedback:

- [ ] "This is so easy!" comments
- [ ] "I got it running in X minutes" stories
- [ ] Feature requests (sign of engagement)
- [ ] Bug reports (sign of real usage)
- [ ] Screenshots/videos from users

---

## Notes

### Key Principles

1. **Show, Don't Tell** - Demo speaks louder than docs
2. **Speed Matters** - Every second counts in demo
3. **Quality > Quantity** - One great video > 10 mediocre posts
4. **Engage Immediately** - Respond to ALL comments in first 24 hours
5. **Be Humble** - Acknowledge limitations, welcome feedback

### Common Pitfalls

- ❌ Demo too long (keep under 60s)
- ❌ Technical jargon (speak plainly)
- ❌ Ignoring comments (kills momentum)
- ❌ Overpromising features (under-promise, over-deliver)
- ✅ Practice demo 10+ times
- ✅ Test on fresh hardware
- ✅ Respond to every comment
- ✅ Be transparent about roadmap

### Marketing Angles (WOW_FACTOR_LAUNCH_MVP.md lines 495-628)

**For r/LocalLLaMA:**
- Emphasize: One-click deployment, multi-GPU, GGUF support
- Avoid: Marketing speak, overhype

**For r/StableDiffusion:**
- Emphasize: Flux support, GPU selection, local inference
- Show: Image generation demo prominently

**For Hacker News:**
- Emphasize: Technical architecture, protocol handler, monorepo
- Discuss: Design decisions, trade-offs
- Be prepared: HN asks hard questions

**For Twitter:**
- Emphasize: Speed, simplicity, wow factor
- Use: Emojis, GIFs, thread format
- Tag: Relevant influencers (but don't spam)

---

## Dependencies

**Must be complete before starting:**
- ✅ CHECKLIST_01 (Shared Components)
- ✅ CHECKLIST_02 (Marketplace SDK)
- ✅ CHECKLIST_03 (Next.js Site)
- ✅ CHECKLIST_04 (Tauri Protocol)
- ✅ CHECKLIST_05 (Keeper UI)

**Blocks:**
- Nothing! This is the final step before public launch 🚀

---

## Post-Launch Roadmap

**Week 1-2 Post-Launch:**
- Monitor metrics daily
- Fix critical bugs immediately
- Respond to all feedback
- Update docs based on questions

**Month 1 Post-Launch:**
- Implement top 3 feature requests
- Write technical blog posts
- Guest post on relevant blogs
- Reach out to podcasts/YouTubers

**Month 2-3 Post-Launch:**
- Premium features (if applicable)
- Enterprise partnerships
- Conference talks
- Version 1.0 release

---

## Celebration! 🎉

**After launch:**
- [ ] Take a break! You earned it.
- [ ] Reflect on what worked well
- [ ] Note what to improve next time
- [ ] Thank early adopters publicly
- [ ] Start planning v1.1

---

**Remember:** The demo is your elevator pitch. Make it count.

**Let's launch!** 🐝🚀
