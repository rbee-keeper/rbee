# 🐝 rbee: Consumer Use Case

**Audience:** Homelab users, power users, AI enthusiasts  
**Date:** November 3, 2025  
**Version:** 3.0

---

## The Problem: "I Have Multiple GPUs, Why Can't I Use Them All?"

### Your Setup

You've invested in multiple computers with GPUs:
- **Gaming PC:** RTX 4090 (24GB VRAM) - great for Stable Diffusion  
- **Mac Studio:** M2 Ultra (192GB unified memory) - great for large LLMs  
- **Old Server:** 2x RTX 3090 (24GB each) - sitting mostly idle  
- **Laptop:** CPU only - could run small models

**Total GPU power:** ~72GB VRAM + 192GB unified memory = massive potential

---

### Current Reality: Tool Juggling Hell

**Want to generate an image?**
```bash
cd ~/stable-diffusion-webui
./webui.sh
# Wait for it to load...
# Open browser to http://localhost:7860
# Generate image
# Uses RTX 4090 ✓
```

**Want to chat with an LLM?**
```bash
cd ~/ollama
ollama run llama3
# Uses... which GPU? 
# Probably the Mac M2 Ultra, but who knows\!
# Hope it doesn't conflict with Stable Diffusion
```

**Want to transcribe audio?**
```bash
cd ~/whisper
python transcribe.py audio.mp3
# Separate Python environment
# Different config file  
# Uses... some GPU? Maybe?
```

**Want to use multiple tools at the same time?**
```
❌ Good luck\!
- Tools fight over GPU memory
- Manual process management
- Different APIs for each tool
- Different config files
- Different ports to remember
- Constant context switching
```

---

## The rbee Solution: Turn Your Computers Into One Unified Hive 🐝

### 5-Minute Setup

```bash
# 1. Install rbee (one time)
cargo install rbee-keeper

# 2. Configure your hives (like ~/.ssh/config)
cat > ~/.config/rbee/hives.conf << 'EOF'
Host gaming-pc
  HostName 192.168.1.100
  User vince
  HivePort 7835

Host mac-studio
  HostName 192.168.1.101
  User vince
  HivePort 7835

Host old-server
  HostName 192.168.1.102
  User vince
  HivePort 7835
EOF

# 3. Install hives on each machine (SSH-based, like Ansible)
rbee hive install gaming-pc
rbee hive install mac-studio
rbee hive install old-server

# Done\! ✅ Your colony is ready
```

---

### Now Your Colony Delivers 🐝

```bash
# Start the queen (orchestrator)
rbee queen start

# Chat with LLM (queen routes to Mac M2 Ultra)
curl http://localhost:7833/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-70b",
    "messages": [{"role": "user", "content": "Hello\!"}]
  }'

# Generate image (queen routes to RTX 4090)
curl http://localhost:7833/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl",
    "prompt": "a cat wearing a top hat"
  }'

# Transcribe audio (queen routes to old server RTX 3090)
curl http://localhost:7833/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-large

# ALL AT THE SAME TIME ✅
# No conflicts, no manual switching
# One API, one port (7833)
# The queen orchestrates, worker bees execute
```

**Your colony is now:**
- 🐝 3 hives (3 machines)
- 🐝 5 GPUs (5 potential worker bees)
- 🐝 1 queen (orchestrating everything)
- 🐝 1 API (one endpoint for everything)

---

## Power User Features

### Option 1: GUI (Point and Click)

Open the web dashboard:
```bash
open http://localhost:7833/ui
```

**Visual Colony Management:**
```
┌─────────────────────────────────────────────────┐
│ 🐝 rbee Colony Dashboard                        │
├─────────────────────────────────────────────────┤
│                                                 │
│ Hive: gaming-pc (RTX 4090 - 24GB)             │
│   ├─ Worker 1: SDXL (pinned) ✓                │
│   │  Status: Buzzing (working)                 │
│   │  VRAM: 18GB / 24GB                         │
│   │  Uptime: 2h 34m                            │
│   └─ Worker 2: Available                       │
│                                                 │
│ Hive: mac-studio (M2 Ultra - 192GB)           │
│   ├─ Worker 1: llama-3-70b (pinned) ✓         │
│   │  Status: Buzzing (working)                 │
│   │  Memory: 140GB / 192GB                     │
│   │  Uptime: 5h 12m                            │
│   └─ Worker 2: Available                       │
│                                                 │
│ Hive: old-server (2x RTX 3090 - 48GB total)   │
│   ├─ Worker 1: Available                       │
│   └─ Worker 2: Available                       │
│                                                 │
│ [+ Spawn Worker] [Download Model] [Settings]   │
└─────────────────────────────────────────────────┘
```

**Pin a model to a specific hive:**
1. Click "Spawn Worker" on gaming-pc
2. Select model: SDXL
3. Check "Pin this worker" ✓
4. Click "Spawn"

**Result:** SDXL always runs on RTX 4090, even after restarts.

---

### Option 2: Rhai Script (Programmable)

Create custom routing for your colony:

```bash
# Edit scheduler
nano ~/.config/rbee/scheduler.rhai
```

```rhai
// 🐝 Custom Colony Routing

fn route_task(task, workers) {
    // Images ALWAYS go to RTX 4090 hive
    if task.type == "image-gen" {
        return workers
            .filter(|w| w.hive == "gaming-pc")
            .filter(|w| w.gpu_type == "RTX4090")
            .first();
    }
    
    // Large LLMs (70B+) go to Mac M2 Ultra hive
    if task.type == "text-gen" && task.model.contains("70b") {
        return workers
            .filter(|w| w.hive == "mac-studio")
            .first();
    }
    
    // Small LLMs (8B) can use old server hive
    if task.type == "text-gen" && task.model.contains("8b") {
        return workers
            .filter(|w| w.hive == "old-server")
            .least_loaded();
    }
    
    // Audio transcription uses old server
    if task.type == "audio-transcription" {
        return workers
            .filter(|w| w.hive == "old-server")
            .first();
    }
    
    // Everything else: pick least loaded worker bee
    return workers.least_loaded();
}
```

**Save and reload:**
```bash
rbee scheduler reload
# ✅ New routing rules active (no restart needed\!)
```

**The queen now routes intelligently based on YOUR rules.**

---

## Real-World Examples

### Example 1: AI-Assisted Development 🐝

**Scenario:** You're coding and need AI help from your IDE

```bash
# Configure Zed IDE to use your rbee colony
export OPENAI_API_BASE=http://localhost:7833/v1
export OPENAI_API_KEY=your-rbee-token

# Now Zed's AI assistant uses your Mac M2 Ultra worker bee
# While you generate UI mockups with SDXL on RTX 4090 worker bee
# Both worker bees buzzing at the same time, zero conflicts ✅
```

**What's happening:**
- 🐝 Queen receives request from Zed (text generation)
- 🐝 Queen routes to Mac M2 Ultra hive (best for LLMs)
- 🐝 Worker bee spawns with llama-3-70b
- 🐝 Simultaneously, SDXL worker bee generates images on RTX 4090
- 🐝 Your colony delivers both results

**Result:** Seamless AI-assisted coding + image generation, no tool juggling

---

### Example 2: Content Creation Workflow 🐝

**Scenario:** Creating a video with AI-generated assets

```bash
# Step 1: Generate script with LLM (Mac M2 Ultra worker bee)
curl http://localhost:7833/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-70b",
    "messages": [
      {"role": "system", "content": "You are a video script writer"},
      {"role": "user", "content": "Write a 60-second video script about bees"}
    ]
  }' > script.json

# Step 2: Generate images for video (RTX 4090 worker bees)
for prompt in "bee on flower" "honeycomb close-up" "beekeeper at work"; do
  curl http://localhost:7833/v1/images/generations \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"sdxl\",
      \"prompt\": \"$prompt, cinematic, 4k\",
      \"size\": \"1024x1024\"
    }" > "image_${prompt// /_}.json"
done

# Step 3: Generate voiceover (old server RTX 3090 worker bee)
curl http://localhost:7833/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "<script from step 1>",
    "voice": "alloy"
  }' --output voiceover.mp3

# All running in parallel across your colony ✅
# The queen orchestrates, worker bees execute
```

**What's happening:**
- 🐝 3 different hives working simultaneously
- 🐝 Mac M2 Ultra hive: LLM worker bee generates script
- 🐝 Gaming PC hive: SDXL worker bees generate images (parallel)
- 🐝 Old server hive: TTS worker bee generates voiceover
- 🐝 Queen coordinates everything, no conflicts

**Result:** Complete video assets generated in minutes, using ALL your GPUs

---

### Example 3: Batch Processing 🐝

**Scenario:** Process 100 images overnight without blocking interactive work

```bash
# Create Rhai script for batch vs interactive priority
cat > ~/.config/rbee/scheduler.rhai << 'EOF'
fn route_task(task, workers) {
    // Batch jobs go to old server hive (keep gaming PC free)
    if task.priority == "batch" {
        return workers
            .filter(|w| w.hive == "old-server")
            .least_loaded();
    }
    
    // Interactive jobs use fast gaming PC hive
    if task.priority == "interactive" {
        return workers
            .filter(|w| w.hive == "gaming-pc")
            .first();
    }
    
    // Default: least loaded worker bee
    return workers.least_loaded();
}
EOF

rbee scheduler reload

# Submit batch job (runs on old server worker bees)
for image in images/*.jpg; do
  curl http://localhost:7833/v1/images/variations \
    -F image=@"$image" \
    -F model=sdxl \
    -F priority=batch &
done

# Your gaming PC hive stays free for interactive work ✅
# Play games, generate images interactively, whatever you want
# The old server hive handles batch processing in the background
```

**What's happening:**
- 🐝 Queen receives 100 batch requests
- 🐝 Queen routes ALL to old server hive (2x RTX 3090 worker bees)
- 🐝 Gaming PC hive remains available for interactive work
- 🐝 Worker bees process batch queue overnight
- 🐝 Next morning: 100 processed images ready

**Result:** Maximize GPU utilization without blocking your main machine

---

### Example 4: IDE Integration (Zed, Cursor, Continue.dev) 🐝

**All OpenAI-compatible IDEs work with your rbee colony:**

```bash
# Zed IDE
export OPENAI_API_BASE=http://localhost:7833/v1
export OPENAI_API_KEY=rbee-local-token

# Cursor IDE
# Settings → AI → Custom OpenAI Endpoint
# Set to: http://localhost:7833/v1

# Continue.dev
# config.json:
{
  "models": [
    {
      "title": "rbee Colony",
      "provider": "openai",
      "model": "llama-3-70b",
      "apiBase": "http://localhost:7833/v1",
      "apiKey": "rbee-local-token"
    }
  ]
}
```

**Result:** Your IDE uses your local colony instead of cloud APIs
- ✅ Complete privacy (code never leaves your network)
- ✅ No rate limits
- ✅ No monthly costs
- ✅ Use your own fine-tuned models

---

## Benefits Summary

### Before rbee ❌

- Multiple tools (ComfyUI, Ollama, Whisper)  
- Different APIs, different ports  
- GPU conflicts and memory fights  
- Manual process management  
- Can't use all GPUs simultaneously  
- Complex setup for each tool  
- No unified monitoring

### After rbee ✅

- One unified colony (one API)  
- OpenAI-compatible (works with existing tools)  
- Use ALL GPUs across ALL machines  
- No conflicts, automatic orchestration  
- Pin models to specific hives (GUI or script)  
- 5-minute setup  
- Unified dashboard

**The queen orchestrates. The worker bees execute. Your colony delivers.** 🐝

---

## Cost Analysis

### Cloud API Costs (Without rbee)

**Typical usage:**
- 100K tokens/day for coding (LLM)
- 20 images/day (Stable Diffusion)
- 1 hour audio/day (transcription)

**Monthly cost with OpenAI:**
- GPT-4: $3/1M tokens × 3M tokens = $9
- DALL-E 3: $0.04/image × 600 images = $24
- Whisper: $0.006/minute × 1,800 minutes = $10.80
- **Total: $43.80/month = $525.60/year**

### rbee Costs (Self-Hosted Colony)

**One-time:**
- Setup time: 5 minutes
- Learning curve: 1 hour

**Ongoing:**
- Electricity: ~$10-30/month (GPU power)
- Internet: $0 (local network)
- **Total: ~$120-360/year**

**Savings: $165-405/year**

**Plus:**
- ✅ No rate limits
- ✅ Complete privacy (data never leaves your network)
- ✅ No dependency on external providers
- ✅ Use hardware you already own
- ✅ Your colony, your rules

---

## Getting Started

### Prerequisites

- 2+ computers with GPUs (or 1 computer with multiple GPUs)
- SSH access between machines
- Basic command-line knowledge

### Installation

```bash
# 1. Install rbee
cargo install rbee-keeper

# 2. Configure hives
nano ~/.config/rbee/hives.conf

# 3. Install hives
rbee hive install <hostname>

# 4. Start queen
rbee queen start

# 5. Test your colony
curl http://localhost:7833/v1/models
```

**Full guide:** See main [README.md](../../README.md)

---

## FAQ

**Q: Do I need multiple computers?**  
A: No\! rbee works great with one computer with multiple GPUs. You can even mix GPU + CPU workers.

**Q: What if I only have a Mac?**  
A: Perfect\! rbee supports Metal (Apple Silicon). Use your Mac's unified memory for large models.

**Q: Can I use rbee with Zed/Cursor/Continue.dev?**  
A: Yes\! rbee is OpenAI-compatible. Just point your IDE to `http://localhost:7833/v1`.

**Q: Do I need to learn Rhai scripting?**  
A: No\! The GUI works great for most users. Rhai is for power users who want custom routing.

**Q: What about Windows?**  
A: rbee supports Windows with NVIDIA GPUs (CUDA). Cross-platform support is built-in.

**Q: Is this free?**  
A: Yes\! rbee is GPL-3.0 (binaries) + MIT (infrastructure). Free and open source forever.

---

## Note on Premium

**Premium products are for businesses.** Consumers use the free version (GPL-3.0 + MIT).

If you run a business with GPU infrastructure, see [Business Use Case](03_BUSINESS_USE_CASE.md) and [Premium Products](05_PREMIUM_PRODUCTS.md).

---

## Next Steps

1. **Try rbee:** See [README.md](../../README.md) for installation
2. **Join community:** GitHub Discussions
3. **Read architecture:** [.arch/README.md](../../.arch/README.md)
4. **Compare alternatives:** [Competitive Analysis](04_COMPETITIVE_ANALYSIS.md)

---

**🐝 Stop juggling AI tools. Turn your computers into one unified hive\!**
