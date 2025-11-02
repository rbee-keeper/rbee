# rbee: Consumer Use Case

**Audience:** Homelab users, power users, AI enthusiasts  
**Date:** November 2, 2025

---

## The Problem: "I Have Multiple GPUs, Why Can't I Use Them All?"

### Your Setup

You've invested in multiple computers with GPUs:
- **Gaming PC:** RTX 4090 (24GB VRAM) - great for Stable Diffusion
- **Mac Studio:** M2 Ultra (192GB unified memory) - great for large LLMs
- **Old Server:** 2x RTX 3090 (24GB each) - sitting mostly idle
- **Laptop:** CPU only - could run small models

**Total GPU power:** ~72GB VRAM + 192GB unified memory = massive potential

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
# Probably the Mac M2 Ultra, but who knows!
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
❌ Good luck!
- Tools fight over GPU memory
- Manual process management
- Different APIs for each tool
- Different config files
- Different ports to remember
- Constant context switching
```

---

## The rbee Solution: One API for Everything

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

# Done! ✅
```

### Now Use Everything Through One API

```bash
# Start the queen (orchestrator)
rbee queen start

# Chat with LLM (automatically uses Mac M2 Ultra)
curl http://localhost:7833/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-70b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Generate image (automatically uses RTX 4090)
curl http://localhost:7833/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl",
    "prompt": "a cat wearing a top hat"
  }'

# Transcribe audio (uses old server RTX 3090)
curl http://localhost:7833/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-large

# ALL AT THE SAME TIME ✅
# No conflicts, no manual switching
# One API, one port (7833)
```

---

## Power User Features

### Option 1: GUI (Point and Click)

Open the web dashboard:
```bash
open http://localhost:7833/ui
```

**Visual GPU Management:**
```
┌─────────────────────────────────────────────────┐
│ rbee Dashboard                                  │
├─────────────────────────────────────────────────┤
│                                                 │
│ Hive: gaming-pc (RTX 4090 - 24GB)             │
│   ├─ Worker 1: SDXL (pinned) ✓                │
│   │  Status: Running                           │
│   │  VRAM: 18GB / 24GB                         │
│   │  Uptime: 2h 34m                            │
│   └─ Worker 2: Available                       │
│                                                 │
│ Hive: mac-studio (M2 Ultra - 192GB)           │
│   ├─ Worker 1: llama-3-70b (pinned) ✓         │
│   │  Status: Running                           │
│   │  Memory: 140GB / 192GB                     │
│   │  Uptime: 5h 12m                            │
│   └─ Worker 2: Available                       │
│                                                 │
│ Hive: old-server (2x RTX 3090 - 48GB total)   │
│   ├─ GPU 0: Available                          │
│   └─ GPU 1: Available                          │
│                                                 │
│ [+ Spawn Worker] [Download Model] [Settings]   │
└─────────────────────────────────────────────────┘
```

**Pin a model to a specific GPU:**
1. Click "Spawn Worker" on gaming-pc
2. Select model: SDXL
3. Check "Pin this worker" ✓
4. Click "Spawn"

**Result:** SDXL always runs on RTX 4090, even after restarts.

---

### Option 2: Rhai Script (Programmable)

Create a custom routing script:

```bash
# Edit scheduler
nano ~/.config/rbee/scheduler.rhai
```

```rhai
// Custom routing logic
fn route_task(task, workers) {
    // Images ALWAYS go to RTX 4090
    if task.type == "image-gen" {
        return workers
            .filter(|w| w.hive == "gaming-pc")
            .filter(|w| w.gpu_type == "RTX4090")
            .first();
    }
    
    // Large LLMs (70B+) go to Mac M2 Ultra
    if task.type == "text-gen" && task.model.contains("70b") {
        return workers
            .filter(|w| w.hive == "mac-studio")
            .first();
    }
    
    // Small LLMs (8B) can use old server
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
    
    // Everything else: pick least loaded GPU
    return workers.least_loaded();
}

// Admission control (optional)
fn should_admit(task, queue_depth) {
    // Reject if queue too deep
    if queue_depth > 10 {
        return reject("Queue full, try again later");
    }
    
    return admit();
}
```

**Save and reload:**
```bash
rbee scheduler reload
# ✅ New routing rules active (no restart needed!)
```

---

## Real-World Examples

### Example 1: AI-Assisted Development

**Scenario:** You're coding and need AI help

```bash
# Configure Zed IDE to use rbee
export OPENAI_API_BASE=http://localhost:7833/v1
export OPENAI_API_KEY=your-rbee-token

# Now Zed's AI assistant uses your Mac M2 Ultra
# While you generate UI mockups with SDXL on RTX 4090
# Both at the same time, zero conflicts
```

### Example 2: Content Creation Workflow

**Scenario:** Creating a video with AI-generated assets

```bash
# 1. Generate script with LLM (Mac M2 Ultra)
curl http://localhost:7833/v1/chat/completions \
  -d '{"model": "llama-3-70b", "messages": [...]}'

# 2. Generate images for video (RTX 4090)
for prompt in "${prompts[@]}"; do
  curl http://localhost:7833/v1/images/generations \
    -d "{\"model\": \"sdxl\", \"prompt\": \"$prompt\"}"
done

# 3. Generate voiceover (old server RTX 3090)
curl http://localhost:7833/v1/audio/speech \
  -d '{"model": "tts-1", "input": "..."}'

# All running in parallel across your GPU farm ✅
```

### Example 3: Batch Processing

**Scenario:** Process 100 images overnight

```bash
# Rhai script for batch jobs
fn route_task(task, workers) {
    if task.priority == "batch" {
        // Use old server for batch jobs
        return workers.filter(|w| w.hive == "old-server").first();
    }
    // Interactive jobs use fast GPUs
    return workers.filter(|w| w.hive == "gaming-pc").first();
}

# Submit batch job
for image in images/*.jpg; do
  curl http://localhost:7833/v1/images/variations \
    -F image=@"$image" \
    -F priority=batch
done

# Your gaming PC stays free for interactive work ✅
```

---

## Benefits Summary

### Before rbee

❌ Multiple tools (ComfyUI, Ollama, Whisper)  
❌ Different APIs, different ports  
❌ GPU conflicts and memory fights  
❌ Manual process management  
❌ Can't use all GPUs simultaneously  
❌ Complex setup for each tool  
❌ No unified monitoring

### After rbee

✅ One API for everything (port 7833)  
✅ OpenAI-compatible (works with existing tools)  
✅ Use ALL GPUs across ALL machines  
✅ No conflicts, automatic orchestration  
✅ Pin models to specific GPUs (GUI or script)  
✅ 5-minute setup  
✅ Unified dashboard

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

### rbee Costs (Self-Hosted)

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

# 5. Test it
curl http://localhost:7833/v1/models
```

**Full guide:** See main [README.md](../../README.md)

---

## FAQ

**Q: Do I need multiple computers?**  
A: No! rbee works great with one computer with multiple GPUs. You can even mix GPU + CPU workers.

**Q: What if I only have a Mac?**  
A: Perfect! rbee supports Metal (Apple Silicon). Use your Mac's unified memory for large models.

**Q: Can I use rbee with Zed/Cursor/Continue.dev?**  
A: Yes! rbee is OpenAI-compatible. Just point your IDE to `http://localhost:7833/v1`.

**Q: Do I need to learn Rhai scripting?**  
A: No! The GUI works great for most users. Rhai is for power users who want custom routing.

**Q: What about Windows?**  
A: rbee supports Windows with NVIDIA GPUs (CUDA). Cross-platform support is built-in.

**Q: Is this free?**  
A: Yes! rbee is GPL-3.0 licensed. Free and open source forever.

---

## Next Steps

1. **Try rbee:** See [README.md](../../README.md) for installation
2. **Join community:** GitHub Discussions
3. **Read architecture:** [.arch/README.md](../../.arch/README.md)
4. **Compare alternatives:** [Technical Differentiators](04_TECHNICAL_DIFFERENTIATORS.md)

---

**Stop juggling AI tools. Use rbee.** 🐝
