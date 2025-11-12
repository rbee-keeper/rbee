# Model Architecture Verification

**Date:** November 12, 2025  
**Source:** Verified online (HuggingFace, Reddit, GitHub)

---

## Verified Model Architectures

### ✅ Illustrious
- **Architecture:** SDXL (Stable Diffusion XL)
- **Base Model:** SDXL 1.0
- **Type:** Fine-tune/refine
- **Compatibility:** ✅ YES - Works with our SDXL implementation
- **Source:** Multiple sources confirm it's SDXL-based
- **Notes:** Anime-focused, trained on large anime dataset

### ✅ Pony Diffusion V6
- **Architecture:** SDXL (Stable Diffusion XL)
- **Base Model:** SDXL 1.0
- **Type:** Fine-tune
- **Compatibility:** ✅ YES - Works with our SDXL implementation
- **Notes:** Character-focused, widely used

### ❌ Pony Diffusion V7
- **Architecture:** AuraFlow (NEW!)
- **Base Model:** AuraFlow (NOT SDXL!)
- **Type:** Complete rewrite
- **Compatibility:** ❌ NO - Requires AuraFlow implementation
- **Source:** https://purplesmart.ai/pony/content7x
- **Quote:** "At the heart of V7 is the AuraFlow architecture"
- **Notes:** Major architecture change from V6!

### ✅ NoobAI XL
- **Architecture:** SDXL (Stable Diffusion XL)
- **Base Model:** SDXL 1.0 / Illustrious
- **Type:** Fine-tune (V-Pred variant)
- **Compatibility:** ✅ YES - Works with our SDXL implementation
- **Source:** Civitai confirms SDXL-based
- **Notes:** Anime/furry focused, uses V-prediction

### ❌ Kolors
- **Architecture:** Latent Diffusion + ChatGLM-6B
- **Base Model:** Custom (NOT SDXL!)
- **Type:** New architecture
- **Compatibility:** ❌ NO - Needs full implementation
- **Source:** https://github.com/Kwai-Kolors/Kolors
- **Details:**
  - Latent diffusion model (similar to SD architecture)
  - Uses ChatGLM-6B text encoder (NOT CLIP!)
  - Supports 256 token context length
  - Bilingual (Chinese + English)
  - Trained on billions of text-image pairs
- **Notes:** Requires ChatGLM + custom diffusion implementation

---

## Summary

| Model | Architecture | Compatible? | Notes |
|-------|-------------|-------------|-------|
| Illustrious | SDXL | ✅ YES | Direct SDXL fine-tune |
| Pony V6 | SDXL | ✅ YES | Direct SDXL fine-tune |
| **Pony V7** | **AuraFlow** | ❌ NO | **NEW ARCHITECTURE!** |
| NoobAI XL | SDXL | ✅ YES | SDXL fine-tune (V-Pred) |
| Kolors | Latent Diffusion + ChatGLM | ❌ NO | Needs full implementation |

---

## Critical Findings

### 🚨 Pony V7 is NOT SDXL!
**Pony V7 uses AuraFlow architecture**, which is completely different from V6.

This means:
- Pony V6 ✅ works with our SDXL
- Pony V7 ❌ requires AuraFlow implementation

### ❌ Kolors is a Complete New Architecture
Kolors is NOT SDXL-based! It's a custom latent diffusion model with:
- Custom diffusion architecture (similar to SD but different)
- ChatGLM-6B text encoder (NOT CLIP!)
- 256 token context length (vs 77 for CLIP)
- Bilingual support (Chinese + English)

To support Kolors, we need to:
1. Implement ChatGLM-6B text encoder
2. Implement Kolors-specific diffusion model
3. Handle 256-token context length
4. This is a MAJOR implementation effort!

---

## Updated Compatibility Matrix

### ✅ Works TODAY (SDXL-based)
1. SD 1.4, 1.5, 2.0, 2.1
2. SDXL 1.0, Lightning, Hyper, Turbo
3. **Illustrious** ✅
4. **Pony V6** ✅
5. **NoobAI XL** ✅
6. FLUX Dev, Schnell, Krea, Kontext

**Total: ~20 models**

### ❌ Needs Implementation
1. **Pony V7** - Requires AuraFlow
2. **Kolors** - Requires ChatGLM text encoder
3. SD3 - Requires MMDiT integration
4. Wuerstchen - Requires integration
5. PixArt α/Σ - Requires DiT
6. Aura Flow - Requires implementation
7. Video models - Different domain

---

## Recommendation

**Object-safe traits are MANDATORY!**

We need to support:
1. Stable Diffusion (U-Net + CLIP)
2. FLUX (DiT + T5 + CLIP)
3. SD3 (MMDiT + T5 + CLIP)
4. Kolors (U-Net + ChatGLM)
5. Pony V7 (AuraFlow)
6. Future: PixArt, Aura Flow, etc.

**At least 5-6 different architectures!**

The enum approach is dead. **Implement object-safe traits NOW.**
