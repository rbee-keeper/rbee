# Candle Model Compatibility Analysis

**Date:** November 12, 2025  
**Source:** Verified against `/reference/candle/` source code

---

## Image Generation Models in Candle

### ✅ Supported in Candle (4 architectures)

1. **Stable Diffusion** (`stable_diffusion/`)
   - SD 1.4, 1.5, 2.0, 2.1, XL, Turbo
   - Inpainting variants
   - LCM, Hyper, Lightning (scheduler variants)

2. **Stable Diffusion 3** (`mmdit/` + example)
   - SD 3 Medium
   - Uses MMDiT architecture

3. **FLUX** (`flux/`)
   - FLUX.1-dev
   - FLUX.1-schnell
   - Quantized support

4. **Wuerstchen** (`wuerstchen/`)
   - Wuerstchen v2
   - Prior + decoder architecture

---

## Models from Image NOT in Candle

### ❌ NOT Supported (Need Implementation)

**From the screenshot:**
1. **Aura Flow** - ❌ Not in Candle
2. **Chroma** - ❌ Not in Candle
3. **CogVideoX** - ❌ Not in Candle (video model)
4. **HiDream** - ❌ Not in Candle
5. **Hunyuan 1** - ❌ Not in Candle
6. **Hunyuan Video** - ❌ Not in Candle (video)
7. **Illustrious** - ❌ Not in Candle (likely SD fine-tune)
8. **Kolors** - ❌ Not in Candle
9. **LTXV** - ❌ Not in Candle (video)
10. **Lumina** - ❌ Not in Candle
11. **Mochi** - ❌ Not in Candle (video)
12. **NoobAI** - ❌ Not in Candle (likely SD fine-tune)
13. **Other** - ❌ Unknown
14. **PixArt α** - ❌ Not in Candle
15. **PixArt Σ** - ❌ Not in Candle
16. **Pony** - ❌ Not in Candle (SD fine-tune, should work)
17. **Pony V7** - ❌ Not in Candle (SD fine-tune, should work)
18. **Owen** - ❌ Not in Candle
19. **Wan Video** variants - ❌ Not in Candle (video models)

---

## Actual Compatibility Matrix

### ✅ We Can Support TODAY (with current Candle)

| Model | Architecture | Status | Notes |
|-------|-------------|--------|-------|
| SD 1.4 | SD | ✅ Compatible | Use V1_5 |
| SD 1.5 | SD | ✅ Supported | Explicit |
| SD 1.5 LCM | SD | ✅ Compatible | Scheduler variant |
| SD 1.5 Hyper | SD | ✅ Compatible | Scheduler variant |
| SD 2.0 | SD | ✅ Compatible | Use V2_1 |
| SD 2.1 | SD | ✅ Supported | Explicit |
| SDXL 1.0 | SD | ✅ Supported | Explicit |
| SDXL Lightning | SD | ✅ Compatible | Scheduler variant |
| SDXL Hyper | SD | ✅ Compatible | Scheduler variant |
| FLUX.1 Dev | FLUX | ✅ Supported | Explicit |
| FLUX.1 Schnell | FLUX | ✅ Supported | Explicit |
| FLUX.1 Krea | FLUX | ✅ Compatible | Fine-tune |
| FLUX.1 Kontext | FLUX | ✅ Compatible | Fine-tune |
| Pony | SD | ✅ Compatible | SD fine-tune |
| Pony V7 | SD | ✅ Compatible | SD fine-tune |
| Illustrious | SD | ✅ Compatible | SD fine-tune |
| NoobAI | SD | ✅ Compatible | SD fine-tune |

**Total: ~17 models** (4 architectures)

### ❌ Need Candle Implementation (NOT in Candle)

| Model | Architecture | Status | Effort |
|-------|-------------|--------|--------|
| SD 3 | MMDiT | ⚠️ Partial | Need to integrate |
| Wuerstchen | Wuerstchen | ⚠️ Partial | Need to integrate |
| Aura Flow | DiT | ❌ Missing | High |
| Kolors | ? | ❌ Missing | High |
| PixArt α/Σ | DiT | ❌ Missing | High |
| Hunyuan | ? | ❌ Missing | High |
| Lumina | ? | ❌ Missing | High |
| Video models | Various | ❌ Missing | Very High |

---

## Conclusion

**We can support ~17 models TODAY** with 4 architectures:
1. Stable Diffusion (+ variants)
2. FLUX (+ fine-tunes)
3. SD3 (needs integration)
4. Wuerstchen (needs integration)

**We CANNOT support ~15+ models** without implementing them in Candle first.

---

## Recommendation

**YES, we need object-safe traits!**

Even with just 4 architectures, we need:
- Stable Diffusion
- FLUX
- SD3 (MMDiT)
- Wuerstchen
- Future: PixArt, Kolors, Aura Flow, etc.

The enum approach won't scale. **Implement object-safe traits now.**
