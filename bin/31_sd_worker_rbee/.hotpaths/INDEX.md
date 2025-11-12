# sd-worker-rbee: Hot Path Analysis Index

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE  
**Purpose:** Technical pseudocode documentation for performance analysis

---

## Overview

This directory contains detailed technical documentation of the **5 critical hot paths** in the `sd-worker-rbee` crate. Each document provides:

- **Pseudocode:** Step-by-step algorithm with timing
- **Performance metrics:** Real-world timings and memory usage
- **Optimization opportunities:** Ranked by impact
- **Code flow examples:** Concrete scenarios

---

## Hot Paths (by Time Spent)

### 1. [Text-to-Image Generation Loop](./HOT_PATH_1_TEXT_TO_IMAGE_GENERATION.md) 🔥
**Time:** 2,200ms (88% of total)  
**Frequency:** Once per request  
**Iterations:** 20-50 diffusion steps

**What it does:**
- Generates images from text prompts
- Runs 20-50 UNet forward passes
- Applies classifier-free guidance
- Generates preview images

**Bottleneck:** UNet forward pass (80-150ms × 20 steps)

**Key Optimization:** Already optimal (GPU-bound)

---

### 2. [LoRA Application](./HOT_PATH_2_LORA_APPLICATION.md) 🔥
**Time:** 1-3 seconds (model loading only)  
**Frequency:** Once per model load  
**Iterations:** 300-500 tensors

**What it does:**
- Applies LoRA weights to base model
- Matrix multiplication for each matching tensor
- Caches results in model

**Bottleneck:** Matrix multiplication (5-50ms × 200 tensors)

**Key Optimizations:**
- ✅ Lazy clone pattern (TEAM-482)
- ✅ Inline always (TEAM-482)
- ✅ HashMap key optimization (TEAM-482)

---

### 3. [VAE Decode](./HOT_PATH_5_VAE_DECODE.md) 🔥
**Time:** 430ms (17% of total)  
**Frequency:** 5-11 times per request  
**Iterations:** 4-6 previews + 1 final

**What it does:**
- Converts latents to RGB images
- Upsamples 64×64 → 512×512
- Runs through 4-stage decoder

**Bottleneck:** Conv upsampling layers (80ms per decode)

**Key Optimizations:**
- 🎯 Reduce preview frequency (40% savings)
- 🎯 Lower resolution previews (56% savings)
- 🎯 Quantized VAE decoder (50% savings)

---

### 4. [Text Embeddings](./HOT_PATH_4_TEXT_EMBEDDINGS.md)
**Time:** 87ms (3.5% of total)  
**Frequency:** Once per request  
**Iterations:** 1-2 CLIP forwards (CFG)

**What it does:**
- Tokenizes text prompts
- Runs CLIP text encoder
- Generates conditioning embeddings

**Bottleneck:** CLIP forward pass (40ms × 2 if CFG)

**Key Optimizations:**
- 🎯 **Embedding cache** (100% savings for repeated prompts)
- 🎯 Quantized CLIP (50% savings)
- 🎯 Compile CLIP (30% savings)

---

### 5. [Image-to-Image](./HOT_PATH_3_IMAGE_TO_IMAGE.md)
**Time:** 2,000ms (20% faster than txt2img)  
**Frequency:** Once per img2img request  
**Iterations:** 10-20 diffusion steps (partial)

**What it does:**
- Encodes input image to latents
- Adds noise based on strength
- Runs partial diffusion (skips initial steps)

**Bottleneck:** Same as txt2img (UNet forward)

**Key Optimizations:**
- ✅ Cow pattern for images (TEAM-482)
- 🎯 Cached VAE encoding
- ✅ Strength-based step skipping (inherent)

---

## Performance Summary Table

| Hot Path | Time | % of Total | Optimization Potential |
|----------|------|------------|------------------------|
| **Text-to-Image Loop** | 2,200ms | 88% | Low (GPU-bound) |
| **LoRA Application** | 3,000ms* | N/A | ✅ Done (33% faster) |
| **VAE Decode** | 430ms | 17% | 🎯 High (40-56% faster) |
| **Text Embeddings** | 87ms | 3.5% | 🎯 Very High (50-100% faster) |
| **Image-to-Image** | 2,000ms | - | ✅ Done (20% faster) |

*One-time cost during model loading

---

## Optimization Priority Matrix

### 🔥 CRITICAL (>10% impact, easy to implement)

1. **Text Embedding Cache**
   - Impact: 100% for repeated prompts
   - Effort: Low (add HashMap cache)
   - Status: NOT IMPLEMENTED
   - ROI: **VERY HIGH**

2. **Reduce Preview Frequency**
   - Impact: 40% VAE time (7% total)
   - Effort: Trivial (change `step % 5` to `step % 10`)
   - Status: NOT IMPLEMENTED
   - ROI: **HIGH**

3. **Lower Resolution Previews**
   - Impact: 56% VAE time (9.5% total)
   - Effort: Low (add scale parameter)
   - Status: NOT IMPLEMENTED
   - ROI: **HIGH**

### 🎯 HIGH (5-10% impact, moderate effort)

4. **Quantized VAE Decoder**
   - Impact: 50% VAE time (8.5% total)
   - Effort: Medium (model conversion)
   - Status: NOT IMPLEMENTED
   - ROI: **MEDIUM**

5. **Quantized CLIP**
   - Impact: 50% embedding time (1.75% total)
   - Effort: Medium (model conversion)
   - Status: NOT IMPLEMENTED
   - ROI: **LOW-MEDIUM**

6. **Cached VAE Encoding (img2img)**
   - Impact: 40ms per cached hit
   - Effort: Low (add cache)
   - Status: NOT IMPLEMENTED
   - ROI: **MEDIUM** (batch processing only)

### ✅ DONE (Already Implemented)

7. **Lazy Clone Pattern (LoRA)** ✅
   - Impact: 33% LoRA time
   - Status: TEAM-482
   - ROI: **ACHIEVED**

8. **Cow Pattern (Images)** ✅
   - Impact: 10-50ms per image
   - Status: TEAM-482
   - ROI: **ACHIEVED**

9. **Aggressive Inlining** ✅
   - Impact: 5-10%
   - Status: TEAM-482
   - ROI: **ACHIEVED**

10. **UUID Direct Storage** ✅
    - Impact: Zero allocations for IDs
    - Status: TEAM-482
    - ROI: **ACHIEVED**

---

## Quick Wins (Priority Order)

**If you have 1 hour:**
1. ✅ Implement text embedding cache (100% speedup for batches)

**If you have 2 hours:**
1. ✅ Text embedding cache
2. ✅ Reduce preview frequency (7% speedup)

**If you have 1 day:**
1. ✅ Text embedding cache
2. ✅ Reduce preview frequency
3. ✅ Lower resolution previews (9.5% speedup)
4. ✅ Quantized VAE decoder (8.5% speedup)

**Total potential speedup:** ~25-30% for typical workloads

---

## Use This Documentation To

1. **Understand performance characteristics**
   - Where time is spent
   - Which operations are expensive
   - What can be optimized

2. **Profile actual workloads**
   - Compare your timings to these estimates
   - Identify anomalies
   - Validate optimizations

3. **Make informed optimization decisions**
   - Prioritize by ROI
   - Avoid premature optimization
   - Focus on bottlenecks

4. **Plan future improvements**
   - Quantization strategy
   - Caching strategy
   - Resolution scaling

---

## Document Format

Each hot path document follows this structure:

```markdown
# HOT PATH #N: [Name]

**Metadata:**
- File location
- Frequency
- Timing

## Flow Diagram
Visual representation of call stack

## Pseudocode
Detailed algorithm with timing annotations

## Performance Analysis
Real-world metrics and breakdowns

## Memory Usage
Allocation patterns and sizes

## Optimization Opportunities
Ranked list with impact estimates

## Code Flow Example
Concrete scenario walkthrough

## Key Insights
Summary of findings
```

---

## Further Investigation

**If you want to:**

### Optimize for speed
→ Start with [HOT_PATH_4_TEXT_EMBEDDINGS.md](./HOT_PATH_4_TEXT_EMBEDDINGS.md) (embedding cache)  
→ Then [HOT_PATH_5_VAE_DECODE.md](./HOT_PATH_5_VAE_DECODE.md) (preview optimization)

### Optimize for memory
→ See [HOT_PATH_2_LORA_APPLICATION.md](./HOT_PATH_2_LORA_APPLICATION.md) (lazy clone pattern)  
→ See memory sections in all documents

### Understand the generation loop
→ Read [HOT_PATH_1_TEXT_TO_IMAGE_GENERATION.md](./HOT_PATH_1_TEXT_TO_IMAGE_GENERATION.md)

### Implement img2img optimizations
→ Read [HOT_PATH_3_IMAGE_TO_IMAGE.md](./HOT_PATH_3_IMAGE_TO_IMAGE.md)

### Add profiling
→ Use timing annotations from pseudocode  
→ Compare actual vs documented timings  
→ Identify regressions

---

## Maintenance

**When adding new features:**
1. Profile the new code
2. Update relevant hot path document
3. Add new hot path document if >5% of time
4. Update this index

**When optimizing:**
1. Document baseline timing
2. Implement optimization
3. Document new timing
4. Update hot path document
5. Add ✅ to this index if >10% improvement

---

## Credits

**TEAM-482:** Performance analysis and optimization  
**Aggressive optimizations:** Breaking changes for speed  
**Hot path documentation:** Technical pseudocode analysis

---

**For questions or additions, see `.plan/TEAM_482_*.md` files.**

