# Candle Idiom Audit: SD Worker Implementation

**Date:** 2025-11-03  
**Auditor:** TEAM-397/398  
**Reference:** `/reference/candle/candle-examples/examples/stable-diffusion/main.rs`

---

## Executive Summary

**Verdict:** ⚠️ **PARTIALLY IDIOMATIC** - Architecture is correct but implementation details deviate from Candle best practices.

**Score:** 6/10

**Key Issues:**
1. ❌ Custom wrapper structs instead of using Candle types directly
2. ❌ Different text embedding approach
3. ⚠️ Missing guidance scale implementation details
4. ⚠️ Scheduler integration differs
5. ✅ VAE decoding is mostly correct
6. ✅ Overall pipeline structure is sound

---

## Detailed Comparison

### 1. Text Encoding ❌ NOT IDIOMATIC

#### Candle Reference (lines 345-433)
```rust
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    // ...
) -> Result<Tensor> {
    // Load tokenizer from file
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    
    // Get pad_id from vocab
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("
