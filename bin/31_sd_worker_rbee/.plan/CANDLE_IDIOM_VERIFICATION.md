# Candle Idiom Verification ✅

**Date:** 2025-11-12  
**Reviewer:** TEAM-488  
**Reference:** `/home/vince/Projects/rbee/reference/candle/candle-examples/`

---

## ⚠️ IMPORTANT: FLUX Status

**FLUX support code exists and is Candle-idiomatic, but is DISABLED:**
- Reason: `Box<dyn flux::WithForward>` is not `Send + Sync`
- Blocks: `tokio::spawn_blocking` in generation engine
- Status: Returns error if user attempts to load FLUX models
- Future: Can be re-enabled if Candle adds Send+Sync bounds

**Users will see:**
```
"FLUX models are temporarily unsupported due to threading limitations. 
Use Stable Diffusion models instead."
```

---

## Verification Against Reference Examples

### ✅ **Stable Diffusion: Text Embeddings**

**Reference:** `stable-diffusion/main.rs` lines 345-433  
**Our Code:** `backend/generation.rs` lines 110-190

#### Reference Pattern:
```rust
fn text_embeddings(prompt: &str, uncond_prompt: &str, ...) -> Result<Tensor> {
    let tokenizer = Tokenizer::from_file(tokenizer)?;
    let pad_id = *tokenizer.get_vocab(true).get("
