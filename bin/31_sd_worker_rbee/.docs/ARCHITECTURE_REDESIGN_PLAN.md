# Architecture Redesign: Object-Safe Traits

**Reason:** 20+ model architectures need true polymorphism, not enums

---

## Problem with Current Design

```rust
enum LoadedModel {
    StableDiffusion(...),
    Flux(...),
    // ❌ Can't add 20+ more variants without recompiling
}
```

---

## Solution: Object-Safe Trait with Callback Trait

### 1. Progress Callback Trait
```rust
pub trait ProgressCallback: Send {
    fn on_progress(&mut self, step: usize, total: usize, preview: Option<DynamicImage>);
}

// Blanket impl for closures
impl<F> ProgressCallback for F
where
    F: FnMut(usize, usize, Option<DynamicImage>) + Send,
{
    fn on_progress(&mut self, step: usize, total: usize, preview: Option<DynamicImage>) {
        self(step, total, preview)
    }
}
```

### 2. Object-Safe ImageModel Trait
```rust
pub trait ImageModel: Send + Sync {
    fn model_type(&self) -> &str;
    fn model_variant(&self) -> &str;
    fn capabilities(&self) -> &ModelCapabilities;
    
    // Object-safe! Uses trait object for callback
    fn generate(
        &mut self,
        request: &GenerationRequest,
        progress: &mut dyn ProgressCallback,
    ) -> Result<DynamicImage>;
}
```

### 3. Usage
```rust
// Can use trait objects now!
let model: Box<dyn ImageModel> = model_loader::load_model(...)?;

// Works with closures
let mut callback = |step, total, preview| {
    println!("Step {}/{}", step, total);
};
model.generate(&request, &mut callback)?;
```

---

## Benefits

✅ **Truly extensible** - Add new models without recompiling  
✅ **Plugin system ready** - Can load models dynamically  
✅ **Clean separation** - Each model is self-contained  
✅ **Future-proof** - Works for 100+ model types  

---

## Implementation Plan

1. Create `ProgressCallback` trait
2. Update `ImageModel` trait to use `&mut dyn ProgressCallback`
3. Update `StableDiffusionModel` and `FluxModel` implementations
4. Update `GenerationEngine` to use `Box<dyn ImageModel>`
5. Update `model_loader` to return `Box<dyn ImageModel>`

**Estimated effort:** 30 minutes

---

## Should I implement this redesign?
