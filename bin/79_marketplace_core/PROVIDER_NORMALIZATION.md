# Provider Normalization - HuggingFace & Civitai Are Equal

**TEAM-460** | Created: Nov 7, 2025

## Problem

The original implementation treated HuggingFace as the "default" and Civitai as an afterthought:
- ❌ Routes: `/models` → HuggingFace only
- ❌ Types: `ModelSource` implied HuggingFace was primary
- ❌ Structure: HuggingFace had special treatment

## Solution: Normalized Provider Architecture

### Core Principle
**All providers are equal.** HuggingFace and Civitai are just different sources for models, with different categories.

### New Type System

#### 1. ModelProvider (replaces ModelSource)
```rust
pub enum ModelProvider {
    HuggingFace,  // Equal
    Civitai,      // Equal
    Local,        // Equal
}
```

**Methods:**
- `display_name()` - "HuggingFace", "Civitai"
- `slug()` - "huggingface", "civitai"
- `from_slug(slug)` - Parse from URL

#### 2. ModelCategory (NEW)
```rust
pub enum ModelCategory {
    Llm,    // Language models (text)
    Image,  // Image generation (Stable Diffusion)
    Audio,  // Audio models
    Video,  // Video models
    Other,  // Unknown
}
```

**Why separate?**
- Provider = WHERE the model comes from
- Category = WHAT the model does
- HuggingFace has LLMs, Civitai has Image models
- Future: HuggingFace could have Image models too

#### 3. Model Type (Updated)
```rust
pub struct Model {
    pub id: String,              // "huggingface-meta-llama-3.1-8b"
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub downloads: f64,
    pub likes: f64,
    pub tags: Vec<String>,
    pub provider: ModelProvider, // WHERE (equal)
    pub category: ModelCategory, // WHAT (equal)
    pub files: Vec<ModelFile>,
}
```

### New URL Structure

#### Before (HuggingFace-centric)
```
/models                    → HuggingFace only
/models/civitai            → Civitai (afterthought)
/models/[slug]             → Assumes HuggingFace
/models/civitai/[slug]     → Civitai special case
```

#### After (Provider-neutral)
```
/models                              → Browse all providers
/models/provider/huggingface         → HuggingFace models
/models/provider/civitai             → Civitai models
/models/provider/huggingface/[slug]  → HuggingFace model detail
/models/provider/civitai/[slug]      → Civitai model detail

/models/category/llm                 → All LLM models (any provider)
/models/category/image               → All Image models (any provider)
```

### New Page Structure

```
app/models/
├── page.tsx                           # Browse all models (grid view)
├── provider/
│   ├── [provider]/
│   │   ├── page.tsx                   # List models by provider
│   │   └── [slug]/
│   │       └── page.tsx               # Model detail
├── category/
│   ├── [category]/
│   │   ├── page.tsx                   # List models by category
│   │   └── [slug]/
│   │       └── page.tsx               # Model detail (redirect to provider)
```

### API Structure (Rust SDK)

#### Before
```rust
// HuggingFace was special
mod huggingface;
mod civitai;  // Added later

pub use huggingface::HuggingFaceClient;
pub use civitai::CivitaiClient;
```

#### After
```rust
// All providers are equal
pub mod providers {
    pub mod huggingface;
    pub mod civitai;
    pub mod local;
}

// Unified interface
pub trait ModelProvider {
    async fn list_models(&self, options: ListOptions) -> Result<Vec<Model>>;
    async fn get_model(&self, id: &str) -> Result<Model>;
    fn provider_type(&self) -> ModelProvider;
    fn supported_categories(&self) -> Vec<ModelCategory>;
}

impl ModelProvider for HuggingFaceClient { ... }
impl ModelProvider for CivitaiClient { ... }
```

### Provider Registry

```rust
pub struct ProviderRegistry {
    providers: HashMap<ModelProvider, Box<dyn ModelProvider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        let mut providers = HashMap::new();
        providers.insert(
            ModelProvider::HuggingFace,
            Box::new(HuggingFaceClient::new())
        );
        providers.insert(
            ModelProvider::Civitai,
            Box::new(CivitaiClient::new())
        );
        Self { providers }
    }

    pub async fn list_all_models(&self) -> Result<Vec<Model>> {
        // Fetch from ALL providers in parallel
        let futures: Vec<_> = self.providers.values()
            .map(|p| p.list_models(Default::default()))
            .collect();
        
        let results = futures::future::join_all(futures).await;
        Ok(results.into_iter().flatten().flatten().collect())
    }

    pub async fn list_by_category(&self, category: ModelCategory) -> Result<Vec<Model>> {
        // Fetch from all providers, filter by category
        let all_models = self.list_all_models().await?;
        Ok(all_models.into_iter()
            .filter(|m| m.category == category)
            .collect())
    }

    pub async fn list_by_provider(&self, provider: ModelProvider) -> Result<Vec<Model>> {
        // Fetch from specific provider
        if let Some(p) = self.providers.get(&provider) {
            p.list_models(Default::default()).await
        } else {
            Ok(vec![])
        }
    }
}
```

### Frontend Components

#### ProviderSelector
```tsx
<ProviderSelector
  providers={[
    { name: 'HuggingFace', slug: 'huggingface', icon: HuggingFaceIcon },
    { name: 'Civitai', slug: 'civitai', icon: CivitaiIcon },
  ]}
  selected={selectedProvider}
  onChange={setSelectedProvider}
/>
```

#### CategorySelector
```tsx
<CategorySelector
  categories={[
    { name: 'Language Models', slug: 'llm', icon: MessageSquare },
    { name: 'Image Generation', slug: 'image', icon: Image },
  ]}
  selected={selectedCategory}
  onChange={setSelectedCategory}
/>
```

#### ModelCard
```tsx
<ModelCard
  model={model}
  showProvider={true}  // Show provider badge
  showCategory={true}  // Show category badge
/>
```

### Migration Path

#### Phase 1: Add New Types (DONE)
- ✅ Add `ModelProvider` enum
- ✅ Add `ModelCategory` enum
- ✅ Update `Model` type
- ✅ Keep `ModelSource` as deprecated alias

#### Phase 2: Update SDK
- [ ] Create `ModelProvider` trait
- [ ] Implement trait for `HuggingFaceClient`
- [ ] Implement trait for `CivitaiClient`
- [ ] Create `ProviderRegistry`

#### Phase 3: Update Frontend
- [ ] Create new page structure
- [ ] Add `ProviderSelector` component
- [ ] Add `CategorySelector` component
- [ ] Update routing

#### Phase 4: Cleanup
- [ ] Remove old routes
- [ ] Remove `ModelSource` alias
- [ ] Update all references

### Benefits

✅ **Equal Treatment** - No provider is "default"
✅ **Extensible** - Easy to add new providers (Replicate, Ollama, etc.)
✅ **Flexible** - Browse by provider OR category
✅ **Clear Separation** - Provider (source) vs Category (type)
✅ **Future-Proof** - HuggingFace could add Image models, Civitai could add LLMs

### Example: Adding a New Provider

```rust
// 1. Create client
pub struct ReplicateClient { ... }

impl ModelProvider for ReplicateClient {
    async fn list_models(&self, options: ListOptions) -> Result<Vec<Model>> {
        // Fetch from Replicate API
        let models = self.fetch_models().await?;
        Ok(models.into_iter().map(|m| Model {
            provider: ModelProvider::Replicate,
            category: self.detect_category(&m),
            ...
        }).collect())
    }
    
    fn provider_type(&self) -> ModelProvider {
        ModelProvider::Replicate
    }
    
    fn supported_categories(&self) -> Vec<ModelCategory> {
        vec![ModelCategory::Llm, ModelCategory::Image]
    }
}

// 2. Register in ProviderRegistry
providers.insert(
    ModelProvider::Replicate,
    Box::new(ReplicateClient::new())
);

// 3. Done! Frontend automatically shows it
```

### Key Insight

**Provider and Category are orthogonal concepts:**

| Provider | Category | Example |
|----------|----------|---------|
| HuggingFace | LLM | Llama 3.1 8B |
| HuggingFace | Image | Stable Diffusion XL (future) |
| Civitai | Image | Realistic Vision v5 |
| Civitai | LLM | (future) |
| Replicate | LLM | Llama 70B |
| Replicate | Image | SDXL Lightning |

This structure supports **any combination** of provider and category.

---

**Status:** 🚧 In Progress  
**Phase:** 1/4 (Types updated)  
**Next:** Implement ModelProvider trait
