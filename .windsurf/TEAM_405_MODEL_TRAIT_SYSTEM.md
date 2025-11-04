# TEAM-405: Model Trait System Design

**Date:** Nov 4, 2025  
**Status:** 🎯 DESIGN PROPOSAL  
**Mission:** Design trait system for different model types (LLM, Image, etc.)

---

## 🚨 THE PROBLEM

**Different model types need DIFFERENT metadata for inference:**

### LLM Models (HuggingFace)
```
CRITICAL for inference:
- tokenizer_config (BOS, EOS, PAD tokens)
- chat_template (how to format messages)
- architectures (which worker can run it)
- model_type (llama, gpt2, mistral, etc.)
- context_length (max tokens)
- quantization (Q4, Q8, FP16, etc.)

NICE to have:
- downloads, likes, tags
- author, license
```

### Image Models (CivitAI)
```
CRITICAL for inference:
- baseModel (SD 1.5, SDXL, etc.)
- vae (which VAE to use)
- clipSkip (CLIP layer to use)
- samplers (compatible samplers)
- resolution (trained resolution)
- trainedWords (trigger words)

NICE to have:
- nsfw, downloads, likes
- images (preview images)
```

**You can't mix these! They're fundamentally different!**

---

## ✅ PROPOSED SOLUTION: Trait System

### Core Concept

```
ModelEntry (base)
    ├── Common fields (id, name, path, size, status)
    └── model_config: Box<dyn ModelConfig>
            ├── LlmConfig (HuggingFace LLMs)
            ├── ImageConfig (CivitAI/SD models)
            ├── AudioConfig (future)
            └── VideoConfig (future)
```

---

## 📐 Trait Design

### 1. Base ModelEntry (Catalog)

```rust
/// Model entry in catalog - MINIMAL, storage-focused
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
pub struct ModelEntry {
    // ========== Core Identity ==========
    pub id: String,
    pub name: String,
    pub author: Option<String>,
    
    // ========== Local Storage ==========
    pub path: PathBuf,
    pub size: u64,
    pub status: ArtifactStatus,
    pub added_at: DateTime<Utc>,
    
    // ========== Model Type ==========
    pub model_type: ModelType,
    
    // ========== Type-Specific Config ==========
    /// Serialized model config (type-specific)
    /// Deserialized based on model_type
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Tsify, PartialEq, Eq)]
pub enum ModelType {
    Llm,
    Image,
    Audio,
    Video,
}
```

### 2. ModelConfig Trait

```rust
/// Trait for model-type-specific configuration
pub trait ModelConfig: Send + Sync {
    /// Get model type
    fn model_type(&self) -> ModelType;
    
    /// Serialize to JSON
    fn to_json(&self) -> serde_json::Value;
    
    /// Check if compatible with worker
    fn is_compatible_with(&self, worker_type: &str) -> bool;
    
    /// Get inference parameters
    fn inference_params(&self) -> InferenceParams;
}

/// Inference parameters (generic)
pub struct InferenceParams {
    pub context_length: Option<u32>,
    pub batch_size: Option<u32>,
    pub additional: HashMap<String, serde_json::Value>,
}
```

### 3. LLM-Specific Config

```rust
/// LLM model configuration (HuggingFace)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    // ========== CRITICAL for inference ==========
    /// Model architecture (e.g., "LlamaForCausalLM")
    pub architecture: String,
    
    /// Model family (e.g., "llama", "gpt2", "mistral")
    pub model_family: String,
    
    /// Tokenizer configuration
    pub tokenizer: TokenizerConfig,
    
    /// Context length (max tokens)
    pub context_length: u32,
    
    /// Quantization type (e.g., "Q4_K_M", "FP16")
    pub quantization: Option<String>,
    
    // ========== Inference settings ==========
    /// Default temperature
    pub default_temperature: Option<f32>,
    
    /// Default top_p
    pub default_top_p: Option<f32>,
    
    /// Supported languages
    pub languages: Vec<String>,
    
    // ========== Metadata ==========
    /// Base model (if fine-tuned)
    pub base_model: Option<String>,
    
    /// License
    pub license: Option<String>,
    
    /// Tags
    pub tags: Vec<String>,
    
    // ========== HuggingFace specific ==========
    /// Files in repo
    pub files: Vec<String>,
    
    /// SHA hash
    pub sha: Option<String>,
    
    /// Is gated
    pub gated: bool,
    
    /// Download/like stats
    pub downloads: u64,
    pub likes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub unk_token: Option<String>,
    pub chat_template: Option<String>,
}

impl ModelConfig for LlmConfig {
    fn model_type(&self) -> ModelType {
        ModelType::Llm
    }
    
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
    
    fn is_compatible_with(&self, worker_type: &str) -> bool {
        match worker_type {
            "llama-cpp" => {
                // Check if GGUF format
                self.quantization.is_some() || 
                self.files.iter().any(|f| f.ends_with(".gguf"))
            }
            "transformers" => {
                // Check if has safetensors
                self.files.iter().any(|f| f.ends_with(".safetensors"))
            }
            _ => false
        }
    }
    
    fn inference_params(&self) -> InferenceParams {
        InferenceParams {
            context_length: Some(self.context_length),
            batch_size: None,
            additional: HashMap::new(),
        }
    }
}
```

### 4. Image-Specific Config

```rust
/// Image model configuration (CivitAI/Stable Diffusion)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    // ========== CRITICAL for inference ==========
    /// Base model (e.g., "SD 1.5", "SDXL", "SD 2.1")
    pub base_model: String,
    
    /// Model type (checkpoint, lora, embedding, etc.)
    pub checkpoint_type: CheckpointType,
    
    /// VAE to use
    pub vae: Option<String>,
    
    /// CLIP skip value
    pub clip_skip: Option<u32>,
    
    /// Trained resolution
    pub resolution: (u32, u32),
    
    /// Trigger words/activation tokens
    pub trigger_words: Vec<String>,
    
    // ========== Inference settings ==========
    /// Recommended samplers
    pub samplers: Vec<String>,
    
    /// Recommended steps
    pub steps: Option<u32>,
    
    /// Recommended CFG scale
    pub cfg_scale: Option<f32>,
    
    // ========== Metadata ==========
    /// NSFW flag
    pub nsfw: bool,
    
    /// License
    pub license: Option<String>,
    
    /// Tags
    pub tags: Vec<String>,
    
    // ========== CivitAI specific ==========
    /// Preview images
    pub preview_images: Vec<ImagePreview>,
    
    /// Download/like stats
    pub downloads: u64,
    pub likes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointType {
    Checkpoint,
    Lora,
    Embedding,
    Hypernetwork,
    VAE,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePreview {
    pub url: String,
    pub width: u32,
    pub height: u32,
    pub nsfw: bool,
}

impl ModelConfig for ImageConfig {
    fn model_type(&self) -> ModelType {
        ModelType::Image
    }
    
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
    
    fn is_compatible_with(&self, worker_type: &str) -> bool {
        match worker_type {
            "stable-diffusion" => true,
            "comfyui" => true,
            _ => false
        }
    }
    
    fn inference_params(&self) -> InferenceParams {
        let mut additional = HashMap::new();
        additional.insert("resolution".to_string(), 
            serde_json::json!([self.resolution.0, self.resolution.1]));
        if let Some(cfg) = self.cfg_scale {
            additional.insert("cfg_scale".to_string(), serde_json::json!(cfg));
        }
        
        InferenceParams {
            context_length: None,
            batch_size: Some(1),
            additional,
        }
    }
}
```

---

## 🎯 Usage Examples

### Creating LLM Model

```rust
// From HuggingFace API
let hf_data: serde_json::Value = /* API response */;

let llm_config = LlmConfig {
    architecture: hf_data["config"]["architectures"][0].as_str().unwrap().to_string(),
    model_family: hf_data["config"]["model_type"].as_str().unwrap().to_string(),
    tokenizer: TokenizerConfig {
        bos_token: hf_data["config"]["tokenizer_config"]["bos_token"]
            .as_str().map(|s| s.to_string()),
        eos_token: hf_data["config"]["tokenizer_config"]["eos_token"]
            .as_str().map(|s| s.to_string()),
        chat_template: hf_data["config"]["tokenizer_config"]["chat_template"]
            .as_str().map(|s| s.to_string()),
        // ...
    },
    context_length: 4096,
    quantization: Some("Q4_K_M".to_string()),
    languages: vec!["en".to_string()],
    // ...
};

let model_entry = ModelEntry {
    id: "meta-llama/Llama-3.2-1B".to_string(),
    name: "Llama-3.2-1B".to_string(),
    path: PathBuf::from("/path/to/model"),
    size: 1024,
    model_type: ModelType::Llm,
    config: llm_config.to_json(),
    // ...
};
```

### Creating Image Model

```rust
// From CivitAI API
let civitai_data: serde_json::Value = /* API response */;

let image_config = ImageConfig {
    base_model: "SD 1.5".to_string(),
    checkpoint_type: CheckpointType::Checkpoint,
    vae: Some("vae-ft-mse-840000-ema-pruned".to_string()),
    clip_skip: Some(2),
    resolution: (512, 512),
    trigger_words: vec!["realistic".to_string(), "photo".to_string()],
    samplers: vec!["DPM++ 2M Karras".to_string()],
    steps: Some(20),
    cfg_scale: Some(7.0),
    nsfw: false,
    // ...
};

let model_entry = ModelEntry {
    id: "civitai:12345".to_string(),
    name: "Realistic Vision V5.1".to_string(),
    path: PathBuf::from("/path/to/model"),
    size: 2048,
    model_type: ModelType::Image,
    config: image_config.to_json(),
    // ...
};
```

### Using Model Config

```rust
// Load from catalog
let model = catalog.get("meta-llama/Llama-3.2-1B")?;

// Deserialize based on type
match model.model_type {
    ModelType::Llm => {
        let llm_config: LlmConfig = serde_json::from_value(model.config)?;
        
        // Use LLM-specific fields
        println!("Architecture: {}", llm_config.architecture);
        println!("Context length: {}", llm_config.context_length);
        println!("Chat template: {:?}", llm_config.tokenizer.chat_template);
        
        // Check compatibility
        if llm_config.is_compatible_with("llama-cpp") {
            println!("Can run with llama.cpp!");
        }
    }
    ModelType::Image => {
        let image_config: ImageConfig = serde_json::from_value(model.config)?;
        
        // Use Image-specific fields
        println!("Base model: {}", image_config.base_model);
        println!("Resolution: {}x{}", image_config.resolution.0, image_config.resolution.1);
        println!("Trigger words: {:?}", image_config.trigger_words);
    }
    _ => {}
}
```

---

## 📦 File Structure

```
artifacts-contract/
├── src/
│   ├── lib.rs
│   ├── status.rs
│   ├── model/
│   │   ├── mod.rs           # ModelEntry, ModelType
│   │   ├── config.rs        # ModelConfig trait
│   │   ├── llm.rs           # LlmConfig
│   │   ├── image.rs         # ImageConfig
│   │   ├── audio.rs         # AudioConfig (future)
│   │   └── video.rs         # VideoConfig (future)
│   └── worker.rs
```

---

## 🎯 Benefits

### 1. **Type Safety**
```rust
// Compile-time guarantee of correct fields
let llm: LlmConfig = serde_json::from_value(model.config)?;
println!("{}", llm.tokenizer.chat_template); // ✅ Type-safe

// Can't access image fields on LLM model
// llm.base_model // ❌ Compile error
```

### 2. **Extensibility**
```rust
// Add new model type without touching existing code
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u32,
    // ...
}

impl ModelConfig for AudioConfig {
    // ...
}
```

### 3. **Clear Separation**
```
LLM concerns:
- Tokenizer, chat templates, context length
- Architecture, quantization
- HuggingFace metadata

Image concerns:
- Base model, VAE, CLIP skip
- Resolution, trigger words
- CivitAI metadata

NO MIXING!
```

### 4. **Inference Ready**
```rust
// Get inference params for any model type
let params = config.inference_params();

match model.model_type {
    ModelType::Llm => {
        // Use context_length for LLM inference
        let context = params.context_length.unwrap();
    }
    ModelType::Image => {
        // Use resolution for image inference
        let resolution = params.additional.get("resolution").unwrap();
    }
}
```

---

## 🚀 Migration Strategy

### Phase 1: Define Traits (Week 1)
- [ ] Create `ModelConfig` trait
- [ ] Implement `LlmConfig`
- [ ] Implement `ImageConfig`
- [ ] Update `ModelEntry` structure

### Phase 2: HuggingFace Integration (Week 2)
- [ ] Update `from_huggingface()` to create `LlmConfig`
- [ ] Extract all critical LLM fields
- [ ] Test with real HuggingFace models
- [ ] Update catalog to use new structure

### Phase 3: CivitAI Integration (Week 3-4)
- [ ] Implement CivitAI client
- [ ] Create `from_civitai()` to create `ImageConfig`
- [ ] Extract all critical image fields
- [ ] Test with real CivitAI models

### Phase 4: Worker Integration (Week 5)
- [ ] Update workers to use `ModelConfig`
- [ ] Implement compatibility checks
- [ ] Test inference with both model types

---

## 📋 Critical Fields Reference

### LLM (MUST HAVE)
```rust
✅ architecture: String           // "LlamaForCausalLM"
✅ model_family: String           // "llama"
✅ tokenizer.bos_token: String    // "<|begin_of_text|>"
✅ tokenizer.eos_token: String    // "<|eot_id|>"
✅ tokenizer.chat_template: String // "{{ ... }}"
✅ context_length: u32            // 4096
✅ quantization: Option<String>   // "Q4_K_M"

📊 Nice to have:
- languages, base_model, license
- downloads, likes, tags
- files, sha, gated
```

### Image (MUST HAVE)
```rust
✅ base_model: String             // "SD 1.5"
✅ checkpoint_type: CheckpointType // Checkpoint
✅ resolution: (u32, u32)         // (512, 512)
✅ trigger_words: Vec<String>     // ["realistic", "photo"]
✅ vae: Option<String>            // "vae-ft-mse-840000"
✅ clip_skip: Option<u32>         // 2

📊 Nice to have:
- samplers, steps, cfg_scale
- nsfw, license, tags
- preview_images, downloads, likes
```

---

## 🎯 Decision Matrix

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Single struct with all fields** | Simple | ❌ Mixing concerns, confusing | ❌ NO |
| **source_metadata HashMap** | Flexible | ❌ No type safety, runtime errors | ❌ NO |
| **Trait system** | ✅ Type-safe, extensible, clear | More code | ✅ **YES** |

---

## 💡 Key Insights

1. **LLM and Image models are FUNDAMENTALLY DIFFERENT**
   - Different inference engines
   - Different critical parameters
   - Different metadata sources

2. **Type safety matters for inference**
   - Wrong tokenizer = broken chat
   - Wrong resolution = bad images
   - Need compile-time guarantees

3. **Extensibility is critical**
   - Audio models coming
   - Video models coming
   - More marketplaces coming

4. **Separation of concerns**
   - Catalog = storage + common fields
   - Config = type-specific inference data
   - Clear boundaries

---

## ✅ Recommendation

**Use the Trait System!**

```rust
pub struct ModelEntry {
    // Common storage fields
    pub id: String,
    pub path: PathBuf,
    pub model_type: ModelType,
    
    // Type-specific config (serialized)
    pub config: serde_json::Value,
}

pub trait ModelConfig {
    fn model_type(&self) -> ModelType;
    fn is_compatible_with(&self, worker: &str) -> bool;
    fn inference_params(&self) -> InferenceParams;
}

pub struct LlmConfig { /* LLM-specific fields */ }
pub struct ImageConfig { /* Image-specific fields */ }
```

**Why?**
- ✅ Type-safe access to critical fields
- ✅ Clear separation of concerns
- ✅ Easy to add new model types
- ✅ Compile-time guarantees
- ✅ Inference-ready

**TEAM-405: Trait system is the way! 🎯**
