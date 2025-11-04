# TEAM-405: Vendor-Agnostic Model Catalog

**Date:** Nov 4, 2025  
**Status:** ✅ COMPLETE  
**Mission:** Make ModelEntry work with ANY marketplace (HuggingFace, CivitAI, Local, etc.)

---

## 🚨 THE PROBLEM

**Original mistake:** I made `ModelEntry` HuggingFace-specific!

```rust
// ❌ WRONG - HuggingFace-specific fields
pub struct ModelEntry {
    pub architectures: Vec<String>,  // HuggingFace only
    pub tokenizer_config: Option<TokenizerConfig>,  // HuggingFace only
    pub siblings: Vec<ModelFile>,  // HuggingFace only
    pub gated: bool,  // HuggingFace only
    // ... etc
}
```

**But we have MULTIPLE marketplaces:**
- **HuggingFace** - LLM models (architectures, tokenizer, gated, etc.)
- **CivitAI** - Image models (nsfw, baseModel, trainedWords, images, etc.)
- **Local** - User's own models
- **Future** - More marketplaces!

Each marketplace has **DIFFERENT** metadata structures!

---

## ✅ THE SOLUTION

### Two Separate Concerns

**1. Marketplace Model** (`marketplace-sdk/types.rs`)
- For **browsing** models in marketplaces
- Minimal, display-only fields
- Already exists and is correct!

```rust
pub struct Model {
    pub id: String,
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub image_url: Option<String>,
    pub tags: Vec<String>,
    pub downloads: f64,
    pub likes: f64,
    pub size: String,
    pub source: ModelSource,  // HuggingFace or CivitAI
}
```

**2. Catalog ModelEntry** (`artifacts-contract/model.rs`)
- For **downloaded** models on disk
- **VENDOR-AGNOSTIC** - works with any source!
- Common fields + `source_metadata` for vendor-specific data

```rust
pub struct ModelEntry {
    // Core Identity
    pub id: String,
    pub name: String,
    pub author: Option<String>,
    pub source: ModelSource,  // HuggingFace, CivitAI, Local, etc.
    
    // Local Storage
    pub path: PathBuf,
    pub size: u64,
    pub status: ArtifactStatus,
    pub added_at: DateTime<Utc>,
    
    // Common Metadata (works for ALL sources)
    pub description: Option<String>,
    pub model_type: Option<String>,
    pub tags: Vec<String>,
    pub license: Option<String>,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<DateTime<Utc>>,
    pub created_at: Option<DateTime<Utc>>,
    
    // Vendor-Specific Metadata
    pub source_metadata: HashMap<String, serde_json::Value>,
}
```

---

## 🎯 Key Design Principle

### Common Fields vs Source Metadata

**Common Fields** - Work across ALL marketplaces:
- `id`, `name`, `author`
- `description`, `model_type`, `tags`
- `license`, `downloads`, `likes`
- `last_modified`, `created_at`

**Source Metadata** - Vendor-specific data:
- **HuggingFace:** architectures, tokenizer_config, siblings, sha, gated, etc.
- **CivitAI:** nsfw, baseModel, trainedWords, images, etc.
- **Local:** Custom user metadata

---

## 📊 Examples

### HuggingFace Model

```json
{
  "id": "meta-llama/Llama-3.2-1B",
  "name": "Llama-3.2-1B",
  "author": "meta-llama",
  "source": "HuggingFace",
  "path": "/home/user/.cache/rbee/models/meta-llama/Llama-3.2-1B",
  "size": 4678484056,
  "status": "Available",
  "addedAt": "2025-11-04T20:19:00Z",
  "description": null,
  "modelType": "text-generation",
  "tags": ["llama", "gguf", "4-bit"],
  "license": "llama3.1",
  "downloads": 151,
  "likes": 0,
  "lastModified": "2025-10-29T00:12:47Z",
  "createdAt": "2025-10-29T00:12:04Z",
  "sourceMetadata": {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "tokenizer_config": {
      "bos_token": "<|begin_of_text|>",
      "eos_token": "<|eot_id|>",
      "chat_template": "..."
    },
    "siblings": [
      { "rfilename": "config.json" },
      { "rfilename": "model.gguf" }
    ],
    "sha": "f46b15b0413595368472a02f7f8aae44713b7767",
    "gated": false,
    "private": false
  }
}
```

### CivitAI Model (Future)

```json
{
  "id": "civitai:12345",
  "name": "Realistic Vision V5.1",
  "author": "SG_161222",
  "source": "CivitAI",
  "path": "/home/user/.cache/rbee/models/civitai/12345",
  "size": 2147483648,
  "status": "Available",
  "addedAt": "2025-11-04T20:19:00Z",
  "description": "Realistic photo generation model",
  "modelType": "checkpoint",
  "tags": ["realistic", "photography", "portrait"],
  "license": "CreativeML Open RAIL-M",
  "downloads": 50000,
  "likes": 1200,
  "lastModified": "2025-10-15T10:30:00Z",
  "createdAt": "2025-09-01T08:00:00Z",
  "sourceMetadata": {
    "nsfw": false,
    "baseModel": "SD 1.5",
    "trainedWords": ["realistic", "photo"],
    "images": [
      {
        "url": "https://image.civitai.com/...",
        "nsfw": false,
        "width": 512,
        "height": 512
      }
    ],
    "modelVersions": [...]
  }
}
```

### Local Model

```json
{
  "id": "local:my-custom-model",
  "name": "My Custom Model",
  "author": "Me",
  "source": "Local",
  "path": "/home/user/models/my-custom-model",
  "size": 1073741824,
  "status": "Available",
  "addedAt": "2025-11-04T20:19:00Z",
  "description": "My fine-tuned model",
  "modelType": "text-generation",
  "tags": ["custom", "fine-tuned"],
  "license": null,
  "downloads": 0,
  "likes": 0,
  "lastModified": null,
  "createdAt": null,
  "sourceMetadata": {
    "training_data": "my_dataset.jsonl",
    "base_model": "llama-2-7b",
    "training_steps": 1000,
    "notes": "Trained on custom dataset"
  }
}
```

---

## 🔧 Usage

### Creating ModelEntry

```rust
// From HuggingFace
let hf_response: serde_json::Value = /* API response */;
let model = ModelEntry::from_huggingface(&hf_response, path);
catalog.add(model)?;

// From CivitAI (future)
let civitai_response: serde_json::Value = /* API response */;
let model = ModelEntry::from_civitai(&civitai_response, path);
catalog.add(model)?;

// Local model
let model = ModelEntry::new(
    "local:my-model".to_string(),
    "My Model".to_string(),
    path,
    size,
);
catalog.add(model)?;
```

### Accessing Metadata

```rust
let model = catalog.get("meta-llama/Llama-3.2-1B")?;

// Common fields (work for all sources)
println!("Name: {}", model.name);
println!("Type: {:?}", model.model_type);
println!("Tags: {:?}", model.tags);

// Vendor-specific fields
match model.source {
    ModelSource::HuggingFace => {
        if let Some(architectures) = model.source_metadata.get("architectures") {
            println!("Architectures: {:?}", architectures);
        }
        if let Some(tokenizer) = model.source_metadata.get("tokenizer_config") {
            println!("Tokenizer: {:?}", tokenizer);
        }
    }
    ModelSource::CivitAI => {
        if let Some(nsfw) = model.source_metadata.get("nsfw") {
            println!("NSFW: {:?}", nsfw);
        }
        if let Some(base_model) = model.source_metadata.get("baseModel") {
            println!("Base Model: {:?}", base_model);
        }
    }
    ModelSource::Local => {
        // Custom metadata
        println!("Custom metadata: {:?}", model.source_metadata);
    }
    _ => {}
}
```

---

## 📋 ModelSource Enum

```rust
pub enum ModelSource {
    /// HuggingFace model hub
    HuggingFace,
    /// CivitAI marketplace
    CivitAI,
    /// Local/custom model
    Local,
    /// Other source
    Other(String),
}
```

---

## 🎯 Benefits

1. **Vendor-Agnostic** - Works with any marketplace
2. **Future-Proof** - Easy to add new marketplaces
3. **No Data Loss** - All vendor-specific data preserved in `source_metadata`
4. **Type-Safe** - Common fields are strongly typed
5. **Flexible** - Can query vendor-specific data when needed
6. **Clean** - No HuggingFace-specific fields polluting the core structure

---

## 🚀 Adding New Marketplace

To add support for a new marketplace (e.g., Ollama):

1. **Add to ModelSource enum:**
```rust
pub enum ModelSource {
    HuggingFace,
    CivitAI,
    Local,
    Ollama,  // NEW
    Other(String),
}
```

2. **Create helper function:**
```rust
impl ModelEntry {
    pub fn from_ollama(ollama_data: &serde_json::Value, path: PathBuf) -> Self {
        // Extract common fields
        let id = ollama_data["name"].as_str().unwrap_or("").to_string();
        let name = id.clone();
        
        // Store ALL Ollama-specific data
        let mut source_metadata = HashMap::new();
        if let Some(obj) = ollama_data.as_object() {
            for (key, value) in obj {
                source_metadata.insert(key.clone(), value.clone());
            }
        }
        
        Self {
            id,
            name,
            source: ModelSource::Ollama,
            // ... common fields ...
            source_metadata,
        }
    }
}
```

3. **Done!** No changes to core structure needed.

---

## 📝 Files Modified

1. `bin/97_contracts/artifacts-contract/src/model.rs`
   - Added `ModelSource` enum
   - Restructured `ModelEntry` to be vendor-agnostic
   - Replaced vendor-specific fields with `source_metadata`
   - Updated `from_huggingface()` to use new structure
   - Removed `TokenizerConfig` and `ModelFile` (now in source_metadata)

---

## ✅ Migration Path

**Old Code (HuggingFace-specific):**
```rust
if model.gated {
    println!("Model is gated");
}
```

**New Code (Vendor-agnostic):**
```rust
if let Some(gated) = model.source_metadata.get("gated") {
    if gated.as_bool().unwrap_or(false) {
        println!("Model is gated");
    }
}
```

**Or create helper methods:**
```rust
impl ModelEntry {
    pub fn is_gated(&self) -> bool {
        match self.source {
            ModelSource::HuggingFace => {
                self.source_metadata.get("gated")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            }
            _ => false
        }
    }
}
```

---

## 🎉 Summary

**Before:**
- ❌ HuggingFace-specific ModelEntry
- ❌ Can't support CivitAI
- ❌ Can't support local models
- ❌ Vendor-specific fields everywhere

**After:**
- ✅ Vendor-agnostic ModelEntry
- ✅ Works with HuggingFace, CivitAI, Local, etc.
- ✅ Common fields for all sources
- ✅ `source_metadata` for vendor-specific data
- ✅ Easy to add new marketplaces
- ✅ No data loss

**TEAM-405: Model catalog is now truly multi-marketplace! 🎉**
