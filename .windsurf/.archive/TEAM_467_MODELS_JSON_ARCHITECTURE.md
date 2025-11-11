# TEAM-467: Models.json Architecture Plan

**Problem**: Currently mixing CivitAI and HuggingFace models in ONE `models.json`  
**Issue**: Different sources have different schemas, different use cases, different update frequencies

---

## 🚨 Why One models.json is BAD

### Different Schemas
```typescript
// HuggingFace Model
{
  id: "meta-llama/Llama-3.2-1B",
  downloads: 138081386,
  likes: 4095,
  tags: ["transformers", "pytorch"],
  siblings: [{ filename: "model.safetensors", size: 1234 }]
}

// CivitAI Model  
{
  id: "66",
  downloads: 102155,
  likes: 0,
  imageUrl: "https://...",
  nsfw: "PG",
  modelVersions: [...]
}

// GitHub Model (future)
{
  id: "owner/repo",
  stars: 1234,
  forks: 567,
  language: "Python"
}

// Docker Image (future)
{
  id: "nginx:latest",
  pulls: 1000000000,
  size: "142MB",
  layers: 5
}
```

### Different Update Frequencies
- **HuggingFace**: Changes frequently (new models daily)
- **CivitAI**: Changes frequently (new models hourly)
- **GitHub**: Rarely changes (code repos)
- **Docker**: Very frequently (new tags constantly)

### Different Use Cases
- **HuggingFace**: LLM inference, embeddings
- **CivitAI**: Image generation, LoRAs
- **GitHub**: Code repositories, tools
- **Docker**: Container images, services

---

## ✅ Proposed Architecture

### Separate JSON files per source
```
public/manifests/
├── huggingface-models.json    # All HF models
├── civitai-models.json         # All CivitAI models
├── github-models.json          # All GitHub repos (future)
├── docker-images.json          # All Docker images (future)
│
├── hf-filter-small.json        # References huggingface-models.json
├── hf-filter-medium.json
├── civitai-filter-pg.json      # References civitai-models.json
└── ...
```

### Filter Manifest Format
```json
{
  "filter": "hf-filter/small",
  "source": "huggingface",           // NEW: Explicit source
  "modelsFile": "huggingface-models.json",  // NEW: Which file to load
  "modelIds": ["meta-llama/...", ...],
  "timestamp": "2025-11-11T01:12:14.402Z"
}
```

### Loading Flow
```typescript
// 1. Load filter manifest
const filter = await fetch('/manifests/hf-filter-small.json')

// 2. Load the CORRECT models file based on source
const modelsDb = await fetch(`/manifests/${filter.modelsFile}`)

// 3. Resolve IDs
const models = filter.modelIds.map(id => modelsDb.models[id])
```

---

## 📊 Benefits

### 1. **Separation of Concerns**
- Each source has its own schema
- Can update HF models without touching CivitAI
- Can add new sources without breaking existing ones

### 2. **Better Caching**
- HuggingFace models cached separately
- CivitAI models cached separately
- Only invalidate cache for changed source

### 3. **Smaller Files**
- Don't load CivitAI models when browsing HuggingFace
- Don't load HuggingFace models when browsing CivitAI

### 4. **Type Safety**
```typescript
interface HuggingFaceModel {
  id: string
  downloads: number
  likes: number
  tags: string[]
  // HF-specific fields
}

interface CivitAIModel {
  id: string
  downloads: number
  likes: number
  imageUrl?: string
  nsfw: string
  // CivitAI-specific fields
}

// NOT: type Model = HuggingFaceModel | CivitAIModel (messy union)
```

### 5. **Future-Proof**
Easy to add:
- `github-models.json` for code repos
- `docker-images.json` for containers
- `npm-packages.json` for packages
- `pypi-packages.json` for Python packages

---

## 🔧 Implementation Plan

### Step 1: Split models.json
```typescript
// In generate-model-manifests.ts

// Separate maps for each source
const hfModels = new Map<string, HFModelMetadata>()
const civitaiModels = new Map<string, CivitAIModelMetadata>()

// Save separate files
await fs.writeFile('huggingface-models.json', JSON.stringify({
  totalModels: hfModels.size,
  models: Object.fromEntries(hfModels),
  timestamp: new Date().toISOString()
}))

await fs.writeFile('civitai-models.json', JSON.stringify({
  totalModels: civitaiModels.size,
  models: Object.fromEntries(civitaiModels),
  timestamp: new Date().toISOString()
}))
```

### Step 2: Update Filter Manifests
```typescript
// Add source metadata
const manifest: FilterManifest = {
  filter: 'hf-filter/small',
  source: 'huggingface',
  modelsFile: 'huggingface-models.json',
  modelIds: [...],
  timestamp: new Date().toISOString()
}
```

### Step 3: Update Client Loader
```typescript
// Cache per source
const modelsCache = new Map<string, ModelsDatabase>()

async function loadModelsDatabase(source: string): Promise<ModelsDatabase> {
  const filename = `${source}-models.json`
  if (modelsCache.has(filename)) {
    return modelsCache.get(filename)!
  }
  
  const db = await fetch(`/manifests/${filename}`).then(r => r.json())
  modelsCache.set(filename, db)
  return db
}
```

### Step 4: Backward Compatibility
- Keep `models.json` for now (deprecated)
- Add migration warning
- Remove in next major version

---

## 📁 File Structure

### Before (BAD)
```
models.json (1.4MB)
  ├── HuggingFace models (882)
  └── CivitAI models (200)
```

### After (GOOD)
```
huggingface-models.json (~1.2MB)
  └── HuggingFace models (882)

civitai-models.json (~200KB)
  └── CivitAI models (200)

github-models.json (future)
  └── GitHub repos

docker-images.json (future)
  └── Docker images
```

---

## 🎯 Migration Steps

1. ✅ **Split generation** - Generate separate JSON files
2. ✅ **Update filter manifests** - Add source metadata
3. ✅ **Update client loader** - Load correct file based on source
4. ✅ **Test** - Verify both sources work
5. ✅ **Deprecate models.json** - Add warning
6. ✅ **Remove models.json** - Clean up

---

## 🚀 Future Sources

### GitHub Repositories
```json
{
  "totalModels": 1000,
  "models": {
    "facebook/react": {
      "id": "facebook/react",
      "stars": 220000,
      "forks": 45000,
      "language": "JavaScript",
      "description": "A declarative, efficient, and flexible JavaScript library"
    }
  }
}
```

### Docker Images
```json
{
  "totalImages": 5000,
  "images": {
    "nginx:latest": {
      "id": "nginx:latest",
      "pulls": 1000000000,
      "size": "142MB",
      "layers": 5,
      "digest": "sha256:..."
    }
  }
}
```

---

## ✅ Decision

**SPLIT INTO SEPARATE FILES** - One per source type

**Rationale**:
- Different schemas
- Different update frequencies
- Better caching
- Type safety
- Future-proof

**Action**: Implement the split architecture NOW before it becomes harder to change.

---

**TEAM-467: This is the correct architecture. Let's implement it.**
