# 3-Step Worker Spawning UX

**Date:** 2025-11-04  
**Status:** 🎯 UX DESIGN  
**Purpose:** Simple, intuitive worker spawning flow

---

## 🎯 The Flow

**Step 0:** Open the hive UI (not part of UI, just navigation)  
**Step 1:** Select worker type  
**Step 2:** Select model  
**Step 3:** Select device  
**Result:** Worker spawns! Tab appears!

---

## 📐 Step 0: Navigate to Hive

**Not part of the spawning UI itself**

**User clicks:**
- Sidebar → "Hives" section → Click hive name
- OR: Sidebar → "localhost" (default hive)

**Result:** Hive tab opens showing hive dashboard

---

## 🎨 Step 1: Select Worker Type

### UI Design

```
┌─────────────────────────────────────────────────────────┐
│  Spawn Worker - Step 1 of 3                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  What type of worker do you want to spawn?              │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │  💬 LLM Worker   │  │  🎨 SD Worker    │            │
│  │                  │  │                  │            │
│  │  Chat, text      │  │  Image           │            │
│  │  generation,     │  │  generation      │            │
│  │  code, etc.      │  │                  │            │
│  │                  │  │                  │            │
│  │  [Select]        │  │  [Select]        │            │
│  └──────────────────┘  └──────────────────┘            │
│                                                          │
│  [Cancel]                                                │
└─────────────────────────────────────────────────────────┘
```

### Component

**File:** `src/components/SpawnWorker/Step1_WorkerType.tsx`

```tsx
import { MessageSquare, ImageIcon } from 'lucide-react'

interface Step1Props {
  onSelect: (workerType: 'llm' | 'sd') => void
  onCancel: () => void
}

export function Step1_WorkerType({ onSelect, onCancel }: Step1Props) {
  return (
    <div className="spawn-step step-1">
      <div className="step-header">
        <h2>Spawn Worker - Step 1 of 3</h2>
        <p>What type of worker do you want to spawn?</p>
      </div>
      
      <div className="worker-type-cards">
        <button
          className="worker-type-card"
          onClick={() => onSelect('llm')}
        >
          <div className="card-icon">💬</div>
          <h3>LLM Worker</h3>
          <p>Chat, text generation, code completion, etc.</p>
          <span className="select-btn">Select</span>
        </button>
        
        <button
          className="worker-type-card"
          onClick={() => onSelect('sd')}
        >
          <div className="card-icon">🎨</div>
          <h3>SD Worker</h3>
          <p>Image generation from text prompts</p>
          <span className="select-btn">Select</span>
        </button>
      </div>
      
      <div className="step-actions">
        <button className="btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}
```

---

## 🎨 Step 2: Select Model

### UI Design (LLM Example)

```
┌─────────────────────────────────────────────────────────┐
│  Spawn Worker - Step 2 of 3                             │
├─────────────────────────────────────────────────────────┤
│  💬 LLM Worker                                           │
│                                                          │
│  Select a model:                                         │
│                                                          │
│  🔍 [Search models...]                                  │
│                                                          │
│  ┌─────────────────────────────────────────┐            │
│  │ ✅ Llama-3.2-1B (Q4_K_M)               │  ← Downloaded│
│  │    4.1 GB • HuggingFace                 │            │
│  │    [Select]                             │            │
│  ├─────────────────────────────────────────┤            │
│  │ ✅ Mistral-7B-v0.3 (Q5_K_M)            │            │
│  │    7.2 GB • HuggingFace                 │            │
│  │    [Select]                             │            │
│  ├─────────────────────────────────────────┤            │
│  │ 📦 Llama-2-7B-Chat (Q4_K_M)            │  ← Not downloaded│
│  │    4.3 GB • HuggingFace                 │            │
│  │    [Download first]                     │            │
│  └─────────────────────────────────────────┘            │
│                                                          │
│  Don't see your model?                                   │
│  [Browse HuggingFace Marketplace] [Browse CivitAI]      │
│                                                          │
│  [← Back]                                    [Cancel]    │
└─────────────────────────────────────────────────────────┘
```

### Component

**File:** `src/components/SpawnWorker/Step2_SelectModel.tsx`

```tsx
import { useState } from 'react'
import { Search, Download, Check, ExternalLink } from 'lucide-react'
import { useModels } from '@/hooks/useModels'

interface Step2Props {
  workerType: 'llm' | 'sd'
  onSelect: (modelId: string) => void
  onBack: () => void
  onCancel: () => void
  onBrowseMarketplace: (type: 'huggingface' | 'civitai') => void
}

export function Step2_SelectModel({
  workerType,
  onSelect,
  onBack,
  onCancel,
  onBrowseMarketplace,
}: Step2Props) {
  const [search, setSearch] = useState('')
  const { data: models = [] } = useModels(workerType)
  
  const filteredModels = models.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase())
  )
  
  // Separate downloaded vs not downloaded
  const downloadedModels = filteredModels.filter(m => m.status === 'downloaded')
  const notDownloadedModels = filteredModels.filter(m => m.status !== 'downloaded')
  
  return (
    <div className="spawn-step step-2">
      <div className="step-header">
        <h2>Spawn Worker - Step 2 of 3</h2>
        <p>
          <span className="worker-icon">{workerType === 'llm' ? '💬' : '🎨'}</span>
          {workerType === 'llm' ? 'LLM Worker' : 'SD Worker'}
        </p>
      </div>
      
      <div className="step-content">
        <h3>Select a model:</h3>
        
        {/* Search */}
        <div className="search-bar">
          <Search size={20} />
          <input
            type="text"
            placeholder="Search models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        
        {/* Downloaded Models */}
        {downloadedModels.length > 0 && (
          <div className="model-list">
            <h4>Downloaded Models</h4>
            {downloadedModels.map(model => (
              <button
                key={model.id}
                className="model-card downloaded"
                onClick={() => onSelect(model.id)}
              >
                <div className="model-info">
                  <div className="model-header">
                    <Check size={16} className="downloaded-icon" />
                    <span className="model-name">{model.name}</span>
                  </div>
                  <div className="model-meta">
                    <span>{formatBytes(model.size)}</span>
                    <span>•</span>
                    <span>{model.vendor}</span>
                  </div>
                </div>
                <span className="select-btn">Select</span>
              </button>
            ))}
          </div>
        )}
        
        {/* Not Downloaded Models */}
        {notDownloadedModels.length > 0 && (
          <div className="model-list">
            <h4>Available for Download</h4>
            {notDownloadedModels.map(model => (
              <div key={model.id} className="model-card not-downloaded">
                <div className="model-info">
                  <div className="model-header">
                    <Download size={16} className="download-icon" />
                    <span className="model-name">{model.name}</span>
                  </div>
                  <div className="model-meta">
                    <span>{formatBytes(model.size)}</span>
                    <span>•</span>
                    <span>{model.vendor}</span>
                  </div>
                </div>
                <button className="download-btn" disabled>
                  Download first
                </button>
              </div>
            ))}
          </div>
        )}
        
        {/* Browse Marketplaces */}
        <div className="marketplace-links">
          <p>Don't see your model?</p>
          <div className="link-buttons">
            <button
              className="marketplace-link"
              onClick={() => onBrowseMarketplace('huggingface')}
            >
              <ExternalLink size={16} />
              Browse HuggingFace Marketplace
            </button>
            {workerType === 'sd' && (
              <button
                className="marketplace-link"
                onClick={() => onBrowseMarketplace('civitai')}
              >
                <ExternalLink size={16} />
                Browse CivitAI Marketplace
              </button>
            )}
          </div>
        </div>
      </div>
      
      <div className="step-actions">
        <button className="btn-ghost" onClick={onBack}>
          ← Back
        </button>
        <button className="btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}
```

---

## 🎨 Step 3: Select Device

### UI Design

```
┌─────────────────────────────────────────────────────────┐
│  Spawn Worker - Step 3 of 3                             │
├─────────────────────────────────────────────────────────┤
│  💬 LLM Worker • Llama-3.2-1B                           │
│                                                          │
│  Select a device:                                        │
│                                                          │
│  ┌─────────────────────────────────────────┐            │
│  │ 🖥️  CPU                                 │            │
│  │    11th Gen Intel Core i7               │            │
│  │    Available                            │            │
│  │    [Select]                             │            │
│  ├─────────────────────────────────────────┤            │
│  │ 🎮 GPU 0: NVIDIA RTX 3080              │            │
│  │    10 GB VRAM • 87% free                │            │
│  │    Available                            │            │
│  │    [Select]                             │            │
│  ├─────────────────────────────────────────┤            │
│  │ 🎮 GPU 1: NVIDIA RTX 3070              │            │
│  │    8 GB VRAM • 45% free                 │            │
│  │    ⚠️  Low VRAM (need ~6GB)             │            │
│  │    [Select anyway]                      │            │
│  └─────────────────────────────────────────┘            │
│                                                          │
│  💡 Tip: Choose GPU for faster inference               │
│                                                          │
│  [← Back]                                    [Cancel]    │
└─────────────────────────────────────────────────────────┘
```

### Component

**File:** `src/components/SpawnWorker/Step3_SelectDevice.tsx`

```tsx
import { Cpu, MonitorDown } from 'lucide-react'
import { useDevices } from '@/hooks/useDevices'

interface Step3Props {
  workerType: 'llm' | 'sd'
  modelId: string
  modelName: string
  onSelect: (device: string) => void
  onBack: () => void
  onCancel: () => void
}

export function Step3_SelectDevice({
  workerType,
  modelId,
  modelName,
  onSelect,
  onBack,
  onCancel,
}: Step3Props) {
  const { data: devices = [] } = useDevices()
  
  return (
    <div className="spawn-step step-3">
      <div className="step-header">
        <h2>Spawn Worker - Step 3 of 3</h2>
        <p>
          <span className="worker-icon">{workerType === 'llm' ? '💬' : '🎨'}</span>
          {workerType === 'llm' ? 'LLM Worker' : 'SD Worker'} • {modelName}
        </p>
      </div>
      
      <div className="step-content">
        <h3>Select a device:</h3>
        
        <div className="device-list">
          {devices.map(device => (
            <button
              key={device.id}
              className={`device-card ${device.available ? '' : 'unavailable'}`}
              onClick={() => device.available && onSelect(device.id)}
              disabled={!device.available}
            >
              <div className="device-info">
                <div className="device-header">
                  {device.type === 'cpu' ? (
                    <Cpu className="device-icon" />
                  ) : (
                    <MonitorDown className="device-icon" />
                  )}
                  <span className="device-name">
                    {device.type === 'cpu' ? 'CPU' : `GPU ${device.index}`}
                    {device.type !== 'cpu' && `: ${device.name}`}
                  </span>
                </div>
                
                <div className="device-meta">
                  {device.type === 'cpu' && (
                    <span>{device.model}</span>
                  )}
                  {device.type !== 'cpu' && (
                    <>
                      <span>{device.vram_total_gb} GB VRAM</span>
                      <span>•</span>
                      <span>{device.vram_free_percent}% free</span>
                    </>
                  )}
                </div>
                
                <div className="device-status">
                  {device.available ? (
                    <span className="status-available">Available</span>
                  ) : (
                    <span className="status-warning">
                      ⚠️ {device.warning}
                    </span>
                  )}
                </div>
              </div>
              
              <span className="select-btn">
                {device.available ? 'Select' : 'Select anyway'}
              </span>
            </button>
          ))}
        </div>
        
        <div className="tip">
          💡 Tip: Choose GPU for faster inference
        </div>
      </div>
      
      <div className="step-actions">
        <button className="btn-ghost" onClick={onBack}>
          ← Back
        </button>
        <button className="btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}
```

---

## 🎯 Main Spawn Worker Component

**File:** `src/components/SpawnWorker/SpawnWorkerWizard.tsx`

```tsx
import { useState } from 'react'
import { Step1_WorkerType } from './Step1_WorkerType'
import { Step2_SelectModel } from './Step2_SelectModel'
import { Step3_SelectDevice } from './Step3_SelectDevice'
import { useSpawnWorker } from '@/hooks/useSpawnWorker'
import { useTabStore } from '@/store/tabStore'

export function SpawnWorkerWizard({ hiveId, onClose }: { hiveId: string, onClose: () => void }) {
  const [step, setStep] = useState<1 | 2 | 3>(1)
  const [workerType, setWorkerType] = useState<'llm' | 'sd' | null>(null)
  const [modelId, setModelId] = useState<string | null>(null)
  const [modelName, setModelName] = useState<string | null>(null)
  
  const { mutate: spawnWorker, isLoading } = useSpawnWorker()
  const { addTab } = useTabStore()
  
  const handleWorkerTypeSelect = (type: 'llm' | 'sd') => {
    setWorkerType(type)
    setStep(2)
  }
  
  const handleModelSelect = (id: string, name: string) => {
    setModelId(id)
    setModelName(name)
    setStep(3)
  }
  
  const handleDeviceSelect = (device: string) => {
    // Spawn the worker!
    spawnWorker({
      hiveId,
      workerType: workerType!,
      modelId: modelId!,
      device,
    }, {
      onSuccess: (workerId) => {
        // Close wizard
        onClose()
        
        // Open new tab for worker
        addTab({
          type: 'worker',
          title: `${workerType === 'llm' ? '💬' : '🎨'} ${modelName}`,
          icon: workerType === 'llm' ? '💬' : '🎨',
          route: `/worker/${workerId}`,
          workerId,
        })
      }
    })
  }
  
  const handleBrowseMarketplace = (type: 'huggingface' | 'civitai') => {
    // Close wizard
    onClose()
    
    // Open marketplace tab
    addTab({
      type: 'marketplace',
      title: type === 'huggingface' ? 'HuggingFace' : 'CivitAI',
      icon: type === 'huggingface' ? '🤗' : '🎨',
      route: `/marketplace/${type}`,
      marketplaceType: type,
    })
  }
  
  return (
    <div className="spawn-worker-wizard">
      {step === 1 && (
        <Step1_WorkerType
          onSelect={handleWorkerTypeSelect}
          onCancel={onClose}
        />
      )}
      
      {step === 2 && workerType && (
        <Step2_SelectModel
          workerType={workerType}
          onSelect={(id) => handleModelSelect(id, '...')}  // Get name from model
          onBack={() => setStep(1)}
          onCancel={onClose}
          onBrowseMarketplace={handleBrowseMarketplace}
        />
      )}
      
      {step === 3 && workerType && modelId && modelName && (
        <Step3_SelectDevice
          workerType={workerType}
          modelId={modelId}
          modelName={modelName}
          onSelect={handleDeviceSelect}
          onBack={() => setStep(2)}
          onCancel={onClose}
        />
      )}
      
      {isLoading && (
        <div className="spawning-overlay">
          <div className="spinner" />
          <p>Spawning worker...</p>
        </div>
      )}
    </div>
  )
}
```

---

## 🎯 Integration with Hive Page

**File:** `src/pages/HivePage.tsx`

```tsx
import { useState } from 'react'
import { Plus } from 'lucide-react'
import { SpawnWorkerWizard } from '@/components/SpawnWorker/SpawnWorkerWizard'

export function HivePage({ hiveId }: { hiveId: string }) {
  const [showSpawnWizard, setShowSpawnWizard] = useState(false)
  
  return (
    <div className="hive-page">
      <div className="hive-header">
        <h1>Hive: {hiveId}</h1>
        <button
          className="btn-primary"
          onClick={() => setShowSpawnWizard(true)}
        >
          <Plus size={16} />
          Spawn Worker
        </button>
      </div>
      
      {/* Hive content */}
      <div className="hive-content">
        {/* Workers, models, etc. */}
      </div>
      
      {/* Spawn Worker Wizard (Modal) */}
      {showSpawnWizard && (
        <div className="modal-overlay" onClick={() => setShowSpawnWizard(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <SpawnWorkerWizard
              hiveId={hiveId}
              onClose={() => setShowSpawnWizard(false)}
            />
          </div>
        </div>
      )}
    </div>
  )
}
```

---

## 🚀 User Experience Flow

### Complete Journey

**1. User opens Bee Keeper**
```
Sidebar → Hives → "localhost"
```

**2. Hive page shows "Spawn Worker" button**
```
[+ Spawn Worker]
```

**3. Click "Spawn Worker" → Wizard opens**

**Step 1:** Select type
```
💬 LLM Worker | 🎨 SD Worker
Click: 💬 LLM Worker
```

**Step 2:** Select model
```
✅ Llama-3.2-1B (Q4_K_M) [Select]
Click: [Select]
```

**Step 3:** Select device
```
🎮 GPU 0: NVIDIA RTX 3080 [Select]
Click: [Select]
```

**4. Worker spawns!**
```
- Wizard closes
- Tab appears: "💬 Llama-3.2-1B"
- Split-screen if another worker exists
- Ready to use!
```

---

## ⚡ Performance

**Total time from click to worker ready:**
- Step 1: <1 second (instant UI)
- Step 2: <1 second (models loaded from catalog)
- Step 3: <1 second (devices detected)
- Spawn: 2-5 seconds (worker startup)
- **Total: 5-8 seconds** from first click to ready!

---

## 🎯 Success Criteria

✅ Takes exactly 3 clicks (type → model → device)  
✅ Clear progress indication (Step X of 3)  
✅ Can go back to previous step  
✅ Can cancel at any time  
✅ Shows only compatible models  
✅ Shows device availability  
✅ Links to marketplaces if model not found  
✅ Opens worker tab automatically on success  
✅ Smooth, intuitive UX  

---

**Simple 3-step flow for worker spawning!** 🚀
