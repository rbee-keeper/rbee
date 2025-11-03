# TEAM-399: Phase 9 - UI Development Part 1

**Team:** TEAM-399  
**Phase:** 9 - UI Foundation  
**Duration:** 45 hours  
**Dependencies:** TEAM-397 (working backend)  
**Parallel Work:** ✅ Can work parallel to TEAM-398 (testing)

---

## 🎯 Mission

Build the UI foundation: WASM SDK, React hooks, and basic text-to-image interface. Create reusable components for generation workflows.

---

## 📦 What You're Building

### Directory Structure

```
ui/
├── packages/
│   ├── sd-worker-sdk/        ← WASM SDK (Rust → JS)
│   │   ├── src/
│   │   │   └── lib.rs
│   │   ├── Cargo.toml
│   │   └── package.json
│   │
│   └── sd-worker-react/      ← React hooks
│       ├── src/
│       │   ├── hooks/
│       │   └── index.ts
│       ├── package.json
│       └── tsconfig.json
│
└── app/                      ← Main application
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   └── App.tsx
    ├── package.json
    └── vite.config.ts
```

---

## 📋 Task Breakdown

### Week 1: WASM SDK (16 hours)

**Day 1: SDK Setup (8 hours)**
- [ ] Create `ui/packages/sd-worker-sdk/` structure (1 hour)
- [ ] Configure wasm-bindgen (1 hour)
- [ ] Configure wasm-pack (1 hour)
- [ ] Create Cargo.toml with wasm target (1 hour)
- [ ] Create package.json (1 hour)
- [ ] Test WASM build (3 hours)

**Day 2: SDK Implementation (8 hours)**
- [ ] Implement job submission API (2 hours)
- [ ] Implement SSE streaming API (3 hours)
- [ ] Add TypeScript types (2 hours)
- [ ] Test SDK in browser (1 hour)

---

### Week 2: React Hooks (16 hours)

**Day 3: Hook Setup (8 hours)**
- [ ] Create `ui/packages/sd-worker-react/` structure (1 hour)
- [ ] Configure TypeScript (1 hour)
- [ ] Configure build (1 hour)
- [ ] Create useTextToImage hook (3 hours)
- [ ] Create useGenerationProgress hook (2 hours)

**Day 4: Hook Features (8 hours)**
- [ ] Add loading states (2 hours)
- [ ] Add error handling (2 hours)
- [ ] Add cancellation support (2 hours)
- [ ] Write hook tests (2 hours)

---

### Week 3: Main Application (13 hours)

**Day 5: App Setup (5 hours)**
- [ ] Create `ui/app/` with Vite (1 hour)
- [ ] Configure TailwindCSS (1 hour)
- [ ] Create basic layout (1 hour)
- [ ] Wire up SDK and hooks (2 hours)

**Day 6: Text-to-Image UI (8 hours)**
- [ ] Create prompt input component (2 hours)
- [ ] Create parameter controls (2 hours)
- [ ] Create image display component (2 hours)
- [ ] Create progress indicator (2 hours)

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] WASM SDK compiles and runs in browser
- [ ] Job submission works from JavaScript
- [ ] SSE streaming works in browser
- [ ] React hooks provide clean API
- [ ] Text-to-image UI functional
- [ ] Progress bar shows real-time updates
- [ ] Generated image displays
- [ ] Error states handled gracefully
- [ ] TypeScript types complete
- [ ] All tests passing
- [ ] Clean build (0 warnings)

---

## 🔧 Implementation Notes

### WASM SDK (Rust)

```rust
// ui/packages/sd-worker-sdk/src/lib.rs
use wasm_bindgen::prelude::*;
use web_sys::{EventSource, MessageEvent};

#[wasm_bindgen]
pub struct SDWorkerClient {
    base_url: String,
}

#[wasm_bindgen]
impl SDWorkerClient {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }
    
    #[wasm_bindgen]
    pub async fn submit_job(&self, request: JsValue) -> Result<String, JsValue> {
        let window = web_sys::window().unwrap();
        let fetch = window.fetch_with_str(&format!("{}/v1/jobs", self.base_url));
        
        // ... fetch implementation
        
        Ok(job_id)
    }
    
    #[wasm_bindgen]
    pub fn stream_job(&self, job_id: String, callback: js_sys::Function) -> Result<(), JsValue> {
        let url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
        let event_source = EventSource::new(&url)?;
        
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Some(data) = event.data().as_string() {
                callback.call1(&JsValue::NULL, &JsValue::from_str(&data)).ok();
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        
        event_source.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();
        
        Ok(())
    }
}
```

### React Hook

```typescript
// ui/packages/sd-worker-react/src/hooks/useTextToImage.ts
import { useState, useCallback } from 'react';
import { SDWorkerClient } from 'sd-worker-sdk';

export interface TextToImageParams {
  prompt: string;
  negativePrompt?: string;
  steps?: number;
  guidanceScale?: number;
  seed?: number;
  width?: number;
  height?: number;
}

export interface GenerationProgress {
  step: number;
  total: number;
}

export function useTextToImage(baseUrl: string) {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const generate = useCallback(async (params: TextToImageParams) => {
    setLoading(true);
    setProgress(null);
    setImage(null);
    setError(null);
    
    try {
      const client = new SDWorkerClient(baseUrl);
      
      // Submit job
      const jobId = await client.submit_job({
        prompt: params.prompt,
        negative_prompt: params.negativePrompt,
        steps: params.steps ?? 20,
        guidance_scale: params.guidanceScale ?? 7.5,
        seed: params.seed,
        width: params.width ?? 512,
        height: params.height ?? 512,
      });
      
      // Stream progress
      client.stream_job(jobId, (data: string) => {
        const event = JSON.parse(data);
        
        if (event.event === 'progress') {
          setProgress({ step: event.step, total: event.total });
        } else if (event.event === 'complete') {
          setImage(`data:image/png;base64,${event.image_base64}`);
          setLoading(false);
        } else if (event.event === 'error') {
          setError(event.message);
          setLoading(false);
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoading(false);
    }
  }, [baseUrl]);
  
  return { generate, loading, progress, image, error };
}
```

### React Component

```typescript
// ui/app/src/components/TextToImageGenerator.tsx
import React, { useState } from 'react';
import { useTextToImage } from 'sd-worker-react';

export function TextToImageGenerator() {
  const [prompt, setPrompt] = useState('');
  const [steps, setSteps] = useState(20);
  const { generate, loading, progress, image, error } = useTextToImage('http://localhost:8600');
  
  const handleGenerate = () => {
    generate({ prompt, steps });
  };
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Text to Image</h1>
      
      {/* Prompt Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full p-3 border rounded-lg"
          rows={3}
          placeholder="a photo of a cat"
        />
      </div>
      
      {/* Steps Control */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">
          Steps: {steps}
        </label>
        <input
          type="range"
          min="10"
          max="50"
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          className="w-full"
        />
      </div>
      
      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={loading || !prompt}
        className="w-full bg-blue-600 text-white py-3 rounded-lg disabled:opacity-50"
      >
        {loading ? 'Generating...' : 'Generate'}
      </button>
      
      {/* Progress */}
      {progress && (
        <div className="mt-4">
          <div className="flex justify-between text-sm mb-2">
            <span>Generating...</span>
            <span>{progress.step} / {progress.total}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${(progress.step / progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}
      
      {/* Error */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-lg">
          {error}
        </div>
      )}
      
      {/* Image */}
      {image && (
        <div className="mt-6">
          <img src={image} alt="Generated" className="w-full rounded-lg shadow-lg" />
        </div>
      )}
    </div>
  );
}
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **LLM Worker UI** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/ui/`
   - Focus: WASM SDK pattern, React hooks

2. **wasm-bindgen Documentation**
   - Web APIs in Rust
   - JavaScript interop
   - TypeScript types

3. **TEAM-397's Backend** (What You're Calling)
   - HTTP API endpoints
   - SSE event format

---

## 🚨 Common Pitfalls

1. **WASM Build Issues**
   - Problem: wasm-pack fails
   - Solution: Install wasm-pack, check Rust version

2. **CORS Errors**
   - Problem: Browser blocks requests
   - Solution: Backend must send CORS headers

3. **SSE Connection**
   - Problem: EventSource doesn't connect
   - Solution: Check URL, check CORS, check network

4. **Base64 Images**
   - Problem: Image doesn't display
   - Solution: Add `data:image/png;base64,` prefix

---

## 🎯 Handoff to TEAM-400

**What TEAM-400 needs from you:**

### Files Created
- `ui/packages/sd-worker-sdk/` - WASM SDK
- `ui/packages/sd-worker-react/` - React hooks
- `ui/app/` - Main application with text-to-image

### What Works
- Job submission from browser
- SSE streaming in browser
- Text-to-image UI functional
- Progress bar updates
- Image display

### What TEAM-400 Will Add
- Image-to-image UI
- Inpainting UI with mask editor
- Image gallery
- Advanced controls

---

## 📊 Progress Tracking

- [ ] Week 1: WASM SDK complete
- [ ] Week 2: React hooks complete
- [ ] Week 3: Text-to-image UI complete, ready for TEAM-400

---

**TEAM-399: You're building the user experience. Make it smooth and intuitive.** 🎨
