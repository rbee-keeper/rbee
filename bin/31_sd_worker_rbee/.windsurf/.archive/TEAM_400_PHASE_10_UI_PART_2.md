# TEAM-400: Phase 10 - UI Development Part 2

**Team:** TEAM-400  
**Phase:** 10 - UI Features  
**Duration:** 45 hours  
**Dependencies:** TEAM-399 (UI foundation)  
**Parallel Work:** None (needs TEAM-399's foundation)

---

## 🎯 Mission

Complete the UI with image-to-image, inpainting with mask editor, image gallery, and advanced parameter controls. Deliver production-ready user interface.

---

## 📦 What You're Building

### Features to Add (3 major features)

1. **Image-to-Image UI** (~15 hours)
   - Image upload
   - Strength slider
   - Preview original + generated

2. **Inpainting UI** (~20 hours)
   - Canvas-based mask editor
   - Brush size controls
   - Mask preview
   - Inpaint generation

3. **Image Gallery** (~10 hours)
   - Grid view of generated images
   - Image metadata
   - Download/delete actions
   - Local storage persistence

---

## 📋 Task Breakdown

### Week 1: Image-to-Image (16 hours)

**Day 1: Image Upload (8 hours)**
- [ ] Create useImageToImage hook (2 hours)
- [ ] Create image upload component (2 hours)
- [ ] Add image preview (2 hours)
- [ ] Add strength slider (2 hours)

**Day 2: Integration (8 hours)**
- [ ] Wire up backend API (2 hours)
- [ ] Add progress tracking (2 hours)
- [ ] Add side-by-side comparison (2 hours)
- [ ] Test and polish (2 hours)

---

### Week 2: Inpainting (20 hours)

**Day 3: Canvas Setup (8 hours)**
- [ ] Create canvas component (2 hours)
- [ ] Add image loading to canvas (2 hours)
- [ ] Add brush tool (2 hours)
- [ ] Add eraser tool (2 hours)

**Day 4: Mask Editor (8 hours)**
- [ ] Add brush size controls (2 hours)
- [ ] Add undo/redo (2 hours)
- [ ] Add clear mask button (2 hours)
- [ ] Add mask preview toggle (2 hours)

**Day 5: Inpaint Integration (4 hours)**
- [ ] Create useInpainting hook (2 hours)
- [ ] Wire up backend API (1 hour)
- [ ] Test and polish (1 hour)

---

### Week 3: Gallery & Polish (9 hours)

**Day 6: Image Gallery (5 hours)**
- [ ] Create gallery component (2 hours)
- [ ] Add local storage (1 hour)
- [ ] Add download/delete actions (1 hour)
- [ ] Add metadata display (1 hour)

**Day 7: Final Polish (4 hours)**
- [ ] Add advanced controls (1 hour)
- [ ] Improve error messages (1 hour)
- [ ] Add keyboard shortcuts (1 hour)
- [ ] Final testing (1 hour)

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] Image-to-image works with uploaded images
- [ ] Strength slider affects generation
- [ ] Inpainting mask editor functional
- [ ] Brush and eraser tools work
- [ ] Undo/redo works in mask editor
- [ ] Inpainting generates correctly
- [ ] Gallery displays all generated images
- [ ] Images persist in local storage
- [ ] Download/delete actions work
- [ ] All features tested
- [ ] Clean build (0 warnings)

---

## 🔧 Implementation Notes

### Image-to-Image Hook

```typescript
// ui/packages/sd-worker-react/src/hooks/useImageToImage.ts
export interface ImageToImageParams {
  prompt: string;
  initImage: string; // base64
  strength: number; // 0.0-1.0
  steps?: number;
  guidanceScale?: number;
  seed?: number;
}

export function useImageToImage(baseUrl: string) {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const generate = useCallback(async (params: ImageToImageParams) => {
    // Similar to useTextToImage but includes init_image and strength
    const client = new SDWorkerClient(baseUrl);
    
    const jobId = await client.submit_job({
      prompt: params.prompt,
      init_image: params.initImage,
      strength: params.strength,
      steps: params.steps ?? 20,
      guidance_scale: params.guidanceScale ?? 7.5,
      seed: params.seed,
    });
    
    // Stream progress...
  }, [baseUrl]);
  
  return { generate, loading, progress, image, error };
}
```

### Mask Editor Component

```typescript
// ui/app/src/components/MaskEditor.tsx
import React, { useRef, useEffect, useState } from 'react';

export function MaskEditor({ image, onMaskChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [brushSize, setBrushSize] = useState(20);
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush');
  const [isDrawing, setIsDrawing] = useState(false);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Load image
    const img = new Image();
    img.src = image;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
  }, [image]);
  
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDrawing(true);
    draw(e);
  };
  
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    draw(e);
  };
  
  const handleMouseUp = () => {
    setIsDrawing(false);
    exportMask();
  };
  
  const draw = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    
    if (tool === 'brush') {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    } else {
      ctx.globalCompositeOperation = 'destination-out';
    }
    
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  };
  
  const exportMask = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) return;
    
    // Extract mask (white pixels only)
    const imageData = canvas.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height);
    if (!imageData) return;
    
    const maskData = maskCtx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const alpha = imageData.data[i + 3];
      maskData.data[i] = alpha > 128 ? 255 : 0;
      maskData.data[i + 1] = alpha > 128 ? 255 : 0;
      maskData.data[i + 2] = alpha > 128 ? 255 : 0;
      maskData.data[i + 3] = 255;
    }
    
    maskCtx.putImageData(maskData, 0, 0);
    const maskBase64 = maskCanvas.toDataURL('image/png');
    onMaskChange(maskBase64);
  };
  
  return (
    <div>
      {/* Controls */}
      <div className="mb-4 flex gap-4">
        <button
          onClick={() => setTool('brush')}
          className={tool === 'brush' ? 'bg-blue-600 text-white' : 'bg-gray-200'}
        >
          Brush
        </button>
        <button
          onClick={() => setTool('eraser')}
          className={tool === 'eraser' ? 'bg-blue-600 text-white' : 'bg-gray-200'}
        >
          Eraser
        </button>
        <input
          type="range"
          min="5"
          max="50"
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value))}
        />
        <span>Size: {brushSize}</span>
      </div>
      
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="border cursor-crosshair"
      />
    </div>
  );
}
```

### Image Gallery

```typescript
// ui/app/src/components/ImageGallery.tsx
import React, { useState, useEffect } from 'react';

interface GeneratedImage {
  id: string;
  image: string; // base64
  prompt: string;
  timestamp: number;
  params: any;
}

export function ImageGallery() {
  const [images, setImages] = useState<GeneratedImage[]>([]);
  
  useEffect(() => {
    // Load from local storage
    const stored = localStorage.getItem('generated_images');
    if (stored) {
      setImages(JSON.parse(stored));
    }
  }, []);
  
  const addImage = (image: GeneratedImage) => {
    const updated = [image, ...images];
    setImages(updated);
    localStorage.setItem('generated_images', JSON.stringify(updated));
  };
  
  const deleteImage = (id: string) => {
    const updated = images.filter(img => img.id !== id);
    setImages(updated);
    localStorage.setItem('generated_images', JSON.stringify(updated));
  };
  
  const downloadImage = (image: GeneratedImage) => {
    const link = document.createElement('a');
    link.href = image.image;
    link.download = `generated_${image.id}.png`;
    link.click();
  };
  
  return (
    <div className="grid grid-cols-3 gap-4">
      {images.map(img => (
        <div key={img.id} className="border rounded-lg overflow-hidden">
          <img src={img.image} alt={img.prompt} className="w-full" />
          <div className="p-3">
            <p className="text-sm truncate">{img.prompt}</p>
            <div className="flex gap-2 mt-2">
              <button
                onClick={() => downloadImage(img)}
                className="text-sm text-blue-600"
              >
                Download
              </button>
              <button
                onClick={() => deleteImage(img.id)}
                className="text-sm text-red-600"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **TEAM-399's Foundation** (Your Dependency)
   - WASM SDK
   - React hooks
   - Text-to-image UI

2. **Canvas API Documentation**
   - Drawing operations
   - Mouse events
   - Image manipulation

3. **Local Storage API**
   - Storing images
   - Size limits (5-10MB typical)

---

## 🚨 Common Pitfalls

1. **Canvas Performance**
   - Problem: Slow drawing with large images
   - Solution: Scale down for editing, scale up for export

2. **Local Storage Limits**
   - Problem: Quota exceeded with many images
   - Solution: Limit gallery size, compress images

3. **Mask Quality**
   - Problem: Jagged mask edges
   - Solution: Use anti-aliasing, smooth brush

4. **Image Upload Size**
   - Problem: Large uploads fail
   - Solution: Resize images before upload

---

## 🎯 Handoff to TEAM-401

**What TEAM-401 needs from you:**

### Files Created
- Image-to-image UI
- Inpainting UI with mask editor
- Image gallery
- Advanced controls

### What Works
- Complete UI for all generation modes
- Mask editor with brush/eraser
- Gallery with persistence
- All features tested

### What TEAM-401 Will Do
- Performance optimization
- Documentation
- Deployment preparation
- Final polish

---

## 📊 Progress Tracking

- [ ] Week 1: Image-to-image complete
- [ ] Week 2: Inpainting complete
- [ ] Week 3: Gallery and polish complete, ready for TEAM-401

---

**TEAM-400: You're completing the user experience. Make it delightful.** ✨
