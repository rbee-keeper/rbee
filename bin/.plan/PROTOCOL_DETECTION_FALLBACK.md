# Protocol Detection & Fallback Pattern

**Date:** 2025-11-04  
**Purpose:** Detect if `rbee://` is registered, fallback to download page if not

---

## 🎯 The Pattern

```
User clicks "Open in Keeper"
    ↓
Try to open rbee:// URL
    ↓
    ├─> Protocol registered? → Open Keeper app
    │                           User sees download dialog
    │                           User clicks "Download"
    │
    └─> Protocol NOT registered? → Redirect to rbee.dev/download
                                    User downloads Keeper
                                    User installs Keeper
                                    User tries again
```

---

## 🔍 Detection Methods

### Method 1: Timeout-Based Detection (Most Reliable)

**How it works:**
1. Try to open `rbee://` URL
2. Set 2-second timeout
3. If app opens → user leaves page (blur event)
4. If timeout fires → app not installed → redirect

**Code:**

```typescript
// marketplace.rbee.dev/lib/protocolDetection.ts

export function openInKeeper(url: string, fallbackUrl: string = 'https://rbee.dev/download') {
  let timeout: NodeJS.Timeout
  let hasBlurred = false
  
  // Detect if user left page (app opened)
  const handleBlur = () => {
    hasBlurred = true
    clearTimeout(timeout)
  }
  
  // Set timeout for fallback
  timeout = setTimeout(() => {
    if (!hasBlurred) {
      // App didn't open - redirect to download page
      window.location.href = fallbackUrl
    }
    window.removeEventListener('blur', handleBlur)
  }, 2000)
  
  window.addEventListener('blur', handleBlur)
  
  // Try to open protocol
  window.location.href = url
}
```

**Usage:**

```tsx
// marketplace.rbee.dev/components/ModelCard.tsx

export function ModelCard({ model }: { model: Model }) {
  const handleOpenInKeeper = () => {
    const rbeeUrl = `rbee://download/model/huggingface/${model.id}`
    openInKeeper(rbeeUrl, 'https://rbee.dev/download')
  }
  
  return (
    <div className="model-card">
      <h3>{model.name}</h3>
      <p>{model.description}</p>
      
      <button onClick={handleOpenInKeeper} className="btn-primary">
        📦 Open in Keeper
      </button>
    </div>
  )
}
```

### Method 2: Hidden Iframe (Cleaner UX)

**How it works:**
1. Create hidden iframe
2. Try to load `rbee://` URL in iframe
3. If fails → show download prompt
4. If succeeds → app opens

**Code:**

```typescript
// marketplace.rbee.dev/lib/protocolDetection.ts

export function openInKeeperWithIframe(url: string) {
  return new Promise<boolean>((resolve) => {
    const iframe = document.createElement('iframe')
    iframe.style.display = 'none'
    document.body.appendChild(iframe)
    
    let timeout: NodeJS.Timeout
    let hasBlurred = false
    
    const handleBlur = () => {
      hasBlurred = true
      clearTimeout(timeout)
      cleanup()
      resolve(true) // App opened
    }
    
    const cleanup = () => {
      window.removeEventListener('blur', handleBlur)
      document.body.removeChild(iframe)
    }
    
    timeout = setTimeout(() => {
      if (!hasBlurred) {
        cleanup()
        resolve(false) // App not installed
      }
    }, 2000)
    
    window.addEventListener('blur', handleBlur)
    
    // Try to open in iframe
    iframe.src = url
  })
}
```

**Usage with Modal:**

```tsx
// marketplace.rbee.dev/components/ModelCard.tsx

import { useState } from 'react'
import { openInKeeperWithIframe } from '@/lib/protocolDetection'

export function ModelCard({ model }: { model: Model }) {
  const [showDownloadPrompt, setShowDownloadPrompt] = useState(false)
  
  const handleOpenInKeeper = async () => {
    const rbeeUrl = `rbee://download/model/huggingface/${model.id}`
    const opened = await openInKeeperWithIframe(rbeeUrl)
    
    if (!opened) {
      // App not installed - show download prompt
      setShowDownloadPrompt(true)
    }
  }
  
  return (
    <>
      <div className="model-card">
        <h3>{model.name}</h3>
        <p>{model.description}</p>
        
        <button onClick={handleOpenInKeeper} className="btn-primary">
          📦 Open in Keeper
        </button>
      </div>
      
      {showDownloadPrompt && (
        <DownloadPromptModal
          onClose={() => setShowDownloadPrompt(false)}
          modelName={model.name}
        />
      )}
    </>
  )
}
```

### Method 3: User Choice (Best UX)

**Show both options upfront:**

```tsx
export function ModelCard({ model }: { model: Model }) {
  const rbeeUrl = `rbee://download/model/huggingface/${model.id}`
  
  return (
    <div className="model-card">
      <h3>{model.name}</h3>
      <p>{model.description}</p>
      
      <div className="actions">
        {/* Primary: Open in app */}
        <a 
          href={rbeeUrl}
          className="btn-primary"
          onClick={(e) => {
            // Optional: Show hint after 2 seconds
            setTimeout(() => {
              if (document.hasFocus()) {
                // User still on page - app might not be installed
                showToast("Don't have Keeper? Download it below")
              }
            }, 2000)
          }}
        >
          📦 Open in Keeper
        </a>
        
        {/* Secondary: Download app */}
        <a 
          href="https://rbee.dev/download"
          className="btn-secondary"
          target="_blank"
        >
          ⬇️ Get Keeper
        </a>
      </div>
    </div>
  )
}
```

---

## 🎨 Download Prompt Modal

**Component:**

```tsx
// marketplace.rbee.dev/components/DownloadPromptModal.tsx

interface DownloadPromptModalProps {
  onClose: () => void
  modelName: string
}

export function DownloadPromptModal({ onClose, modelName }: DownloadPromptModalProps) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Keeper Not Installed</h2>
        <p>
          To download <strong>{modelName}</strong>, you need to install rbee Keeper first.
        </p>
        
        <div className="modal-actions">
          <a 
            href="https://rbee.dev/download" 
            className="btn-primary"
            target="_blank"
          >
            Download Keeper
          </a>
          
          <button onClick={onClose} className="btn-secondary">
            Cancel
          </button>
        </div>
        
        <div className="modal-hint">
          <p className="text-sm text-muted">
            After installing, come back and click "Open in Keeper" again.
          </p>
        </div>
      </div>
    </div>
  )
}
```

**Styling:**

```css
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  max-width: 500px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-actions {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
}

.modal-hint {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}
```

---

## 🚀 Complete Implementation

### marketplace.rbee.dev/lib/protocolDetection.ts

```typescript
/**
 * Try to open rbee:// URL, fallback to download page if not installed
 */
export function openInKeeper(
  url: string, 
  options?: {
    fallbackUrl?: string
    onSuccess?: () => void
    onFallback?: () => void
  }
) {
  const fallbackUrl = options?.fallbackUrl || 'https://rbee.dev/download'
  let timeout: NodeJS.Timeout
  let hasBlurred = false
  
  const handleBlur = () => {
    hasBlurred = true
    clearTimeout(timeout)
    options?.onSuccess?.()
  }
  
  timeout = setTimeout(() => {
    if (!hasBlurred) {
      // App didn't open
      options?.onFallback?.()
      
      // Redirect to download page
      window.location.href = fallbackUrl
    }
    window.removeEventListener('blur', handleBlur)
  }, 2000)
  
  window.addEventListener('blur', handleBlur)
  
  // Try to open protocol
  window.location.href = url
}

/**
 * Try to open rbee:// URL with iframe (cleaner, no navigation)
 */
export async function openInKeeperWithIframe(url: string): Promise<boolean> {
  return new Promise((resolve) => {
    const iframe = document.createElement('iframe')
    iframe.style.display = 'none'
    document.body.appendChild(iframe)
    
    let timeout: NodeJS.Timeout
    let hasBlurred = false
    
    const handleBlur = () => {
      hasBlurred = true
      clearTimeout(timeout)
      cleanup()
      resolve(true)
    }
    
    const cleanup = () => {
      window.removeEventListener('blur', handleBlur)
      document.body.removeChild(iframe)
    }
    
    timeout = setTimeout(() => {
      if (!hasBlurred) {
        cleanup()
        resolve(false)
      }
    }, 2000)
    
    window.addEventListener('blur', handleBlur)
    iframe.src = url
  })
}

/**
 * Check if protocol is registered (best effort)
 */
export async function isProtocolRegistered(protocol: string): Promise<boolean> {
  // This is not 100% reliable, but gives a hint
  try {
    const result = await openInKeeperWithIframe(`${protocol}://test`)
    return result
  } catch {
    return false
  }
}
```

### marketplace.rbee.dev/components/OpenInKeeperButton.tsx

```tsx
import { useState } from 'react'
import { openInKeeperWithIframe } from '@/lib/protocolDetection'
import { DownloadPromptModal } from './DownloadPromptModal'

interface OpenInKeeperButtonProps {
  url: string
  label?: string
  modelName?: string
}

export function OpenInKeeperButton({ 
  url, 
  label = 'Open in Keeper',
  modelName 
}: OpenInKeeperButtonProps) {
  const [showDownloadPrompt, setShowDownloadPrompt] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  
  const handleClick = async () => {
    setIsLoading(true)
    
    const opened = await openInKeeperWithIframe(url)
    
    setIsLoading(false)
    
    if (!opened) {
      setShowDownloadPrompt(true)
    }
  }
  
  return (
    <>
      <button 
        onClick={handleClick}
        disabled={isLoading}
        className="btn-primary"
      >
        {isLoading ? '⏳ Opening...' : `📦 ${label}`}
      </button>
      
      {showDownloadPrompt && (
        <DownloadPromptModal
          onClose={() => setShowDownloadPrompt(false)}
          modelName={modelName || 'this model'}
        />
      )}
    </>
  )
}
```

### marketplace.rbee.dev/components/ModelCard.tsx

```tsx
import { OpenInKeeperButton } from './OpenInKeeperButton'

export function ModelCard({ model }: { model: Model }) {
  const rbeeUrl = `rbee://download/model/huggingface/${model.id}`
  
  return (
    <div className="model-card">
      <img src={model.imageUrl} alt={model.name} />
      <h3>{model.name}</h3>
      <p>{model.description}</p>
      
      <div className="model-stats">
        <span>⬇️ {formatNumber(model.downloads)}</span>
        <span>⭐ {model.rating}</span>
        <span>📦 {model.size}</span>
      </div>
      
      <div className="model-actions">
        <OpenInKeeperButton 
          url={rbeeUrl}
          modelName={model.name}
        />
        
        <a 
          href={`https://huggingface.co/${model.id}`}
          target="_blank"
          className="btn-secondary"
        >
          View on HuggingFace
        </a>
      </div>
    </div>
  )
}
```

---

## 🎯 User Flows

### Flow 1: Keeper Installed

```
1. User clicks "Open in Keeper"
2. Browser opens rbee:// URL
3. OS launches Keeper app
4. User leaves page (blur event)
5. ✅ Success - no redirect
6. Keeper shows: "Download Llama 3.2 1B?"
7. User clicks "Download"
8. Model downloads
```

### Flow 2: Keeper NOT Installed

```
1. User clicks "Open in Keeper"
2. Browser tries to open rbee:// URL
3. Nothing happens (protocol not registered)
4. 2-second timeout fires
5. Modal appears: "Keeper Not Installed"
6. User clicks "Download Keeper"
7. User goes to rbee.dev/download
8. User downloads and installs Keeper
9. User returns to marketplace
10. User clicks "Open in Keeper" again
11. ✅ Now it works
```

### Flow 3: User Choice (Alternative)

```
1. User sees two buttons:
   - "Open in Keeper" (primary)
   - "Get Keeper" (secondary)
2. User clicks "Open in Keeper"
3. If installed → Keeper opens
4. If not installed → Toast: "Don't have Keeper? Click 'Get Keeper'"
5. User clicks "Get Keeper"
6. User downloads and installs
```

---

## 📱 Mobile Considerations

**On mobile (iOS/Android):**
- Protocol detection is less reliable
- Better to show both options upfront

```tsx
export function ModelCard({ model }: { model: Model }) {
  const isMobile = /iPhone|iPad|Android/i.test(navigator.userAgent)
  const rbeeUrl = `rbee://download/model/huggingface/${model.id}`
  
  if (isMobile) {
    return (
      <div className="model-card">
        <h3>{model.name}</h3>
        <p>{model.description}</p>
        
        <div className="mobile-actions">
          <p className="text-sm text-muted">
            To download this model, install rbee Keeper on your PC
          </p>
          
          <a href="https://rbee.dev/download" className="btn-primary">
            Get Keeper for PC
          </a>
          
          {/* Optional: Show QR code */}
          <button onClick={() => showQRCode(rbeeUrl)}>
            📱 Scan QR Code
          </button>
        </div>
      </div>
    )
  }
  
  // Desktop: Show "Open in Keeper" button
  return (
    <div className="model-card">
      <h3>{model.name}</h3>
      <p>{model.description}</p>
      
      <OpenInKeeperButton 
        url={rbeeUrl}
        modelName={model.name}
      />
    </div>
  )
}
```

---

## ✅ Recommended Approach

**Use Method 2 (Hidden Iframe) + Modal:**

1. User clicks "Open in Keeper"
2. Try to open in hidden iframe
3. If succeeds (blur event) → App opens ✅
4. If fails (timeout) → Show modal with download link
5. User downloads Keeper
6. User tries again → Works ✅

**Why this is best:**
- ✅ Clean UX (no page navigation)
- ✅ Clear feedback (modal explains what to do)
- ✅ Works on all browsers
- ✅ Fallback is obvious
- ✅ User stays on marketplace page

---

## 🚀 Implementation Checklist

- [ ] Create `protocolDetection.ts` utility
- [ ] Create `OpenInKeeperButton` component
- [ ] Create `DownloadPromptModal` component
- [ ] Add to all model/worker cards
- [ ] Test with Keeper installed
- [ ] Test with Keeper NOT installed
- [ ] Add mobile detection
- [ ] Add QR code for mobile users
- [ ] Add analytics (track install rate)

---

**This gives users a seamless experience whether Keeper is installed or not!** 🎯
