# URL Scheme Pattern (Custom Protocol Handlers)

**Date:** 2025-11-04  
**Purpose:** Simple "Open in App" pattern - no backend needed!

---

## 🎯 The Simple Pattern

### How It Works

```
Website (marketplace.rbee.dev)
    ↓ User clicks "Open in Keeper"
    ↓ Browser opens: rbee://download/llama-3.2-1b
    ↓ OS launches Keeper app
    ↓ Keeper receives: download/llama-3.2-1b
    ↓ Keeper downloads model
```

**No backend mediator needed!** ✅

---

## 📱 Real-World Examples

### 1. Steam
```html
<a href="steam://install/730">Install CS:GO</a>
```
- Browser opens `steam://` URL
- OS launches Steam app
- Steam receives command: `install/730`
- Steam downloads game

### 2. Spotify
```html
<a href="spotify:track:3n3Ppam7vgaVa1iaRUc9Lp">Play Song</a>
```
- Browser opens `spotify:` URL
- OS launches Spotify app
- Spotify plays track

### 3. VS Code
```html
<a href="vscode://file/path/to/file.ts">Open in VS Code</a>
```
- Browser opens `vscode://` URL
- OS launches VS Code
- VS Code opens file

### 4. Zoom
```html
<a href="zoommtg://zoom.us/join?confno=123456789">Join Meeting</a>
```
- Browser opens `zoommtg://` URL
- OS launches Zoom
- Zoom joins meeting

---

## 🐝 rbee URL Scheme

### Protocol: `rbee://`

**Examples:**
```
rbee://download/model/huggingface/llama-3.2-1b
rbee://download/model/civitai/sdxl-turbo
rbee://install/worker/llm-worker-rbee-cuda
rbee://open/hive/localhost
rbee://open/queen
```

### Marketplace Integration

**HTML:**
```html
<!-- HuggingFace Model Card -->
<div class="model-card">
  <h3>Llama 3.2 1B</h3>
  <p>Small, fast language model</p>
  
  <!-- Regular download button (works in browser) -->
  <button onclick="downloadInBrowser()">
    Download (Web)
  </button>
  
  <!-- Open in Keeper button -->
  <a href="rbee://download/model/huggingface/llama-3.2-1b" class="btn-primary">
    📦 Open in Keeper
  </a>
</div>
```

**User Experience:**
1. User browses marketplace.rbee.dev on phone
2. User finds "Llama 3.2 1B"
3. User clicks "Open in Keeper"
4. Browser shows: "Open rbee?"
5. User clicks "Open"
6. Keeper app launches on PC (if running)
7. Keeper shows: "Download Llama 3.2 1B?"
8. User confirms
9. Model downloads

---

## 🔧 Implementation

### 1. Register URL Scheme (Keeper)

**On Linux (`.desktop` file):**
```desktop
[Desktop Entry]
Name=rbee Keeper
Exec=/usr/local/bin/rbee-keeper %u
MimeType=x-scheme-handler/rbee;
Type=Application
```

**On macOS (Info.plist):**
```xml
<key>CFBundleURLTypes</key>
<array>
  <dict>
    <key>CFBundleURLName</key>
    <string>rbee</string>
    <key>CFBundleURLSchemes</key>
    <array>
      <string>rbee</string>
    </array>
  </dict>
</array>
```

**On Windows (Registry):**
```
HKEY_CLASSES_ROOT\rbee
  (Default) = "URL:rbee Protocol"
  URL Protocol = ""
  
HKEY_CLASSES_ROOT\rbee\shell\open\command
  (Default) = "C:\Program Files\rbee\keeper.exe" "%1"
```

### 2. Handle URL in Keeper

**File:** `bin/00_rbee_keeper/ui/src/lib/urlScheme.ts`

```typescript
export function handleUrlScheme(url: string) {
  // Parse: rbee://download/model/huggingface/llama-3.2-1b
  const parsed = new URL(url)
  
  const action = parsed.hostname // "download"
  const parts = parsed.pathname.split('/').filter(Boolean)
  
  switch (action) {
    case 'download':
      return handleDownload(parts)
    case 'install':
      return handleInstall(parts)
    case 'open':
      return handleOpen(parts)
    default:
      console.error('Unknown action:', action)
  }
}

async function handleDownload(parts: string[]) {
  // parts = ["model", "huggingface", "llama-3.2-1b"]
  const [type, source, id] = parts
  
  if (type === 'model') {
    // Show confirmation dialog
    const confirmed = await showDialog({
      title: 'Download Model',
      message: `Download ${id} from ${source}?`,
      buttons: ['Cancel', 'Download']
    })
    
    if (confirmed) {
      // Submit job to Queen
      await fetch('http://localhost:8500/v1/jobs', {
        method: 'POST',
        body: JSON.stringify({
          operation: {
            ModelDownload: {
              hive_id: 'localhost',
              model_id: id,
              source: source
            }
          }
        })
      })
      
      // Show notification
      showNotification('Download started', `Downloading ${id}`)
    }
  }
}

async function handleInstall(parts: string[]) {
  // parts = ["worker", "llm-worker-rbee-cuda"]
  const [type, workerId] = parts
  
  if (type === 'worker') {
    const confirmed = await showDialog({
      title: 'Install Worker',
      message: `Install ${workerId}?`,
      buttons: ['Cancel', 'Install']
    })
    
    if (confirmed) {
      await fetch('http://localhost:8500/v1/jobs', {
        method: 'POST',
        body: JSON.stringify({
          operation: {
            WorkerInstall: {
              hive_id: 'localhost',
              worker_id: workerId
            }
          }
        })
      })
      
      showNotification('Installation started', `Installing ${workerId}`)
    }
  }
}

async function handleOpen(parts: string[]) {
  // parts = ["hive", "localhost"] or ["queen"]
  const [type, id] = parts
  
  // Open tab in Keeper
  const { addTab } = useTabStore.getState()
  
  if (type === 'hive') {
    addTab({
      type: 'hive',
      title: `Hive: ${id}`,
      icon: '🏠',
      route: `/hive/${id}`,
      hiveId: id
    })
  } else if (type === 'queen') {
    addTab({
      type: 'queen',
      title: 'Queen',
      icon: '👑',
      route: '/queen'
    })
  }
}
```

### 3. Initialize on Startup

**File:** `bin/00_rbee_keeper/ui/src/App.tsx`

```typescript
import { handleUrlScheme } from '@/lib/urlScheme'

function App() {
  useEffect(() => {
    // Check if launched with URL scheme
    const urlParam = new URLSearchParams(window.location.search).get('url')
    if (urlParam) {
      handleUrlScheme(urlParam)
    }
    
    // Listen for URL scheme events (if app is already running)
    window.addEventListener('open-url', (event: CustomEvent) => {
      handleUrlScheme(event.detail.url)
    })
  }, [])
  
  // ... rest of app
}
```

### 4. Marketplace Website

**File:** `marketplace.rbee.dev/components/ModelCard.tsx`

```tsx
export function ModelCard({ model }: { model: Model }) {
  const rbeeUrl = `rbee://download/model/huggingface/${model.id}`
  
  return (
    <div className="model-card">
      <h3>{model.name}</h3>
      <p>{model.description}</p>
      
      <div className="actions">
        {/* Web download (fallback) */}
        <button onClick={() => downloadInBrowser(model)}>
          Download (Web)
        </button>
        
        {/* Open in Keeper (preferred) */}
        <a 
          href={rbeeUrl}
          className="btn-primary"
          onClick={(e) => {
            // Check if Keeper is installed
            if (!isKeeperInstalled()) {
              e.preventDefault()
              showInstallPrompt()
            }
          }}
        >
          📦 Open in Keeper
        </a>
      </div>
    </div>
  )
}

function isKeeperInstalled(): boolean {
  // Try to detect if Keeper is installed
  // This is tricky - usually just show the button and let browser handle it
  return true
}

function showInstallPrompt() {
  alert('Keeper is not installed. Download it from rbee.dev')
}
```

---

## 🎨 User Experience

### Scenario 1: Browse on Phone, Download on PC

**Steps:**
1. User browses marketplace.rbee.dev on phone
2. User finds model they want
3. User clicks "Open in Keeper"
4. Phone shows: "Open rbee?" → "Open"
5. **Nothing happens** (Keeper not on phone)
6. User goes to PC
7. User opens Keeper
8. **Alternative:** User copies link, sends to PC, opens there

**Better UX:**
```html
<!-- Show QR code for PC -->
<button onclick="showQRCode('rbee://download/model/...')">
  📱 Scan to Download on PC
</button>
```

### Scenario 2: Browse on PC, Download Immediately

**Steps:**
1. User browses marketplace.rbee.dev on PC
2. User finds model they want
3. User clicks "Open in Keeper"
4. Browser shows: "Open rbee?" → "Open"
5. Keeper launches (if not running) or focuses (if running)
6. Keeper shows: "Download Llama 3.2 1B?"
7. User clicks "Download"
8. Model downloads

### Scenario 3: Share Link with Friend

**Steps:**
1. User copies link: `rbee://download/model/huggingface/llama-3.2-1b`
2. User sends to friend via Discord/Slack
3. Friend clicks link
4. Friend's Keeper opens
5. Friend downloads model

---

## 🔒 Security Considerations

### 1. Confirmation Dialogs

**Always confirm before executing:**
```typescript
// ❌ BAD - Auto-execute
handleUrlScheme('rbee://download/model/...')
  → Immediately starts download

// ✅ GOOD - Confirm first
handleUrlScheme('rbee://download/model/...')
  → Show dialog: "Download X?"
  → User clicks "Download"
  → Start download
```

### 2. Validate Input

```typescript
function handleDownload(parts: string[]) {
  const [type, source, id] = parts
  
  // Validate source
  const allowedSources = ['huggingface', 'civitai']
  if (!allowedSources.includes(source)) {
    throw new Error('Invalid source')
  }
  
  // Validate model ID (no path traversal)
  if (id.includes('..') || id.includes('/')) {
    throw new Error('Invalid model ID')
  }
  
  // Proceed with download
}
```

### 3. Rate Limiting

```typescript
const recentCommands = new Map<string, number>()

function handleUrlScheme(url: string) {
  const now = Date.now()
  const lastExecution = recentCommands.get(url) || 0
  
  // Prevent spam (max 1 per 5 seconds)
  if (now - lastExecution < 5000) {
    console.warn('Rate limited:', url)
    return
  }
  
  recentCommands.set(url, now)
  
  // Execute command
}
```

---

## 📊 Comparison: URL Scheme vs Backend Mediator

| Feature | URL Scheme | Backend Mediator (Spotify) |
|---------|------------|---------------------------|
| Complexity | Low ⭐ | High |
| Backend needed | No ⭐ | Yes |
| Works offline | Yes ⭐ | No |
| Remote control | No | Yes ⭐ |
| Multi-device | No | Yes ⭐ |
| Phone → PC | Limited | Yes ⭐ |
| Shareable links | Yes ⭐ | Yes ⭐ |
| Development time | 1 week ⭐ | 6 weeks |
| User must have app | Yes | Yes |

---

## ✅ Final Recommendation

### **Use URL Scheme Pattern** ✅

**Why:**
- ✅ Simple (no backend needed)
- ✅ Fast to implement (1 week)
- ✅ Works offline
- ✅ Industry standard (Steam, Spotify, VS Code all use it)
- ✅ Shareable links
- ✅ "Open in App" UX is familiar to users

**Limitations:**
- ❌ Can't remote control (phone → PC requires app on phone)
- ❌ Can't manage multiple devices from one interface

**Solution:** Start with URL scheme, add backend mediator later if needed

---

## 🚀 Implementation Plan

### Phase 1: URL Scheme Registration (2 days)
- Register `rbee://` protocol on Linux/macOS/Windows
- Test protocol handler launches Keeper
- Handle command-line arguments

### Phase 2: URL Parser & Handlers (2 days)
- Parse `rbee://` URLs
- Implement download/install/open handlers
- Add confirmation dialogs
- Add security validation

### Phase 3: Marketplace Website (1 week)
- Build Next.js site
- Add "Open in Keeper" buttons
- Add QR codes for phone → PC
- Deploy to Cloudflare Pages

### Phase 4: Polish (2 days)
- Error handling (Keeper not installed)
- Rate limiting
- Notifications
- Testing

**Total: 2 weeks** (vs 6 weeks for backend mediator)

---

## 🎯 URL Scheme Syntax

### Download Model
```
rbee://download/model/{source}/{model_id}

Examples:
rbee://download/model/huggingface/llama-3.2-1b
rbee://download/model/civitai/sdxl-turbo
```

### Install Worker
```
rbee://install/worker/{worker_id}

Examples:
rbee://install/worker/llm-worker-rbee-cpu
rbee://install/worker/llm-worker-rbee-cuda
```

### Open Tab
```
rbee://open/{type}/{id}

Examples:
rbee://open/hive/localhost
rbee://open/queen
rbee://open/marketplace/huggingface
```

### Spawn Worker
```
rbee://spawn/worker/{hive_id}/{worker_type}/{model_id}

Examples:
rbee://spawn/worker/localhost/cpu/llama-3.2-1b
```

---

**This is the simple solution you wanted!** 🎯
