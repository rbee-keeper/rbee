# Marketplace System - Implementation Guide

**Date:** 2025-11-04  
**Status:** 🎯 ACTIVE PLAN  
**Timeline:** 3.5 weeks

---

## 📖 Read These Documents In Order

### 1. **[COMPLETE_ONBOARDING_FLOW.md](./COMPLETE_ONBOARDING_FLOW.md)** ⭐ START HERE

**What it covers:**
- Complete user journey from Google search to running model
- SEO strategy (marketplace.rbee.dev)
- Tauri app integration (NOT SPA!)
- Auto-run flow (one-click to running model)
- Installation-aware buttons
- Multi-hive support (based on SSH config, NOT hives.conf)

**Why read this first:**
- Explains the complete vision
- Shows the user experience
- Clarifies why we need this architecture

**Key takeaway:**
```
Google → marketplace.rbee.dev → "Run with rbee" button
    ↓
    ├─> rbee installed? → Opens Keeper → Auto-downloads → Auto-installs → RUNNING!
    └─> rbee NOT installed? → Install instructions → Try again → Works!
```

---

### 2. **[MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md](./MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md)**

**What it covers:**
- Shared components package (`@rbee/marketplace-components`)
- Works in both Next.js (SSG) and Tauri
- Zero duplication architecture
- Marketplace SDK abstraction
- Monorepo structure
- Component design (DUMB components - no data fetching)

**Why read this:**
- Explains how to avoid code duplication
- Shows the technical architecture
- Provides code examples

**Key takeaway:**
```
@rbee/marketplace-components (SHARED - DUMB)
    ↓                    ↓
Next.js (SSG)      Keeper (Tauri)
SEO goldmine       Native app
```

---

### 3. **[URL_SCHEME_PATTERN.md](./URL_SCHEME_PATTERN.md)**

**What it covers:**
- `rbee://` protocol implementation
- "Open in App" pattern (like Steam, Spotify, VS Code)
- Cross-platform registration (Linux, macOS, Windows)
- URL syntax (`rbee://download/model/huggingface/llama-3.2-1b`)
- Security considerations

**Why read this:**
- Explains how the "Open in App" button works
- Shows how to register custom protocols
- Provides implementation examples

**Key takeaway:**
```html
<a href="rbee://download/model/huggingface/llama-3.2-1b">
  📦 Open in Keeper
</a>
```

---

### 4. **[PROTOCOL_DETECTION_FALLBACK.md](./PROTOCOL_DETECTION_FALLBACK.md)**

**What it covers:**
- Detect if rbee is installed
- Fallback to install page if not
- Installation-aware button implementation
- Hidden iframe technique
- Modal for install instructions

**Why read this:**
- Shows how to handle users without rbee installed
- Provides UX patterns for graceful fallback
- Code examples for detection

**Key takeaway:**
```typescript
// Try to open rbee://
const opened = await openInKeeperWithIframe(rbeeUrl)

if (!opened) {
  // Show install modal
  setShowInstallModal(true)
}
```

---

### 5. **[URL_PROTOCOL_REGISTRATION.md](./URL_PROTOCOL_REGISTRATION.md)**

**What it covers:**
- How to register `rbee://` on Linux/macOS/Windows
- Tauri integration
- Desktop file creation (Linux)
- Info.plist configuration (macOS)
- Registry keys (Windows)
- Code examples in Rust

**Why read this:**
- Implementation details for protocol registration
- Platform-specific code
- Tauri configuration

**Key takeaway:**
```json
// tauri.conf.json
{
  "tauri": {
    "bundle": {
      "protocols": [
        {
          "name": "rbee",
          "schemes": ["rbee"]
        }
      ]
    }
  }
}
```

---

## 🗑️ Outdated Documents (DO NOT USE)

These documents have been superseded by the new architecture:

- ❌ **MARKETPLACE_SYSTEM.md** - Old embedded components plan
- ❌ **MARKETPLACE_ARCHITECTURE_ANALYSIS.md** - Old comparison (deleted)
- ❌ **SPOTIFY_CONNECT_PATTERN.md** - Backend mediator approach (deleted)

**Why outdated:**
- Old plan: Embedded components in Keeper (no SEO)
- New plan: Separate Next.js site + Tauri app (SEO goldmine)
- Old plan: Duplication between web and app
- New plan: Shared components (zero duplication)

---

## 🎯 Quick Start for Engineers

**Want to implement the marketplace?**

1. **Read COMPLETE_ONBOARDING_FLOW.md** (15 min)
   - Understand the vision
   - See the user journey
   - Clarify requirements

2. **Read MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md** (20 min)
   - Understand the architecture
   - See the monorepo structure
   - Review component design

3. **Read URL_SCHEME_PATTERN.md** (10 min)
   - Understand protocol handling
   - See implementation examples

4. **Read PROTOCOL_DETECTION_FALLBACK.md** (10 min)
   - Understand fallback UX
   - See detection code

5. **Read URL_PROTOCOL_REGISTRATION.md** (15 min)
   - Understand platform-specific registration
   - See Tauri integration

**Total reading time: ~70 minutes**

---

## 🚀 Implementation Phases

### Phase 1: Shared Components (1 week)
- Create `@rbee/marketplace-components` package
- Create `@rbee/marketplace-sdk` package
- Make components dumb (props only, no data fetching)
- Test in both Next.js and Tauri

**Deliverables:**
- `ModelCard.tsx`
- `WorkerCard.tsx`
- `MarketplaceGrid.tsx`
- `FilterSidebar.tsx`
- `HuggingFaceClient.ts`
- `CivitAIClient.ts`
- `WorkerCatalogClient.ts`

### Phase 2: Next.js Site (1 week)
- Create `apps/marketplace-site`
- Set up Next.js with App Router
- Implement SSG for model pages
- Add installation-aware button
- Add install modal
- Deploy to Cloudflare Pages

**Deliverables:**
- `marketplace.rbee.dev/models/[id]` (SSG)
- `marketplace.rbee.dev/workers/[id]` (SSG)
- Installation detection
- Install instructions page

### Phase 3: Tauri Integration (1 week)
- Update Keeper to use Tauri
- Register `rbee://` protocol
- Implement protocol handler (Rust)
- Add Tauri commands (download, install, auto-run)
- Wire up frontend to Tauri commands
- Test end-to-end

**Deliverables:**
- `rbee://` protocol registration
- Protocol handler in Rust
- Tauri commands
- Frontend integration

### Phase 4: Polish (0.5 weeks)
- Multi-hive support (SSH config dropdown)
- Error handling
- Progress tracking
- Notifications
- Testing

**Deliverables:**
- Multi-hive dropdown
- Error messages
- Progress bars
- Toast notifications

---

## ✅ Success Criteria

**User Experience:**
- Google search → Running model: **5 minutes**
- Returning user → Running model: **30 seconds**
- Install rate: **>50%** (users who click "Run with rbee")
- Success rate: **>80%** (users who install and run model)

**SEO:**
- 1000+ model pages indexed
- "model name + rbee" rankings in top 10
- Backlinks from model searches

**Technical:**
- Zero component duplication
- SSG build time: **<5 minutes**
- Protocol detection: **<2 seconds**
- Auto-run flow: **<30 seconds** (download + install + run)

---

## 🔑 Key Concepts

### 1. **Shared Components = Zero Duplication**

Same components work in Next.js (SSG) and Tauri:

```tsx
// DUMB component - no data fetching
export function ModelCard({ model, onDownload }: Props) {
  return (
    <Card>
      <h3>{model.name}</h3>
      <p>{model.description}</p>
      <button onClick={() => onDownload(model.id)}>
        Download
      </button>
    </Card>
  )
}
```

Used in Next.js:
```tsx
// SSG - data fetched at build time
export default async function ModelPage({ params }) {
  const model = await fetchModel(params.id)
  return <ModelCard model={model} onDownload={handleDownload} />
}
```

Used in Tauri:
```tsx
// Client-side - data fetched at runtime
export function MarketplacePage() {
  const [models, setModels] = useState([])
  useEffect(() => {
    fetchModels().then(setModels)
  }, [])
  return models.map(model => (
    <ModelCard model={model} onDownload={handleDownload} />
  ))
}
```

### 2. **Protocol Links = Seamless Integration**

```html
<!-- Next.js site -->
<a href="rbee://download/model/huggingface/llama-3.2-1b">
  📦 Open in Keeper
</a>
```

When clicked:
1. Browser tries to open `rbee://` URL
2. OS checks if protocol is registered
3. If yes → Launches Keeper (Tauri app)
4. If no → Nothing happens (we detect this and show install modal)

### 3. **Auto-Run Flow = One-Click Magic**

User clicks "Run with rbee" → Everything happens automatically:

```rust
// Tauri command
#[tauri::command]
async fn auto_run_model(model_id: String) -> Result<String> {
    // 1. Start hive (if needed)
    ensure_hive_running().await?;
    
    // 2. Download model (if needed)
    ensure_model_downloaded(&model_id).await?;
    
    // 3. Install worker (if needed)
    let worker_type = detect_best_worker().await?;
    ensure_worker_installed(&worker_type).await?;
    
    // 4. Spawn worker
    let worker_id = spawn_worker(&model_id, &worker_type).await?;
    
    Ok(worker_id)
}
```

---

## 📚 Additional Resources

**Tauri Documentation:**
- https://tauri.app/v1/guides/features/deep-link

**Next.js Documentation:**
- https://nextjs.org/docs/app/building-your-application/routing/dynamic-routes
- https://nextjs.org/docs/app/building-your-application/data-fetching/fetching-caching-and-revalidating

**Protocol Registration:**
- Linux: https://specifications.freedesktop.org/desktop-entry-spec/latest/
- macOS: https://developer.apple.com/documentation/xcode/defining-a-custom-url-scheme-for-your-app
- Windows: https://docs.microsoft.com/en-us/windows/win32/shell/fa-progids

---

## 🤝 Contributing

**Before starting work:**
1. Read all 5 documents in order
2. Understand the architecture
3. Ask questions if anything is unclear

**When implementing:**
1. Follow the phase plan
2. Test in both Next.js and Tauri
3. Keep components dumb (no data fetching)
4. Use the SDK for data access

**When stuck:**
1. Re-read the relevant document
2. Check the code examples
3. Ask for help (reference the document)

---

**Start with COMPLETE_ONBOARDING_FLOW.md!** 🚀
