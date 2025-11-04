# Marketplace System - SUPERSEDED

**Status:** ⚠️ OUTDATED - See new architecture  
**Date:** 2025-11-04

---

## 🚨 THIS DOCUMENT IS OUTDATED

**This plan has been replaced by a better architecture.**

### ⭐ Read These Instead (In Order):

1. **[COMPLETE_ONBOARDING_FLOW.md](./COMPLETE_ONBOARDING_FLOW.md)** ← START HERE
   - Complete user journey from Google search to running model
   - SEO strategy (marketplace.rbee.dev)
   - Tauri app integration (NOT SPA!)
   - Auto-run flow (one-click to running model)
   - Installation-aware buttons

2. **[MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md](./MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md)**
   - Shared components package (`@rbee/marketplace-components`)
   - Works in both Next.js (SSG) and Tauri
   - Zero duplication architecture
   - Marketplace SDK abstraction

3. **[URL_SCHEME_PATTERN.md](./URL_SCHEME_PATTERN.md)**
   - `rbee://` protocol implementation
   - "Open in App" pattern (like Steam, Spotify, VS Code)
   - Cross-platform registration

4. **[PROTOCOL_DETECTION_FALLBACK.md](./PROTOCOL_DETECTION_FALLBACK.md)**
   - Detect if rbee is installed
   - Fallback to install page if not
   - Installation-aware button implementation

5. **[URL_PROTOCOL_REGISTRATION.md](./URL_PROTOCOL_REGISTRATION.md)**
   - How to register `rbee://` on Linux/macOS/Windows
   - Tauri integration
   - Code examples

---

## 🎯 New Architecture Summary

### The Vision

**SEO Goldmine:**
- Every AI model gets its own page on `marketplace.rbee.dev`
- Pre-rendered with Next.js SSG
- Google indexes: "Llama 3.2 + rbee", "SDXL + rbee", etc.
- Massive backlinks from model searches

**Zero Duplication:**
- ONE set of presentation components
- Works in Next.js (SSG for SEO)
- Works in Tauri (native app)
- Shared package: `@rbee/marketplace-components`

**User Flow:**
```
Google search → marketplace.rbee.dev → "Run with rbee" button
    ↓
    ├─> rbee installed? → Opens Keeper (Tauri)
    │                      → Auto-downloads model
    │                      → Auto-installs worker
    │                      → 🎉 RUNNING!
    │
    └─> rbee NOT installed? → Shows install instructions
                               → User installs
                               → Clicks button again
                               → Now works! ✅
```

---

## 📦 Architecture

### Monorepo Structure

```
frontend/
├─> packages/
│   ├─> marketplace-components/     ← SHARED (Next.js + Tauri)
│   │   ├─> ModelCard.tsx
│   │   ├─> WorkerCard.tsx
│   │   ├─> MarketplaceGrid.tsx
│   │   └─> FilterSidebar.tsx
│   │
│   └─> marketplace-sdk/            ← DATA LAYER
│       ├─> HuggingFaceClient.ts
│       ├─> CivitAIClient.ts
│       └─> WorkerCatalogClient.ts
│
├─> apps/
│   ├─> marketplace-site/           ← NEXT.JS (SSG)
│   │   └─> marketplace.rbee.dev
│   │
│   └─> keeper/                     ← TAURI APP
│       ├─> src/                    (React)
│       └─> src-tauri/              (Rust)
```

### Key Principles

1. **Components are DUMB**
   - No data fetching
   - Props in, JSX out
   - Work in SSG AND Tauri

2. **SDK handles data**
   - Abstract interface
   - Multiple implementations
   - Apps choose how to use

3. **Next.js for SEO**
   - Pre-render top 1000 models
   - Each model = own page
   - Button: `rbee://` protocol link

4. **Tauri for native**
   - Same components
   - Button: Tauri command
   - Auto-run flow

---

## 🚀 Implementation

**Timeline: 3.5 weeks**

1. **Shared Components** (1 week)
   - Create `@rbee/marketplace-components`
   - Make components dumb
   - Test in Next.js and Tauri

2. **Next.js Site** (1 week)
   - Build marketplace.rbee.dev
   - SSG for models
   - Installation-aware button

3. **Tauri Integration** (1 week)
   - Protocol handler
   - Tauri commands
   - Auto-run flow

4. **Polish** (0.5 weeks)
   - Multi-hive support
   - Error handling
   - Testing

---

## ✅ Success Metrics

- Google search → Running model: **5 minutes**
- Returning user → Running model: **30 seconds**
- 1000+ model pages indexed
- "model name + rbee" rankings

---

## 🗑️ Why This Document is Outdated

**Old plan:**
- Embedded React components in Keeper
- No SEO
- No public marketplace
- Duplication between web and app

**New plan:**
- Separate Next.js site (SEO goldmine)
- Shared components (zero duplication)
- Tauri app (native performance)
- `rbee://` protocol (seamless integration)

**The new architecture is better in every way.**

---

## 📚 Read the New Docs

**Start here:** [COMPLETE_ONBOARDING_FLOW.md](./COMPLETE_ONBOARDING_FLOW.md)

Then read:
- [MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md](./MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md)
- [URL_SCHEME_PATTERN.md](./URL_SCHEME_PATTERN.md)
- [PROTOCOL_DETECTION_FALLBACK.md](./PROTOCOL_DETECTION_FALLBACK.md)
- [URL_PROTOCOL_REGISTRATION.md](./URL_PROTOCOL_REGISTRATION.md)

---

**Don't implement this old plan - use the new architecture!** 🚀
