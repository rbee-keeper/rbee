# TEAM-421: Environment-Aware Action Handlers

**Date:** 2025-11-06  
**Status:** Analysis & Implementation Plan

---

## Question

> "The presentation layer should be data-layer aware. If we are in the Next.js environment, we open the `rbee://` script. And if we're in a Tauri environment, then we actually download according to the provisioner. Is that a thing already?"

---

## TL;DR Answer

**Partially exists, needs completion:**

✅ **Tauri (rbee-keeper):** Model download via `ModelProvisioner` EXISTS  
✅ **Tauri (rbee-keeper):** Tauri command `model_download` EXISTS  
❌ **Next.js (marketplace):** Deep link support (`rbee://`) DOES NOT EXIST  
❌ **Presentation Layer:** Environment detection NOT IMPLEMENTED  
❌ **Worker Installation:** Similar system DOES NOT EXIST

---

## Current State Analysis

### 1. Model Download (Tauri) ✅ EXISTS

**Backend:**
- **`ModelProvisioner`** exists in `bin/25_rbee_hive_crates/model-provisioner/`
- Downloads models from HuggingFace
- Caches to `~/.cache/rbee/models/`
- Handles progress tracking

**Tauri Command:**
```rust
// bin/00_rbee_keeper/src/tauri_commands.rs
#[tauri::command]
pub async fn model_download(
    hive_id: String,
    model_id: String,
) -> Result<String, String>
```

**Frontend (Tauri):**
```tsx
// bin/00_rbee_keeper/ui/src/pages/ModelDetailsPage.tsx
const handleDownload = async () => {
  const jobId = await invoke("model_download", {
    hiveId: "localhost",
    modelId: model.id,
  });
};
```

**Status:** ✅ **FULLY FUNCTIONAL**

---

### 2. Worker Installation (Tauri) ❌ DOES NOT EXIST

**Backend:**
- No `WorkerProvisioner` equivalent
- No `worker_install` Tauri command implementation
- `worker_download` command exists but incomplete

**Frontend:**
- Install button in `WorkerDetailsPage` has no onClick handler
- No progress tracking
- No error handling

**Status:** ❌ **NOT IMPLEMENTED** (see `TEAM_421_NEXT_STEPS_WORKER_INSTALL.md`)

---

### 3. Deep Links (Next.js) ❌ DOES NOT EXIST

**What's Needed:**
- `rbee://` protocol registration
- Deep link handler in Tauri
- "Open in rbee App" buttons in Next.js marketplace

**Current State:**
- No protocol registration in `tauri.conf.json`
- No deep link handler
- Next.js marketplace has no "Open in App" buttons

**Status:** ❌ **NOT IMPLEMENTED**

---

### 4. Environment Detection ❌ NOT IMPLEMENTED

**What's Needed:**
- Utility to detect Tauri vs Next.js vs Browser
- Conditional action handlers based on environment
- Unified API that works in all environments

**Current State:**
- No environment detection utility
- No conditional action handlers
- Each environment has separate implementations

**Status:** ❌ **NOT IMPLEMENTED**

---

## Proposed Architecture

### Environment Detection Utility

**Location:** `frontend/packages/rbee-ui/src/utils/environment.ts`

```typescript
// TEAM-421: Environment detection for action handlers

/**
 * Detect if running in Tauri (rbee-keeper)
 */
export function isTauriEnvironment(): boolean {
  return typeof window !== 'undefined' && '__TAURI__' in window;
}

/**
 * Detect if running in Next.js (marketplace)
 */
export function isNextJsEnvironment(): boolean {
  return typeof window !== 'undefined' && 
         typeof window.next !== 'undefined';
}

/**
 * Detect if running in browser (not Tauri, not Next.js SSR)
 */
export function isBrowserEnvironment(): boolean {
  return typeof window !== 'undefined' && 
         !isTauriEnvironment();
}

/**
 * Get current environment type
 */
export type Environment = 'tauri' | 'nextjs' | 'browser' | 'ssr';

export function getEnvironment(): Environment {
  if (typeof window === 'undefined') return 'ssr';
  if (isTauriEnvironment()) return 'tauri';
  if (isNextJsEnvironment()) return 'nextjs';
  return 'browser';
}
```

---

### Environment-Aware Action Handlers

**Location:** `frontend/packages/rbee-ui/src/marketplace/hooks/useArtifactActions.ts`

```typescript
// TEAM-421: Environment-aware action handlers for artifacts

import { invoke } from '@tauri-apps/api/core';
import { getEnvironment } from '../../utils/environment';

export interface ArtifactActionHandlers {
  downloadModel: (modelId: string) => Promise<void>;
  installWorker: (workerId: string) => Promise<void>;
  viewOnPlatform: (url: string) => void;
}

/**
 * Get environment-appropriate action handlers
 */
export function useArtifactActions(): ArtifactActionHandlers {
  const env = getEnvironment();

  return {
    downloadModel: async (modelId: string) => {
      switch (env) {
        case 'tauri':
          // Direct download via Tauri command
          await invoke('model_download', {
            hiveId: 'localhost',
            modelId,
          });
          break;

        case 'nextjs':
        case 'browser':
          // Open deep link to rbee-keeper
          window.location.href = `rbee://download-model/${encodeURIComponent(modelId)}`;
          break;

        case 'ssr':
          throw new Error('Cannot download in SSR environment');
      }
    },

    installWorker: async (workerId: string) => {
      switch (env) {
        case 'tauri':
          // Direct installation via Tauri command
          await invoke('marketplace_install_worker', {
            workerId,
          });
          break;

        case 'nextjs':
        case 'browser':
          // Open deep link to rbee-keeper
          window.location.href = `rbee://install-worker/${encodeURIComponent(workerId)}`;
          break;

        case 'ssr':
          throw new Error('Cannot install in SSR environment');
      }
    },

    viewOnPlatform: (url: string) => {
      if (env === 'ssr') return;
      window.open(url, '_blank', 'noopener,noreferrer');
    },
  };
}
```

---

### Update ArtifactDetailPageTemplate

**Make it environment-aware:**

```typescript
// frontend/packages/rbee-ui/src/marketplace/templates/ArtifactDetailPageTemplate/ArtifactDetailPageTemplate.tsx

import { useArtifactActions } from '../../hooks/useArtifactActions';
import { getEnvironment } from '../../../utils/environment';

export function ArtifactDetailPageTemplate({
  // ... existing props
  artifactType, // 'model' | 'worker'
  artifactId,
}: ArtifactDetailPageTemplateProps) {
  const actions = useArtifactActions();
  const env = getEnvironment();

  // Determine button label based on environment
  const primaryButtonLabel = env === 'tauri' 
    ? (artifactType === 'model' ? 'Download Model' : 'Install Worker')
    : 'Open in rbee App';

  // Handle primary action
  const handlePrimaryAction = async () => {
    if (artifactType === 'model') {
      await actions.downloadModel(artifactId);
    } else if (artifactType === 'worker') {
      await actions.installWorker(artifactId);
    }
  };

  return (
    // ... existing JSX with updated button
    <Button onClick={handlePrimaryAction}>
      {primaryButtonLabel}
    </Button>
  );
}
```

---

## Implementation Plan

### Phase 1: Environment Detection ⚠️ PRIORITY

- [ ] **1.1** Create `frontend/packages/rbee-ui/src/utils/environment.ts`
  - `isTauriEnvironment()`
  - `isNextJsEnvironment()`
  - `getEnvironment()`

- [ ] **1.2** Add tests for environment detection
  - Mock `window.__TAURI__`
  - Mock `window.next`
  - Test all environment types

### Phase 2: Deep Link Support (Tauri)

- [ ] **2.1** Register `rbee://` protocol in Tauri
  - Update `tauri.conf.json`
  - Add `deeplink` configuration

- [ ] **2.2** Implement deep link handler
  - Create `bin/00_rbee_keeper/src/protocol.rs`
  - Handle `rbee://download-model/{id}`
  - Handle `rbee://install-worker/{id}`
  - Emit navigation events

- [ ] **2.3** Test deep links
  - Click link in browser
  - Verify Tauri app opens
  - Verify correct page opens
  - Verify action triggers

### Phase 3: Environment-Aware Actions

- [ ] **3.1** Create `useArtifactActions` hook
  - Implement `downloadModel()`
  - Implement `installWorker()`
  - Implement `viewOnPlatform()`

- [ ] **3.2** Update `ArtifactDetailPageTemplate`
  - Accept `artifactType` and `artifactId` props
  - Use `useArtifactActions` hook
  - Conditional button labels
  - Conditional button actions

- [ ] **3.3** Update `ModelDetailPageTemplate`
  - Pass `artifactType="model"`
  - Pass `artifactId={model.id}`

- [ ] **3.4** Update `WorkerDetailsPage`
  - Pass `artifactType="worker"`
  - Pass `artifactId={worker.id}`

### Phase 4: Next.js Marketplace Integration

- [ ] **4.1** Add "Open in rbee App" buttons
  - Model detail pages
  - Worker detail pages
  - Use deep links

- [ ] **4.2** Add "Install rbee" fallback
  - Detect if rbee-keeper is installed
  - Show install instructions if not
  - Link to download page

- [ ] **4.3** Test Next.js → Tauri flow
  - Browse marketplace on web
  - Click "Open in rbee App"
  - Verify app opens
  - Verify action completes

### Phase 5: Worker Installation (Tauri)

- [ ] **5.1** Implement `marketplace_install_worker` command
  - Download worker binary
  - Install to `~/.local/bin/`
  - Set permissions
  - Verify installation

- [ ] **5.2** Add progress tracking
  - Use narration events
  - Show progress bar in UI
  - Allow cancellation

- [ ] **5.3** Update `WorkerDetailsPage`
  - Use environment-aware actions
  - Show installation progress
  - Handle errors

---

## User Flows

### Flow 1: Download Model (Tauri)

```
User in rbee-keeper
    ↓
Navigate to Model Detail Page
    ↓
Click "Download Model"
    ↓
invoke('model_download', { modelId })
    ↓
ModelProvisioner downloads from HuggingFace
    ↓
Progress shown via narration events
    ↓
Model cached to ~/.cache/rbee/models/
    ↓
Success notification
```

### Flow 2: Download Model (Next.js)

```
User on marketplace.rbee.ai
    ↓
Navigate to Model Detail Page
    ↓
Click "Open in rbee App"
    ↓
window.location.href = 'rbee://download-model/{id}'
    ↓
Browser asks "Open rbee-keeper?"
    ↓
User clicks "Open"
    ↓
rbee-keeper opens
    ↓
Model Detail Page opens
    ↓
Download starts automatically
    ↓
(Same as Flow 1 from here)
```

### Flow 3: Install Worker (Tauri)

```
User in rbee-keeper
    ↓
Navigate to Worker Detail Page
    ↓
Click "Install Worker"
    ↓
invoke('marketplace_install_worker', { workerId })
    ↓
WorkerProvisioner downloads binary
    ↓
Binary installed to ~/.local/bin/
    ↓
Permissions set (chmod +x)
    ↓
Worker verified (--version check)
    ↓
Success notification
```

### Flow 4: Install Worker (Next.js)

```
User on marketplace.rbee.ai
    ↓
Navigate to Worker Detail Page
    ↓
Click "Open in rbee App"
    ↓
window.location.href = 'rbee://install-worker/{id}'
    ↓
Browser asks "Open rbee-keeper?"
    ↓
User clicks "Open"
    ↓
rbee-keeper opens
    ↓
Worker Detail Page opens
    ↓
Installation starts automatically
    ↓
(Same as Flow 3 from here)
```

---

## Technical Decisions

### Why Deep Links?

**Pros:**
- ✅ Standard way to open desktop apps from browser
- ✅ Works on all platforms (Linux, macOS, Windows)
- ✅ No backend required
- ✅ User stays in control (browser asks permission)

**Cons:**
- ❌ Requires protocol registration
- ❌ Browser shows "Open app?" dialog
- ❌ Doesn't work if app not installed

### Why Environment Detection?

**Pros:**
- ✅ Single codebase for all environments
- ✅ Automatic behavior based on context
- ✅ No manual configuration needed
- ✅ Easy to test (mock environment)

**Cons:**
- ❌ Adds complexity
- ❌ Requires careful testing
- ❌ May be overkill for simple cases

### Alternative: Separate Implementations

**Instead of environment detection, we could:**
- Have separate components for Tauri vs Next.js
- Duplicate code between environments
- Manually choose which component to use

**Why we're NOT doing this:**
- ❌ Code duplication
- ❌ Harder to maintain
- ❌ Inconsistent behavior
- ❌ More prone to bugs

---

## Success Criteria

### Environment Detection ✅
- [ ] Correctly detects Tauri environment
- [ ] Correctly detects Next.js environment
- [ ] Correctly detects browser environment
- [ ] Correctly detects SSR environment

### Deep Links ✅
- [ ] `rbee://` protocol registered
- [ ] Deep links open rbee-keeper
- [ ] Correct page opens in app
- [ ] Action triggers automatically

### Model Download ✅
- [ ] Works in Tauri (direct download)
- [ ] Works in Next.js (deep link)
- [ ] Progress shown in both environments
- [ ] Errors handled in both environments

### Worker Installation ✅
- [ ] Works in Tauri (direct install)
- [ ] Works in Next.js (deep link)
- [ ] Progress shown in both environments
- [ ] Errors handled in both environments

### User Experience ✅
- [ ] Buttons have correct labels per environment
- [ ] Actions work as expected
- [ ] No confusing error messages
- [ ] Smooth transitions between environments

---

## Current Status Summary

| Feature | Tauri | Next.js | Status |
|---------|-------|---------|--------|
| **Model Download** | ✅ Implemented | ❌ Missing | 50% |
| **Worker Install** | ❌ Missing | ❌ Missing | 0% |
| **Deep Links** | ❌ Missing | ❌ Missing | 0% |
| **Environment Detection** | ❌ Missing | ❌ Missing | 0% |
| **Unified Actions** | ❌ Missing | ❌ Missing | 0% |

**Overall Progress:** ~10% complete

---

## Next Steps (Priority Order)

1. **HIGH:** Implement environment detection utility
2. **HIGH:** Create `useArtifactActions` hook
3. **HIGH:** Update `ArtifactDetailPageTemplate` to use hook
4. **MEDIUM:** Register `rbee://` protocol in Tauri
5. **MEDIUM:** Implement deep link handler
6. **MEDIUM:** Add "Open in rbee App" buttons to Next.js
7. **LOW:** Implement worker installation in Tauri
8. **LOW:** Add progress tracking for all actions

**Start with Phase 1.1** - Environment detection is the foundation for everything else!
