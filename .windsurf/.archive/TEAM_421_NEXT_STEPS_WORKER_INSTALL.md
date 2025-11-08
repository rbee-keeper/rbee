# TEAM-421: Next Steps - Worker Installation & Next.js Integration

**Date:** 2025-11-06  
**Status:** Planning Phase

---

## Objectives

1. **Make worker installation button functional** in rbee-keeper GUI
2. **Implement worker detail pages in Next.js marketplace** (different from Tauri approach)
3. **Ensure consistency** between Tauri and Next.js implementations

---

## Part 1: Worker Installation Button (Tauri/rbee-keeper)

### Current State

**Location:** `bin/00_rbee_keeper/ui/src/pages/WorkerDetailsPage.tsx`

**Current Button (Line ~220):**
```tsx
<Button size="lg" className="min-w-[200px]">
  <Download className="size-4 mr-2" />
  Install Worker
</Button>
```

**Problem:** Button has no onClick handler - does nothing when clicked.

---

### Implementation Plan

#### Step 1: Add Tauri Command for Worker Installation

**File:** `bin/00_rbee_keeper/src/tauri_commands.rs`

**Add new command:**
```rust
/// Install a worker binary from the catalog
/// TEAM-421: Worker installation from marketplace
#[tauri::command]
#[specta::specta]
pub async fn marketplace_install_worker(
    worker_id: String,
) -> Result<String, String> {
    use observability_narration_core::n;
    
    n!("marketplace_install_worker", "📦 Installing worker: {}", worker_id);
    
    // TODO: Implement worker installation logic
    // 1. Fetch worker details from catalog
    // 2. Download worker binary
    // 3. Install to ~/.local/bin/ or appropriate location
    // 4. Set executable permissions
    // 5. Verify installation
    
    // For now, return a job_id for tracking
    let job_id = format!("install-worker-{}", worker_id);
    
    n!("marketplace_install_worker", "✅ Installation started: {}", job_id);
    Ok(job_id)
}
```

**Register command in main.rs:**
```rust
// Add to invoke_handler list
marketplace_install_worker,
```

#### Step 2: Update WorkerDetailsPage to Call Command

**File:** `bin/00_rbee_keeper/ui/src/pages/WorkerDetailsPage.tsx`

**Add installation handler:**
```tsx
import { invoke } from "@tauri-apps/api/core";
import { useState } from "react";

export function WorkerDetailsPage() {
  const [isInstalling, setIsInstalling] = useState(false);
  
  // ... existing code ...
  
  const handleInstall = async () => {
    if (!worker) return;
    
    setIsInstalling(true);
    try {
      const jobId = await invoke<string>("marketplace_install_worker", {
        workerId: worker.id,
      });
      
      console.log("Installation started:", jobId);
      // TODO: Show success notification
      // TODO: Track installation progress via job_id
      
    } catch (error) {
      console.error("Installation failed:", error);
      // TODO: Show error notification
    } finally {
      setIsInstalling(false);
    }
  };
  
  return (
    // ... existing JSX ...
    <Button 
      size="lg" 
      className="min-w-[200px]"
      onClick={handleInstall}
      disabled={isInstalling}
    >
      <Download className="size-4 mr-2" />
      {isInstalling ? "Installing..." : "Install Worker"}
    </Button>
  );
}
```

#### Step 3: Implement Actual Installation Logic

**Options for implementation:**

**Option A: Direct Download & Install**
- Download binary from `worker.pkgbuildUrl` or GitHub releases
- Save to `~/.local/bin/` (Linux/macOS) or `%LOCALAPPDATA%\Programs\` (Windows)
- Set executable permissions (`chmod +x`)
- Verify installation with `--version` command

**Option B: Use Package Manager (Future)**
- Generate PKGBUILD for Arch Linux
- Use `cargo install` for Rust workers
- Use platform-specific package managers

**Option C: Job-Based Installation (Recommended)**
- Create installation job in job queue
- Track progress with narration events
- Allow cancellation
- Show progress in UI

**Recommended: Option C** - Consistent with existing download system

---

## Part 2: Next.js Marketplace Worker Pages

### Current State

**Location:** `frontend/apps/marketplace/`

**Problem:** Next.js marketplace is for **browsing and discovery**, not for **installation**. It's a static site that can't invoke Tauri commands.

---

### Key Differences: Tauri vs Next.js

| Feature | Tauri (rbee-keeper) | Next.js (marketplace) |
|---------|---------------------|----------------------|
| **Purpose** | Local app, manage workers | Public website, browse catalog |
| **Installation** | ✅ Can install workers | ❌ Cannot install (no backend) |
| **Worker Catalog** | Fetches from Hono API | Fetches from Hono API |
| **Detail Pages** | Full details + install | Full details + "View in App" |
| **User Action** | Click "Install" → Downloads | Click "Open in App" → Deep link to Tauri |

---

### Implementation Plan for Next.js

#### Step 1: Create Worker Detail Page Route

**File:** `frontend/apps/marketplace/app/workers/[workerId]/page.tsx`

```tsx
// TEAM-421: Worker detail page for Next.js marketplace
// This is a STATIC page for SEO and discovery, not for installation

import { notFound } from 'next/navigation';

interface Props {
  params: Promise<{ workerId: string }>;
}

export async function generateStaticParams() {
  // Fetch workers from catalog API
  const response = await fetch('http://localhost:8787/workers');
  const data = await response.json();
  
  return data.workers.map((worker: any) => ({
    workerId: worker.id,
  }));
}

export default async function WorkerDetailPage({ params }: Props) {
  const { workerId } = await params;
  
  // Fetch worker details
  const response = await fetch(`http://localhost:8787/workers/${workerId}`);
  if (!response.ok) notFound();
  
  const worker = await response.json();
  
  return (
    <div className="container mx-auto px-4 py-12">
      <h1>{worker.name}</h1>
      <p>{worker.description}</p>
      
      {/* IMPORTANT: No "Install" button - this is a static site! */}
      {/* Instead: Link to open in rbee-keeper app */}
      
      <a 
        href={`rbee://install-worker/${worker.id}`}
        className="btn btn-primary"
      >
        Open in rbee App
      </a>
      
      {/* Show all worker details for SEO */}
      <WorkerDetailsTemplate worker={worker} />
    </div>
  );
}
```

#### Step 2: Add Deep Link Support (rbee:// protocol)

**File:** `bin/00_rbee_keeper/src/protocol.rs` (if exists) or create it

**Handle `rbee://install-worker/{workerId}` URLs:**
```rust
// TEAM-421: Handle deep links from Next.js marketplace
pub fn handle_protocol_url(url: &str) -> Result<(), String> {
    if url.starts_with("rbee://install-worker/") {
        let worker_id = url.strip_prefix("rbee://install-worker/").unwrap();
        
        // Emit event to navigate to worker detail page
        // and trigger installation
        emit_navigation_event(format!("/marketplace/rbee-workers/{}", worker_id));
        emit_install_event(worker_id.to_string());
        
        Ok(())
    } else {
        Err("Unknown protocol URL".to_string())
    }
}
```

**Register protocol handler in Tauri config:**
```json
// tauri.conf.json
{
  "tauri": {
    "bundle": {
      "identifier": "com.rbee.keeper",
      "deeplink": {
        "schemes": ["rbee"]
      }
    }
  }
}
```

#### Step 3: Create Worker List Page for Next.js

**File:** `frontend/apps/marketplace/app/workers/page.tsx`

```tsx
// TEAM-421: Worker catalog page for Next.js marketplace
// Static page for SEO, links to rbee-keeper for installation

export default async function WorkersPage() {
  const response = await fetch('http://localhost:8787/workers');
  const data = await response.json();
  
  return (
    <div className="container mx-auto px-4 py-12">
      <h1>rbee Workers</h1>
      <p>Browse available inference workers. Install via rbee app.</p>
      
      <div className="grid grid-cols-3 gap-6">
        {data.workers.map((worker: any) => (
          <WorkerCard 
            key={worker.id}
            worker={worker}
            onClick={() => {
              // Navigate to detail page
              window.location.href = `/workers/${worker.id}`;
            }}
          />
        ))}
      </div>
    </div>
  );
}
```

---

## Part 3: Architectural Decisions

### Why Different Approaches?

**Tauri (rbee-keeper):**
- ✅ Has access to filesystem
- ✅ Can download and install binaries
- ✅ Can execute system commands
- ✅ Can track installation progress
- ✅ User is already authenticated/trusted

**Next.js (marketplace):**
- ❌ No filesystem access (browser sandbox)
- ❌ Cannot install binaries
- ❌ Cannot execute system commands
- ✅ Great for SEO and discovery
- ✅ Can be hosted publicly
- ✅ Can link to Tauri app via deep links

### Recommended Flow

```
User Journey 1: Discovery → Installation
1. User browses marketplace.rbee.ai (Next.js)
2. Finds interesting worker
3. Clicks "Open in rbee App"
4. Deep link opens rbee-keeper
5. Worker detail page opens
6. User clicks "Install"
7. Worker is downloaded and installed

User Journey 2: Direct Installation
1. User opens rbee-keeper (Tauri)
2. Goes to Marketplace → Workers
3. Clicks worker card
4. Worker detail page opens
5. User clicks "Install"
6. Worker is downloaded and installed
```

---

## Implementation Checklist

### Phase 1: Tauri Worker Installation ⚠️ PRIORITY

- [ ] **1.1** Add `marketplace_install_worker` Tauri command
  - Create command in `tauri_commands.rs`
  - Register in `main.rs`
  - Add to specta bindings

- [ ] **1.2** Implement installation logic
  - Fetch worker details from catalog
  - Download binary (use existing download system)
  - Install to appropriate location
  - Set permissions
  - Verify installation

- [ ] **1.3** Update WorkerDetailsPage
  - Add `handleInstall` function
  - Add loading state
  - Add error handling
  - Show installation progress

- [ ] **1.4** Add installation progress tracking
  - Use narration events
  - Show progress bar
  - Allow cancellation

### Phase 2: Next.js Worker Pages 📋 FUTURE

- [ ] **2.1** Create worker list page
  - `app/workers/page.tsx`
  - Fetch from Hono API
  - Display worker cards
  - Link to detail pages

- [ ] **2.2** Create worker detail page
  - `app/workers/[workerId]/page.tsx`
  - SSG with `generateStaticParams`
  - Show full worker details
  - "Open in rbee App" button (deep link)

- [ ] **2.3** Add deep link support
  - Register `rbee://` protocol
  - Handle `rbee://install-worker/{id}`
  - Navigate to worker page in Tauri
  - Trigger installation

### Phase 3: Integration & Testing 🧪 FUTURE

- [ ] **3.1** Test Tauri installation flow
  - Install worker from GUI
  - Verify binary is downloaded
  - Verify permissions are set
  - Verify worker runs

- [ ] **3.2** Test Next.js → Tauri flow
  - Click "Open in rbee App" on Next.js
  - Verify deep link opens Tauri
  - Verify worker page opens
  - Verify installation works

- [ ] **3.3** Test error handling
  - Network failures
  - Permission errors
  - Invalid worker IDs
  - Duplicate installations

---

## Technical Notes

### Worker Installation Location

**Linux/macOS:**
- `~/.local/bin/` (user-local, no sudo required)
- Add to PATH if not already

**Windows:**
- `%LOCALAPPDATA%\Programs\rbee\workers\`
- Add to PATH

### Binary Verification

**After installation:**
1. Check file exists
2. Check file is executable
3. Run `{binary} --version` to verify
4. Update worker registry/catalog

### Progress Tracking

**Use narration events:**
```rust
n!("worker_install", "📥 Downloading worker binary...");
n!("worker_install", "📦 Installing to ~/.local/bin/...");
n!("worker_install", "🔒 Setting permissions...");
n!("worker_install", "✅ Installation complete!");
```

**Frontend listens to events and shows progress bar.**

---

## Open Questions

1. **Where to download binaries from?**
   - GitHub releases?
   - Custom CDN?
   - Build on-demand?

2. **How to handle updates?**
   - Check for newer versions?
   - Auto-update?
   - Manual update button?

3. **How to handle dependencies?**
   - CUDA libraries for GPU workers?
   - System packages?
   - Show requirements in UI?

4. **How to handle uninstallation?**
   - Remove binary?
   - Clean up config files?
   - Update registry?

---

## Success Criteria

### Tauri (rbee-keeper)
✅ Click "Install Worker" → Binary is downloaded and installed  
✅ Progress is shown during installation  
✅ Success/error notifications appear  
✅ Installed worker appears in worker list  
✅ Worker can be launched/used  

### Next.js (marketplace)
✅ Worker list page is accessible  
✅ Worker detail pages are pre-rendered (SSG)  
✅ "Open in rbee App" button works  
✅ Deep link opens Tauri app  
✅ SEO is optimized (meta tags, structured data)  

---

## Priority Order

1. **HIGH:** Tauri installation button (Phase 1.1-1.3)
2. **MEDIUM:** Installation progress tracking (Phase 1.4)
3. **LOW:** Next.js worker pages (Phase 2)
4. **LOW:** Deep link integration (Phase 2.3)

**Start with Phase 1.1** - Get the basic installation working in Tauri first!
