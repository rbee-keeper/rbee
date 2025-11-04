# Worker Management Component

**TEAM-382: Clean MVP with card-based layout**  
**TEAM-405: Removed marketplace catalog - now focuses on LOCAL CATALOG management**

## Structure

```
WorkerManagement/
├── index.tsx                    # Main component (orchestration)
├── types.ts                     # Shared TypeScript types
├── WorkerCard.tsx               # Individual worker card
├── InstalledWorkersView.tsx     # Installed workers list
├── ActiveWorkersView.tsx        # Active workers grid
├── SpawnWorkerView.tsx          # Spawn worker form
├── README.md                    # This file
└── WorkerCatalogView.tsx        # ❌ REMOVED (TEAM-405) - use MarketplaceSearch
```

## Component Responsibilities

### `index.tsx` - Main Component
- **Role:** Orchestration and state management
- **State:** View mode (installed/active/spawn)
- **Responsibilities:**
  - Manage tabs (Installed, Active, Spawn)
  - Coordinate between child components
  - Handle worker operations (spawn, terminate)
- **Size:** ~120 lines (TEAM-405: reduced from 169)

### `types.ts` - Type Definitions
- **Role:** Shared TypeScript interfaces
- **Exports:**
  - `ViewMode` - Tab selection type ('installed' | 'active' | 'spawn')
  - `SpawnFormState` - Spawn form state
  - Re-exports `ProcessStats` from SDK (auto-generated from Rust)
- **Size:** ~16 lines (TEAM-405: removed 'catalog' from ViewMode)

### `WorkerCard.tsx` - Worker Card
- **Role:** Display individual worker metrics
- **Features:**
  - Status badge (Idle/Active)
  - GPU utilization progress bar
  - VRAM usage progress bar
  - CPU and RAM metrics
  - Uptime display
  - Terminate button (optional)
- **Size:** ~130 lines

### `InstalledWorkersView.tsx` - Installed Workers List
- **Role:** Display installed worker binaries from local catalog
- **Features:**
  - Lists workers from WorkerListInstalled operation
  - Shows type, version, size, path
  - Loading/error/empty states
- **Size:** ~150 lines (TEAM-382)

### `ActiveWorkersView.tsx` - Active Workers Grid
- **Role:** Display all active workers in a grid
- **Features:**
  - Loading skeleton
  - Error state
  - Empty state
  - Responsive grid (1-3 columns)
- **Size:** ~70 lines

### `SpawnWorkerView.tsx` - Spawn Worker Form
- **Role:** Form to spawn new workers
- **Features:**
  - Model selection dropdown
  - Worker type selection (CPU/CUDA/Metal)
  - Device ID input (for GPU workers)
  - Submit button with loading state
- **Size:** ~160 lines

### ❌ `WorkerCatalogView.tsx` - REMOVED (TEAM-405)
- **Reason:** Marketplace catalog moved to separate MarketplaceSearch component
- **Replacement:** Use `@rbee/marketplace-sdk` WorkerClient
- **Size:** 410 lines removed

## Usage

```tsx
import { WorkerManagement } from './components/WorkerManagement'

function App() {
  return <WorkerManagement />
}
```

## Key Differences from ModelManagement

### Card-Based Layout
- Workers displayed as cards (not table)
- Better for visual metrics (progress bars, badges)
- Easier to scan at a glance

### Real-Time Metrics
- GPU utilization percentage
- VRAM usage (used/total)
- CPU and RAM usage
- Uptime tracking

### Simpler Operations
- Only spawn operation (no load/unload/delete)
- Workers are ephemeral (terminate via card button)
- No search functionality (workers are local)

## Data Flow

```
useWorkers() → ProcessStats[] → ActiveWorkersView → WorkerCard[]
                                                    ↓
                                            GPU/VRAM/CPU metrics
                                            Status badges
                                            Progress bars

useModels() → ModelInfo[] → SpawnWorkerView → Model dropdown
                                              ↓
useHiveOperations() → spawnWorker() → Submit form
```

## Benefits of This Structure

### ✅ Visual Clarity
- Cards show metrics at a glance
- Progress bars for GPU/VRAM usage
- Color-coded status badges

### ✅ Responsive Design
- Grid adapts to screen size (1-3 columns)
- Cards stack on mobile
- Form is centered and max-width

### ✅ Clean MVP
- No unnecessary features
- Focus on core functionality (spawn, monitor)
- Easy to extend later

## Future Improvements

### Phase 2: Worker Operations
- [ ] Implement terminate worker operation
- [ ] Show worker logs in modal
- [ ] Real-time metric updates via SSE
- [ ] Worker health indicators

### Phase 3: Advanced Features
- [ ] Worker groups/pools
- [ ] Batch operations (spawn multiple)
- [ ] Worker templates (saved configurations)
- [ ] Performance history charts

## Dependencies

**UI Components:**
- `@rbee/ui/atoms` - Base components (Card, Button, Progress, etc.)

**Hooks:**
- `@rbee/rbee-hive-react` - Worker operations and data fetching
  - `useWorkers()` - Fetch worker list
  - `useHiveOperations()` - Spawn worker
  - `useModels()` - Get available models

**Icons:**
- `lucide-react` - Icon library

## Summary

This component provides a clean MVP for worker management:
- **Card-based layout** - Visual metrics at a glance
- **Simple operations** - Spawn workers, monitor performance
- **Responsive design** - Works on all screen sizes
- **Easy to extend** - Clear structure for future features

**TEAM-405 Update:**
- Before: ~885 lines split across 6 files (including WorkerCatalogView)
- After: ~475 lines split across 5 files
- **Removed:** 410 lines (WorkerCatalogView)
- **Reason:** Marketplace catalog moved to separate component using marketplace-sdk
