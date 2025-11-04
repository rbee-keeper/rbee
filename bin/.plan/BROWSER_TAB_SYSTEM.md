# Browser-Like Tab System for Bee Keeper

**Date:** 2025-11-04  
**Status:** 🎯 ARCHITECTURE DESIGN  
**Purpose:** Handle multiple routes/pages simultaneously like a browser

---

## 🎯 The Goal

**Problem:** React Router only shows ONE route at a time.  
**Need:** Show MULTIPLE routes simultaneously (like browser tabs).  
**Solution:** Tab system with Zustand state management + dynamic routing.

---

## 🔍 Research Findings

### React Libraries Found

**1. react-tabtab** ✅ RECOMMENDED
- GitHub: https://github.com/ctxhou/react-tabtab
- Features:
  - ✅ Draggable tabs (reorder)
  - ✅ Closeable tabs
  - ✅ Add new tab
  - ✅ Mobile support
  - ✅ Custom styling
- Perfect for browser-like tabs!

**2. dnd-kit** ✅ USE FOR DRAG & DROP
- Website: https://dndkit.com/
- Features:
  - ✅ Modern drag & drop
  - ✅ Performant
  - ✅ Accessible
  - ✅ Works with React 18+
- Use for tab reordering

**3. rc-tabs**
- React Component tabs
- More complex, overkill for our needs

### Architecture Pattern

**Zustand + Dynamic Routing + Tab Component**

**Don't use React Router's Routes:**
```tsx
// ❌ OLD WAY (only one route at a time)
<Routes>
  <Route path="/" element={<HomePage />} />
  <Route path="/queen" element={<QueenPage />} />
</Routes>
```

**Use Zustand to manage tabs:**
```tsx
// ✅ NEW WAY (multiple "routes" as tabs)
<TabSystem>
  {tabs.map(tab => (
    <TabPane key={tab.id}>
      {renderContent(tab.route)}
    </TabPane>
  ))}
</TabSystem>
```

---

## 🏗️ Architecture Design

### 1. Zustand Store

**File:** `src/store/tabStore.ts`

```typescript
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface Tab {
  id: string
  type: 'home' | 'queen' | 'hive' | 'marketplace' | 'worker' | 'settings'
  title: string
  icon: string
  route: string
  
  // Optional metadata
  hiveId?: string        // For hive tabs
  workerId?: string      // For worker tabs
  marketplaceType?: 'huggingface' | 'civitai' | 'workers'  // For marketplace tabs
  
  // Tab state
  closeable: boolean
  pinned: boolean
  dirty: boolean         // Has unsaved changes
}

interface TabStore {
  // State
  tabs: Tab[]
  activeTabId: string | null
  layout: 'single' | 'split' | 'grid'
  
  // Actions
  addTab: (tab: Omit<Tab, 'id'>) => string
  removeTab: (tabId: string) => void
  setActiveTab: (tabId: string) => void
  updateTab: (tabId: string, updates: Partial<Tab>) => void
  reorderTabs: (fromIndex: number, toIndex: number) => void
  closeAllTabs: () => void
  closeOtherTabs: (tabId: string) => void
  duplicateTab: (tabId: string) => void
  
  // Layout
  setLayout: (layout: 'single' | 'split' | 'grid') => void
  autoLayout: () => void  // Auto-detect best layout based on tab count
}

export const useTabStore = create<TabStore>()(
  persist(
    (set, get) => ({
      tabs: [],
      activeTabId: null,
      layout: 'single',
      
      addTab: (tabData) => {
        const id = `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        const newTab: Tab = {
          ...tabData,
          id,
          closeable: tabData.closeable ?? true,
          pinned: tabData.pinned ?? false,
          dirty: false,
        }
        
        set(state => ({
          tabs: [...state.tabs, newTab],
          activeTabId: id,
        }))
        
        get().autoLayout()
        return id
      },
      
      removeTab: (tabId) => {
        set(state => {
          const tabs = state.tabs.filter(t => t.id !== tabId)
          let activeTabId = state.activeTabId
          
          // If removing active tab, select another
          if (activeTabId === tabId) {
            activeTabId = tabs.length > 0 ? tabs[tabs.length - 1].id : null
          }
          
          return { tabs, activeTabId }
        })
        
        get().autoLayout()
      },
      
      setActiveTab: (tabId) => {
        set({ activeTabId: tabId })
      },
      
      updateTab: (tabId, updates) => {
        set(state => ({
          tabs: state.tabs.map(tab =>
            tab.id === tabId ? { ...tab, ...updates } : tab
          )
        }))
      },
      
      reorderTabs: (fromIndex, toIndex) => {
        set(state => {
          const tabs = [...state.tabs]
          const [removed] = tabs.splice(fromIndex, 1)
          tabs.splice(toIndex, 0, removed)
          return { tabs }
        })
      },
      
      closeAllTabs: () => {
        set({ tabs: [], activeTabId: null, layout: 'single' })
      },
      
      closeOtherTabs: (tabId) => {
        set(state => ({
          tabs: state.tabs.filter(t => t.id === tabId || t.pinned),
          activeTabId: tabId,
        }))
        get().autoLayout()
      },
      
      duplicateTab: (tabId) => {
        const tab = get().tabs.find(t => t.id === tabId)
        if (tab) {
          get().addTab({ ...tab, title: `${tab.title} (Copy)` })
        }
      },
      
      setLayout: (layout) => {
        set({ layout })
      },
      
      autoLayout: () => {
        const count = get().tabs.length
        if (count === 0) {
          set({ layout: 'single' })
        } else if (count === 1) {
          set({ layout: 'single' })
        } else if (count === 2) {
          set({ layout: 'split' })
        } else {
          set({ layout: 'grid' })
        }
      },
    }),
    {
      name: 'bee-keeper-tabs',
      partialize: (state) => ({
        tabs: state.tabs,
        activeTabId: state.activeTabId,
        layout: state.layout,
      }),
    }
  )
)
```

### 2. Tab Bar Component

**File:** `src/components/TabBar.tsx`

```tsx
import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { X, Pin, Copy } from 'lucide-react'
import { useTabStore } from '@/store/tabStore'
import type { Tab } from '@/store/tabStore'

interface SortableTabProps {
  tab: Tab
  isActive: boolean
}

function SortableTab({ tab, isActive }: SortableTabProps) {
  const { setActiveTab, removeTab, updateTab, duplicateTab } = useTabStore()
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: tab.id })
  
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  }
  
  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={`
        tab
        ${isActive ? 'active' : ''}
        ${tab.pinned ? 'pinned' : ''}
        ${tab.dirty ? 'dirty' : ''}
      `}
      onClick={() => setActiveTab(tab.id)}
      onContextMenu={(e) => {
        e.preventDefault()
        // Show context menu
      }}
    >
      <span className="tab-icon">{tab.icon}</span>
      <span className="tab-title">{tab.title}</span>
      
      {tab.dirty && <span className="dirty-indicator">●</span>}
      
      {tab.pinned && (
        <Pin 
          className="pin-icon" 
          size={12}
          onClick={(e) => {
            e.stopPropagation()
            updateTab(tab.id, { pinned: false })
          }}
        />
      )}
      
      {tab.closeable && !tab.pinned && (
        <X
          className="close-icon"
          size={14}
          onClick={(e) => {
            e.stopPropagation()
            removeTab(tab.id)
          }}
        />
      )}
    </div>
  )
}

export function TabBar() {
  const { tabs, activeTabId, reorderTabs } = useTabStore()
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )
  
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    
    if (over && active.id !== over.id) {
      const oldIndex = tabs.findIndex(t => t.id === active.id)
      const newIndex = tabs.findIndex(t => t.id === over.id)
      reorderTabs(oldIndex, newIndex)
    }
  }
  
  return (
    <div className="tab-bar">
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={tabs.map(t => t.id)}
          strategy={horizontalListSortingStrategy}
        >
          <div className="tabs-container">
            {tabs.map(tab => (
              <SortableTab
                key={tab.id}
                tab={tab}
                isActive={tab.id === activeTabId}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>
      
      <button className="new-tab-button" onClick={() => {
        // Open new tab menu
      }}>
        +
      </button>
    </div>
  )
}
```

### 3. Tab Content Renderer

**File:** `src/components/TabContent.tsx`

```tsx
import { useTabStore } from '@/store/tabStore'
import type { Tab } from '@/store/tabStore'

// Import all page components
import { ServicesPage } from '@/pages/ServicesPage'
import { QueenPage } from '@/pages/QueenPage'
import { HivePage } from '@/pages/HivePage'
import { MarketplacePage } from '@/pages/MarketplacePage'
import { WorkerPage } from '@/pages/WorkerPage'
import { SettingsPage } from '@/pages/SettingsPage'

function renderTabContent(tab: Tab) {
  switch (tab.type) {
    case 'home':
      return <ServicesPage />
    
    case 'queen':
      return <QueenPage />
    
    case 'hive':
      return <HivePage hiveId={tab.hiveId!} />
    
    case 'marketplace':
      return <MarketplacePage type={tab.marketplaceType!} />
    
    case 'worker':
      return <WorkerPage workerId={tab.workerId!} />
    
    case 'settings':
      return <SettingsPage />
    
    default:
      return <div>Unknown tab type: {tab.type}</div>
  }
}

export function TabContent() {
  const { tabs, activeTabId, layout } = useTabStore()
  
  if (tabs.length === 0) {
    return (
      <div className="empty-state">
        <h2>No tabs open</h2>
        <p>Click + to open a new tab</p>
      </div>
    )
  }
  
  if (layout === 'single') {
    const activeTab = tabs.find(t => t.id === activeTabId)
    if (!activeTab) return null
    
    return (
      <div className="tab-content single">
        {renderTabContent(activeTab)}
      </div>
    )
  }
  
  if (layout === 'split') {
    const [tab1, tab2] = tabs.slice(0, 2)
    
    return (
      <div className="tab-content split">
        <div className="pane left">
          {renderTabContent(tab1)}
        </div>
        <div className="pane right">
          {renderTabContent(tab2)}
        </div>
      </div>
    )
  }
  
  if (layout === 'grid') {
    return (
      <div className="tab-content grid">
        {tabs.slice(0, 4).map(tab => (
          <div key={tab.id} className="pane">
            {renderTabContent(tab)}
          </div>
        ))}
      </div>
    )
  }
  
  return null
}
```

### 4. New App Structure

**File:** `src/App.tsx` (REPLACEMENT)

```tsx
import { useEffect } from 'react'
import { BrowserRouter } from 'react-router-dom'
import { SidebarProvider } from '@/components/ui/sidebar'
import { Shell } from '@/components/Shell'
import { TabBar } from '@/components/TabBar'
import { TabContent } from '@/components/TabContent'
import { setupNarrationListener } from '@/lib/narration'
import { broadcastThemeChanges } from '@/lib/theme'
import { useTabStore } from '@/store/tabStore'

function App() {
  const { tabs, addTab } = useTabStore()
  
  // Setup listeners
  useEffect(() => {
    const cleanupNarration = setupNarrationListener()
    return cleanupNarration
  }, [])
  
  useEffect(() => {
    const cleanupTheme = broadcastThemeChanges()
    return cleanupTheme
  }, [])
  
  // Initialize with home tab if no tabs exist
  useEffect(() => {
    if (tabs.length === 0) {
      addTab({
        type: 'home',
        title: 'Services',
        icon: '🏠',
        route: '/',
        closeable: false,
        pinned: true,
      })
    }
  }, [])
  
  return (
    <BrowserRouter>
      <SidebarProvider>
        <Shell>
          <div className="app-content">
            <TabBar />
            <TabContent />
          </div>
        </Shell>
      </SidebarProvider>
    </BrowserRouter>
  )
}

export default App
```

---

## 🎯 Key Features

### 1. Tab Actions

**Right-click context menu:**
```
┌─────────────────────┐
│ Pin Tab             │
│ Duplicate Tab       │
│ ───────────────     │
│ Close Tab           │
│ Close Other Tabs    │
│ Close All Tabs      │
│ ───────────────     │
│ New Tab to Right    │
└─────────────────────┘
```

### 2. Keyboard Shortcuts

```
Ctrl + T        → New tab
Ctrl + W        → Close current tab
Ctrl + Tab      → Next tab
Ctrl + Shift + Tab → Previous tab
Ctrl + 1-9      → Switch to tab 1-9
Ctrl + Shift + T → Reopen closed tab
```

### 3. Tab Persistence

**Auto-save to localStorage:**
- Which tabs are open
- Which tab is active
- Tab order
- Pinned state

**On reload:** Restore all tabs exactly as they were!

### 4. Dirty State

**Track unsaved changes:**
```tsx
// Mark tab as dirty
updateTab(tabId, { dirty: true })

// Show dot indicator
{tab.dirty && <span className="dirty-indicator">●</span>}

// Confirm before closing
if (tab.dirty) {
  const confirmed = confirm('You have unsaved changes. Close anyway?')
  if (!confirmed) return
}
```

---

## 🎨 Styling

### CSS Structure

```css
/* Tab Bar */
.tab-bar {
  display: flex;
  align-items: center;
  height: 40px;
  background: var(--background);
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
  overflow-y: hidden;
}

.tabs-container {
  display: flex;
  flex: 1;
  gap: 2px;
  padding: 0 4px;
}

/* Individual Tab */
.tab {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--muted);
  border-radius: 8px 8px 0 0;
  cursor: pointer;
  user-select: none;
  transition: all 0.2s;
}

.tab.active {
  background: var(--background);
  border-bottom: 2px solid var(--primary);
}

.tab.pinned {
  padding-left: 8px;
}

.tab:hover .close-icon {
  opacity: 1;
}

/* Tab Content */
.tab-content {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.tab-content.single {
  width: 100%;
}

.tab-content.split {
  gap: 1px;
  background: var(--border);
}

.tab-content.split .pane {
  flex: 1;
  background: var(--background);
  overflow: auto;
}

.tab-content.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 1px;
  background: var(--border);
}

.tab-content.grid .pane {
  background: var(--background);
  overflow: auto;
}
```

---

## 🔗 Integration with Sidebar

### Update Sidebar Links

**File:** `src/components/KeeperSidebar.tsx`

```tsx
import { useTabStore } from '@/store/tabStore'

export function KeeperSidebar() {
  const { addTab } = useTabStore()
  
  const openTab = (type: Tab['type'], title: string, icon: string, route: string, metadata?: any) => {
    addTab({
      type,
      title,
      icon,
      route,
      ...metadata,
    })
  }
  
  return (
    <div className="sidebar">
      {/* Replace <Link> with onClick */}
      <button onClick={() => openTab('home', 'Services', '🏠', '/')}>
        <HomeIcon />
        <span>Services</span>
      </button>
      
      <button onClick={() => openTab('queen', 'Queen', '👑', '/queen')}>
        <CrownIcon />
        <span>Queen</span>
      </button>
      
      {/* Marketplaces */}
      <button onClick={() => openTab('marketplace', 'HuggingFace', '🤗', '/marketplace/huggingface', {
        marketplaceType: 'huggingface'
      })}>
        <PackageIcon />
        <span>HuggingFace</span>
      </button>
      
      {/* etc */}
    </div>
  )
}
```

---

## 📦 Dependencies

**Add to package.json:**
```json
{
  "dependencies": {
    "zustand": "^4.5.0",
    "@dnd-kit/core": "^6.1.0",
    "@dnd-kit/sortable": "^8.0.0",
    "@dnd-kit/utilities": "^3.2.2"
  }
}
```

---

## 🚀 Implementation Steps

### Phase 1: Core Tab System (2 days)

1. Install dependencies
2. Create Zustand store
3. Create TabBar component
4. Create TabContent component
5. Update App.tsx
6. Test basic tab creation/closing

### Phase 2: Drag & Drop (1 day)

1. Add dnd-kit integration
2. Implement tab reordering
3. Test drag & drop

### Phase 3: Layouts (1 day)

1. Implement split layout
2. Implement grid layout
3. Auto-layout logic
4. Test all layouts

### Phase 4: Polish (1 day)

1. Keyboard shortcuts
2. Context menu
3. Dirty state tracking
4. Persistence
5. Animations

**Total: 5 days**

---

## 🎯 Success Criteria

✅ Can open multiple tabs  
✅ Can reorder tabs by dragging  
✅ Can close tabs (with unsaved warning)  
✅ Can pin tabs  
✅ Split-screen works (2 tabs)  
✅ Grid works (3-4 tabs)  
✅ Tabs persist across reloads  
✅ Keyboard shortcuts work  
✅ Context menu works  
✅ Smooth animations  
✅ No performance issues  

---

**This is how we handle multiple routes like a browser!** 🚀
