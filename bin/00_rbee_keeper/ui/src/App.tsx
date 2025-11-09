// TEAM-295: rbee-keeper GUI - Main application component with routing
// TEAM-334: Uses Shell component for layout (titlebar + sidebar + content)
// TEAM-340: Added Queen page with iframe
// TEAM-342: Added Hive page with dynamic iframe
// TEAM-XXX: Added narration listener for Queen iframe events
// TEAM-350: Log build mode on startup
// TEAM-405: Added Marketplace pages (LLM Models, Image Models, Rbee Workers)
// TEAM-423: Renamed to HuggingFace and Civitai for clarity
// TEAM-413: Added protocol listener for rbee:// URL handling

import { broadcastThemeChanges } from '@rbee/iframe-bridge'
import { SidebarProvider } from '@rbee/ui/atoms'
import { useEffect } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Shell } from './components/Shell'
import { useProtocol } from './hooks/useProtocol'
import HelpPage from './pages/HelpPage'
import HivePage from './pages/HivePage'
import { MarketplaceCivitai } from './pages/MarketplaceCivitai'
import { MarketplaceHuggingFace } from './pages/MarketplaceHuggingFace'
import { MarketplaceRbeeWorkers } from './pages/MarketplaceRbeeWorkers'
import { ModelDetailsPage } from './pages/ModelDetailsPage'
import QueenPage from './pages/QueenPage'
import KeeperPage from './pages/ServicesPage'
import SettingsPage from './pages/SettingsPage'
import { WorkerDetailsPage } from './pages/WorkerDetailsPage'
import { setupNarrationListener } from './utils/narrationListener'

// TEAM-350: Log build mode on startup
const isDev = import.meta.env.DEV
if (isDev) {
  console.log('🔧 [KEEPER UI] Running in DEVELOPMENT mode')
  console.log('   - Vite dev server active (hot reload enabled)')
  console.log('   - Running on: http://localhost:5173')
} else {
  console.log('🚀 [KEEPER UI] Running in PRODUCTION mode')
  console.log('   - Tauri app (embedded)')
}

// TEAM-XXX: Inner component that has Router context
function AppRoutes() {
  // TEAM-XXX: Setup listener for narration events from Queen iframe
  useEffect(() => {
    const cleanupNarration = setupNarrationListener()
    return cleanupNarration
  }, [])

  // TEAM-375: Broadcast theme changes to iframes (Queen, Hive)
  useEffect(() => {
    const cleanupTheme = broadcastThemeChanges()
    return cleanupTheme
  }, [])

  // TEAM-413: Listen for rbee:// protocol events (MUST be inside Router)
  useProtocol()

  return (
    <SidebarProvider>
      <Shell>
        <Routes>
          <Route path="/" element={<KeeperPage />} />
          <Route path="/queen" element={<QueenPage />} />
          <Route path="/hive/:hiveId" element={<HivePage />} />
          <Route path="/marketplace/huggingface" element={<MarketplaceHuggingFace />} />
          <Route path="/marketplace/huggingface/:modelId" element={<ModelDetailsPage />} />
          <Route path="/marketplace/civitai" element={<MarketplaceCivitai />} />
          <Route path="/marketplace/rbee-workers" element={<MarketplaceRbeeWorkers />} />
          <Route path="/marketplace/rbee-workers/:workerId" element={<WorkerDetailsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/help" element={<HelpPage />} />
        </Routes>
      </Shell>
    </SidebarProvider>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  )
}

export default App
