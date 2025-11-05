// TEAM-413: React hook for protocol events

import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'
import { useNavigate } from 'react-router-dom'

/**
 * Hook to listen for rbee:// protocol events from the backend
 * 
 * Events emitted by protocol.rs:
 * - 'protocol:install-model' - Model installation requested
 * - 'protocol:install-worker' - Worker installation requested
 * - 'navigate' - Navigation to a specific route
 */
export function useProtocol() {
  const navigate = useNavigate()

  useEffect(() => {
    // TEAM-413: Listen for model installation events
    const unlistenModelPromise = listen<string>('protocol:install-model', (event) => {
      console.log('📦 Protocol: Installing model:', event.payload)
      
      // Navigate to marketplace/models page
      navigate('/marketplace')
      
      // TODO: Show notification or trigger download UI
      // This could be integrated with a global notification system
    })

    // TEAM-413: Listen for worker installation events
    const unlistenWorkerPromise = listen<string>('protocol:install-worker', (event) => {
      console.log('👷 Protocol: Installing worker:', event.payload)
      
      // Navigate to marketplace page
      navigate('/marketplace')
      
      // TODO: Show notification or trigger worker spawn
    })

    // TEAM-413: Listen for general navigation events
    const unlistenNavigatePromise = listen<string>('navigate', (event) => {
      console.log('🧭 Protocol: Navigating to:', event.payload)
      
      // Navigate to the specified route
      navigate(event.payload)
    })

    // Cleanup listeners on unmount
    return () => {
      unlistenModelPromise.then((unlisten) => unlisten())
      unlistenWorkerPromise.then((unlisten) => unlisten())
      unlistenNavigatePromise.then((unlisten) => unlisten())
    }
  }, [navigate])
}
