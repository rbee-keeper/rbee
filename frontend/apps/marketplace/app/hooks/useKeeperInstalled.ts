'use client'

// TEAM-413: Client-side detection if Keeper is installed

import { useEffect, useState } from 'react'

export interface KeeperInstallationStatus {
  installed: boolean
  checking: boolean
}

export function useKeeperInstalled(): KeeperInstallationStatus {
  const [installed, setInstalled] = useState(false)
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    async function checkInstallation() {
      try {
        // TEAM-413: Method 1 - Try to detect rbee:// protocol support
        // This is a heuristic approach since browsers don't expose protocol handler info
        
        // Check if we're in a browser environment
        if (typeof window === 'undefined') {
          setChecking(false)
          return
        }

        // TEAM-413: Method 2 - Check localStorage for previous successful protocol use
        const previouslyDetected = localStorage.getItem('rbee-keeper-detected')
        if (previouslyDetected === 'true') {
          setInstalled(true)
          setChecking(false)
          return
        }

        // TEAM-413: Method 3 - Try to ping a local endpoint (if Keeper is running)
        try {
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 1000)
          
          await fetch('http://localhost:9200/health', {
            signal: controller.signal,
            mode: 'no-cors', // Avoid CORS issues
          })
          
          clearTimeout(timeoutId)
          
          // If we get any response, Keeper is likely running
          setInstalled(true)
          localStorage.setItem('rbee-keeper-detected', 'true')
        } catch {
          // Keeper not running
          setInstalled(false)
        }
      } catch (error) {
        console.error('Error checking Keeper installation:', error)
        setInstalled(false)
      } finally {
        setChecking(false)
      }
    }

    checkInstallation()
  }, [])

  return { installed, checking }
}
