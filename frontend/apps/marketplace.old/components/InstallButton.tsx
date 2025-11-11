'use client'

// TEAM-427: Install button using rbee-ui Button atom
// Refactored from ad-hoc implementation to use proper atomic design

import { Button, Spinner } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import { Download, ExternalLink } from 'lucide-react'
import { useKeeperInstalled } from '../app/hooks/useKeeperInstalled'

interface InstallButtonProps {
  modelId: string
  className?: string
}

/**
 * Install button with Keeper detection
 * Uses rbee-ui Button atom following atomic design pattern
 */
export function InstallButton({ modelId, className }: InstallButtonProps) {
  const { installed, checking } = useKeeperInstalled()

  // Checking state
  if (checking) {
    return (
      <Button size="lg" disabled className={cn(className)}>
        <Spinner className="size-4" />
        Checking...
      </Button>
    )
  }

  // Keeper installed - trigger rbee:// protocol
  if (installed) {
    return (
      <Button
        size="lg"
        onClick={() => {
          // Store that user clicked (for future detection)
          localStorage.setItem('rbee-keeper-detected', 'true')
          // Trigger protocol
          window.location.href = `rbee://model/${modelId}`
        }}
        className={cn(className)}
      >
        <Download className="size-4" />
        Run with rbee
      </Button>
    )
  }

  // Keeper not installed - show download link
  return (
    <Button size="lg" variant="secondary" asChild className={cn(className)}>
      <a href="https://github.com/veighnsche/llama-orch/releases" target="_blank" rel="noopener noreferrer">
        <Download className="size-4" />
        Download Keeper
        <ExternalLink className="size-4" />
      </a>
    </Button>
  )
}
