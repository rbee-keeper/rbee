'use client'

// TEAM-413: Install button component with Keeper detection

import { useKeeperInstalled } from '../app/hooks/useKeeperInstalled'

interface InstallButtonProps {
  modelId: string
  className?: string
}

export function InstallButton({ modelId, className = '' }: InstallButtonProps) {
  const { installed, checking } = useKeeperInstalled()

  if (checking) {
    return (
      <button
        disabled
        className={`inline-flex items-center justify-center rounded-md bg-muted px-6 py-3 text-sm font-medium text-muted-foreground cursor-not-allowed ${className}`}
      >
        <svg className="mr-2 h-4 w-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
        Checking...
      </button>
    )
  }

  if (installed) {
    // TEAM-413: Keeper is installed - trigger rbee:// protocol
    return (
      <button
        onClick={() => {
          // Store that user clicked (for future detection)
          localStorage.setItem('rbee-keeper-detected', 'true')
          // Trigger protocol
          window.location.href = `rbee://model/${modelId}`
        }}
        className={`inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors ${className}`}
      >
        <svg
          className="mr-2 h-4 w-4"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        Run with rbee
      </button>
    )
  }

  // TEAM-413: Keeper not installed - show download link
  return (
    <a
      href="https://github.com/veighnsche/llama-orch/releases"
      target="_blank"
      rel="noopener noreferrer"
      className={`inline-flex items-center justify-center rounded-md bg-secondary px-6 py-3 text-sm font-medium text-secondary-foreground hover:bg-secondary/80 transition-colors ${className}`}
    >
      <svg
        className="mr-2 h-4 w-4"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
      Download Keeper
      <svg
        className="ml-2 h-4 w-4"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
        <polyline points="15 3 21 3 21 9" />
        <line x1="10" y1="14" x2="21" y2="3" />
      </svg>
    </a>
  )
}
