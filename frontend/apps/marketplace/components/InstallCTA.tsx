'use client'

// TEAM-421: Environment-aware conversion CTA
// Only shows in Next.js (marketplace), hidden in Tauri (rbee-keeper)

import { getEnvironment } from '@rbee/ui/utils'
import { Download, ExternalLink, Sparkles } from 'lucide-react'

interface InstallCTAProps {
  artifactType: 'model' | 'worker'
  artifactName: string
}

/**
 * Conversion CTA for Next.js marketplace
 * Prompts users to install rbee to download/install artifacts
 *
 * Only renders in Next.js environment (hidden in Tauri)
 */
export function InstallCTA({ artifactType, artifactName }: InstallCTAProps) {
  const env = getEnvironment()

  // Only show in Next.js (not in Tauri)
  if (env === 'tauri') return null

  const action = artifactType === 'model' ? 'download' : 'install'
  const actionVerb = artifactType === 'model' ? 'Download' : 'Install'

  return (
    <div className="rounded-lg border border-border bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/50 dark:to-purple-950/50 p-6">
      <div className="flex items-start gap-4">
        <div className="rounded-full bg-primary/10 p-3">
          <Download className="size-6 text-primary" />
        </div>

        <div className="flex-1 space-y-3">
          <div>
            <h3 className="text-lg font-semibold mb-1">
              {actionVerb} {artifactName} with rbee
            </h3>
            <p className="text-sm text-muted-foreground">
              rbee is a free, open-source AI orchestration tool that lets you {action} {artifactType}s directly to your
              system with one click.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <a
              href="https://docs.rbee.dev/docs/getting-started/installation"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              <Download className="size-4 mr-2" />
              Download rbee
            </a>
            <a
              href="https://rbee.dev"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center rounded-md border border-border bg-background px-6 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
            >
              <ExternalLink className="size-4 mr-2" />
              Learn More
            </a>
          </div>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Sparkles className="size-3" />
            <span>Free • Open Source • Works on Linux, macOS, and Windows</span>
          </div>
        </div>
      </div>
    </div>
  )
}
