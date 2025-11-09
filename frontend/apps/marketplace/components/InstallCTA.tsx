'use client'

// TEAM-427: Environment-aware conversion CTA using rbee-ui components
// Refactored from ad-hoc implementation to use proper atomic design

import { getEnvironment } from '@rbee/ui/utils'
import { Button, Card, CardContent } from '@rbee/ui/atoms'
import { Download, ExternalLink, Sparkles } from 'lucide-react'

interface InstallCTAProps {
  artifactType: 'model' | 'worker'
  artifactName: string
}

/**
 * Conversion CTA for Next.js marketplace
 * Prompts users to install rbee to download/install artifacts
 *
 * Uses rbee-ui atoms (Card, Button) following atomic design pattern
 * Only renders in Next.js environment (hidden in Tauri)
 */
export function InstallCTA({ artifactType, artifactName }: InstallCTAProps) {
  const env = getEnvironment()

  // Only show in Next.js (not in Tauri)
  if (env === 'tauri') return null

  const action = artifactType === 'model' ? 'download' : 'install'
  const actionVerb = artifactType === 'model' ? 'Download' : 'Install'

  return (
    <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/50 dark:to-purple-950/50">
      <CardContent className="p-6">
        <div className="flex items-start gap-4">
          {/* Icon */}
          <div className="rounded-full bg-primary/10 p-3 shrink-0">
            <Download className="size-6 text-primary" />
          </div>

          {/* Content */}
          <div className="flex-1 space-y-3">
            {/* Heading & Description */}
            <div>
              <h3 className="text-lg font-semibold mb-1">
                {actionVerb} {artifactName} with rbee
              </h3>
              <p className="text-sm text-muted-foreground">
                rbee is a free, open-source AI orchestration tool that lets you {action} {artifactType}s directly to
                your system with one click.
              </p>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-wrap gap-3">
              <Button size="lg" asChild>
                <a
                  href="https://docs.rbee.dev/docs/getting-started/installation"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Download className="size-4" />
                  Download rbee
                </a>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <a href="https://rbee.dev" target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="size-4" />
                  Learn More
                </a>
              </Button>
            </div>

            {/* Footer */}
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Sparkles className="size-3" />
              <span>Free • Open Source • Works on Linux, macOS, and Windows</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
