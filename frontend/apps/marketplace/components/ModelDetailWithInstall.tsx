'use client'

// TEAM-413: Client wrapper for ModelDetailPageTemplate with InstallButton

import { ModelDetailPageTemplate } from '@rbee/ui/marketplace'
import type { ModelDetailData } from '@rbee/ui/marketplace'
import { InstallButton } from './InstallButton'

interface ModelDetailWithInstallProps {
  model: ModelDetailData
  compatibleWorkers?: Array<{ id: string; name: string; confidence: string }>
}

export function ModelDetailWithInstall({ model }: ModelDetailWithInstallProps) {
  return (
    <div className="space-y-6">
      {/* Install Button Section */}
      <div className="rounded-lg border border-border bg-card p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h3 className="text-lg font-semibold">One-Click Installation</h3>
            <p className="text-sm text-muted-foreground">
              Install and run this model with rbee Keeper
            </p>
          </div>
          <InstallButton modelId={model.id} />
        </div>
      </div>

      {/* Model Details */}
      <ModelDetailPageTemplate
        model={model}
        showBackButton={false}
      />
    </div>
  )
}
