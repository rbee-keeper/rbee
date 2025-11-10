'use client'

// TEAM-413: Client wrapper for ModelDetailPageTemplate with InstallButton
// TEAM-421: Updated to use shared InstallCTA component

import type { ModelDetailData } from '@rbee/ui/marketplace'
import { HuggingFaceModelDetail } from '@rbee/ui/marketplace'
import { InstallCTA } from './InstallCTA'

interface ModelDetailWithInstallProps {
  model: ModelDetailData
  compatibleWorkers?: Array<{ id: string; name: string; confidence: string }>
}

export function ModelDetailWithInstall({ model }: ModelDetailWithInstallProps) {
  return (
    <div className="space-y-6">
      {/* Conversion CTA - Only shows in Next.js */}
      <InstallCTA artifactType="model" artifactName={model.name} />

      {/* Model Details */}
      <HuggingFaceModelDetail model={model} showBackButton={false} />
    </div>
  )
}
