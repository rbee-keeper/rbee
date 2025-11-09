'use client'

// TEAM-421: Client wrapper for worker detail page with conversion CTA
// Uses shared ArtifactDetailPageTemplate from rbee-ui

import { Badge, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { ArtifactDetailPageTemplate, useArtifactActions } from '@rbee/ui/marketplace'
import { CheckCircle, Cpu, GitBranch, Package } from 'lucide-react'
import { InstallCTA } from './InstallCTA'

interface Worker {
  id: string
  name: string
  description: string
  type: 'cpu' | 'cuda' | 'metal' | 'rocm'
  platform: string[]
  version: string
  requirements: string[]
  features: string[]
}

interface WorkerDetailWithInstallProps {
  worker: Worker
}

export function WorkerDetailWithInstall({ worker }: WorkerDetailWithInstallProps) {
  const actions = useArtifactActions()

  // Worker type badge configuration
  const typeConfig = {
    cpu: { label: 'CPU', variant: 'secondary' as const },
    cuda: { label: 'CUDA', variant: 'default' as const },
    metal: { label: 'Metal', variant: 'accent' as const },
    rocm: { label: 'ROCm', variant: 'default' as const },
  }

  const config = typeConfig[worker.type]

  // Build main content cards
  const mainContent = (
    <>
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Platforms & Architecture */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Package className="size-4" />
              Platform Support
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="text-sm font-medium text-muted-foreground mb-2">Platforms</div>
              <div className="flex flex-wrap gap-2">
                {worker.platform.map((platform) => (
                  <Badge key={platform} variant="outline" className="capitalize">
                    {platform}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Requirements */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="size-4" />
              Requirements
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {worker.requirements.map((req, index) => (
                <li key={index} className="flex items-start gap-2 text-sm">
                  <span className="text-primary mt-0.5">•</span>
                  <span>{req}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Features */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <CheckCircle className="size-4" />
            Features
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2">
            {worker.features.map((feature, index) => (
              <div key={index} className="flex items-start gap-2 text-sm">
                <div className="mt-0.5 size-5 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs text-primary">✓</span>
                </div>
                <span>{feature}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </>
  )

  return (
    <div className="space-y-6">
      {/* Conversion CTA - Only shows in Next.js */}
      <InstallCTA artifactType="worker" artifactName={worker.name} />

      {/* Worker Details - Uses shared template */}
      <ArtifactDetailPageTemplate
        name={worker.name}
        description={worker.description}
        badges={[
          { label: `v${worker.version}`, variant: 'outline' },
          { label: config.label, variant: config.variant },
        ]}
        primaryAction={{
          label: actions.getButtonLabel('install'),
          icon: <GitBranch className="size-4 mr-2" />,
          onClick: () => actions.installWorker(worker.id),
        }}
        mainContent={mainContent}
      />
    </div>
  )
}
