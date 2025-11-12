// TEAM-482: Worker detail page - fetches from GWC API

import type { MarketplaceModel } from '@rbee/marketplace-core'
import { fetchGWCWorker } from '@rbee/marketplace-core'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms/Card'
import { DevelopmentBanner } from '@rbee/ui/molecules'
import { FeatureHeader } from '@rbee/ui/molecules/FeatureHeader'
import { Cpu, Download } from 'lucide-react'
import { notFound } from 'next/navigation'

interface WorkerDetailPageProps {
  params: Promise<{ slug: string }>
}

const workerTypeConfig = {
  cpu: { label: 'CPU', variant: 'secondary' as const },
  cuda: { label: 'CUDA', variant: 'default' as const },
  metal: { label: 'Metal', variant: 'accent' as const },
  rocm: { label: 'ROCm', variant: 'destructive' as const },
}

export default async function WorkerDetailPage({ params }: WorkerDetailPageProps) {
  const { slug } = await params
  
  // Fetch worker from GWC API
  let model: MarketplaceModel
  try {
    model = await fetchGWCWorker(slug)
  } catch (error) {
    console.error(`Failed to fetch worker ${slug}:`, error)
    notFound()
  }

  // Extract worker type from tags (tags[1] is the backend)
  const workerType = (model.tags[1] as 'cpu' | 'cuda' | 'metal' | 'rocm') || 'cpu'
  const typeConfig = workerTypeConfig[workerType]
  const version = model.metadata?.version as string || '0.1.0'
  const backends = model.metadata?.backends as string || workerType
  const platforms = model.tags.filter(t => ['linux', 'macos', 'windows'].includes(t))
  const architectures = model.tags.filter(t => ['x86_64', 'aarch64'].includes(t))

  return (
    <>
      {/* MVP Notice */}
      <DevelopmentBanner
        variant="mvp"
        message="🔨 Worker detail page is under development."
        details="Installation and configuration features coming soon."
      />

      <div className="container mx-auto py-8">
        <div className="space-y-8">
          {/* Header */}
          <div className="space-y-4">
            <FeatureHeader title={model.name} subtitle={`Version ${version}`} />
          </div>

          {/* Main content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left column - Details */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>About</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{model.description}</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Compatibility</CardTitle>
                  <CardDescription>Supported platforms and architectures</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm font-medium mb-2">Backends</div>
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline">{backends}</Badge>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium mb-2">Platforms</div>
                    <div className="flex flex-wrap gap-2">
                      {platforms.length > 0 ? (
                        platforms.map((platform) => (
                          <Badge key={platform} variant="outline">
                            {platform}
                          </Badge>
                        ))
                      ) : (
                        <Badge variant="outline">linux, macos, windows</Badge>
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium mb-2">Architecture</div>
                    <div className="flex flex-wrap gap-2">
                      {architectures.length > 0 ? (
                        architectures.map((arch) => (
                          <Badge key={arch} variant="secondary">
                            {arch}
                          </Badge>
                        ))
                      ) : (
                        <Badge variant="secondary">x86_64, aarch64</Badge>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right column - Sidebar */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Cpu className="size-5 text-muted-foreground" />
                    <CardTitle>Worker Type</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <Badge variant={typeConfig.variant} className="text-sm">
                    {typeConfig.label}
                  </Badge>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Install</CardTitle>
                  <CardDescription>Add this worker to your cluster</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button className="w-full" disabled>
                    <Download className="size-4" />
                    Install Worker
                  </Button>
                  <p className="text-xs text-muted-foreground mt-2">Installation coming soon</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
