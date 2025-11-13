// TEAM-501: Worker detail template - matches HFModelDetail structure
// 3-column layout: README (2 cols) + Sidebar (1 col)

'use client'

import { Badge, Button, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { MarkdownContent } from '@rbee/ui/molecules'
import { Download, ExternalLink, Package, Shield } from 'lucide-react'
import Image from 'next/image'

export interface WorkerDetailData {
  // Basic fields
  id: string
  name: string
  description: string
  version: string
  author: string
  license: string

  // Backends
  backends: Array<'cpu' | 'cuda' | 'metal' | 'rocm'>

  // Capabilities
  supportedFormats: string[]
  supportsStreaming: boolean
  supportsBatching: boolean
  maxContextLength?: number

  // Platforms
  platforms: string[]
  architectures: string[]

  // Implementation
  implementation: 'rust' | 'python' | 'cpp'
  buildSystem: 'cargo' | 'cmake' | 'pip' | 'npm'

  // URLs
  externalUrl?: string
  externalLabel?: string
  coverImage?: string

  // README markdown
  readmeMarkdown?: string
}

const backendConfig = {
  cpu: { label: 'CPU', variant: 'secondary' as const },
  cuda: { label: 'CUDA', variant: 'default' as const },
  metal: { label: 'Metal', variant: 'accent' as const },
  rocm: { label: 'ROCm', variant: 'destructive' as const },
}

export function WorkerDetail({ worker }: { worker: WorkerDetailData }) {
  return (
    <div className="space-y-6">
      {/* 3-column layout: README (2 cols) + Sidebar (1 col) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: README ONLY (2 columns) */}
        <div className="lg:col-span-2">
          {worker.readmeMarkdown ? (
            <MarkdownContent markdown={worker.readmeMarkdown} asCard={false} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>About</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">{worker.description}</p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: Sidebar (1 column) */}
        <div className="space-y-6">
          {/* Image and Title Card */}
          {/* 1:1 Image */}
          {worker.coverImage && (
            <div className="relative w-full aspect-square rounded-lg overflow-hidden mb-4">
              <Image src={worker.coverImage} alt={worker.name} fill className="object-cover" />
            </div>
          )}

          {/* Title and Author */}
          <div className="space-y-2">
            <h1 className="text-2xl font-bold tracking-tight">{worker.name}</h1>
            <p className="text-sm text-muted-foreground">by {worker.author}</p>
            <p className="text-sm text-muted-foreground">v{worker.version}</p>
          </div>

          {/* Backend badges */}
          <div className="flex flex-wrap gap-1 mt-4">
            {worker.backends.map((backend) => {
              const config = backendConfig[backend]
              return (
                <Badge key={backend} variant="outline" className="text-xs">
                  {config.label}
                </Badge>
              )
            })}
          </div>

          {/* Install */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Download className="size-5 text-muted-foreground" />
                <CardTitle>Install</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button className="w-full" disabled>
                <Download className="size-4" />
                Install Worker
              </Button>
              <p className="text-xs text-muted-foreground">Installation coming soon</p>
            </CardContent>
          </Card>

          {/* Metadata */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Package className="size-5 text-muted-foreground" />
                <CardTitle>Metadata</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <div className="text-sm text-muted-foreground mb-1">Author</div>
                <div className="text-sm font-medium">{worker.author}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">Version</div>
                <div className="text-sm font-medium">v{worker.version}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">Implementation</div>
                <Badge variant="outline">{worker.implementation}</Badge>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">Build System</div>
                <Badge variant="outline">{worker.buildSystem}</Badge>
              </div>
            </CardContent>
          </Card>

          {/* Capabilities */}
          <Card>
            <CardHeader>
              <CardTitle>Capabilities</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-sm font-medium mb-2">Supported Formats</div>
                <div className="flex flex-wrap gap-2">
                  {worker.supportedFormats.map((format) => (
                    <Badge key={format} variant="outline">
                      {format.toUpperCase()}
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <div>
                  <div className="text-sm font-medium mb-1">Streaming</div>
                  <Badge variant={worker.supportsStreaming ? 'default' : 'secondary'}>
                    {worker.supportsStreaming ? 'Supported' : 'Not Supported'}
                  </Badge>
                </div>
                <div>
                  <div className="text-sm font-medium mb-1">Batching</div>
                  <Badge variant={worker.supportsBatching ? 'default' : 'secondary'}>
                    {worker.supportsBatching ? 'Supported' : 'Not Supported'}
                  </Badge>
                </div>
              </div>
              {worker.maxContextLength && (
                <div>
                  <div className="text-sm font-medium mb-1">Max Context Length</div>
                  <Badge variant="outline">{worker.maxContextLength.toLocaleString()} tokens</Badge>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Platform Compatibility */}
          <Card>
            <CardHeader>
              <CardTitle>Platform Compatibility</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-sm font-medium mb-2">Backends</div>
                <div className="flex flex-wrap gap-2">
                  {worker.backends.map((backend) => {
                    const config = backendConfig[backend]
                    return (
                      <Badge key={backend} variant="outline">
                        {config.label}
                      </Badge>
                    )
                  })}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium mb-2">Operating Systems</div>
                <div className="flex flex-wrap gap-2">
                  {worker.platforms.map((platform) => (
                    <Badge key={platform} variant="outline">
                      {platform}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium mb-2">Architectures</div>
                <div className="flex flex-wrap gap-2">
                  {worker.architectures.map((arch) => (
                    <Badge key={arch} variant="outline">
                      {arch}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* License */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Shield className="size-5 text-muted-foreground" />
                <CardTitle>License</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <Badge variant="outline">{worker.license}</Badge>
            </CardContent>
          </Card>

          {/* External Link */}
          {worker.externalUrl && (
            <Card>
              <CardHeader>
                <CardTitle>Links</CardTitle>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full" asChild>
                  <a href={worker.externalUrl} target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="size-4" />
                    {worker.externalLabel || 'View Source'}
                  </a>
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
