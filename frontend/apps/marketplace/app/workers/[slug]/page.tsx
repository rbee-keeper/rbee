// TEAM-482: Worker detail page - fetches from GWC API
// TEAM-501: Refactored to match HuggingFace pattern with WorkerDetail template

import { fetchGWCWorker, fetchGWCWorkerReadme } from '@rbee/marketplace-core'
import { WorkerDetail } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

// Cache for 1 hour
export const revalidate = 3600

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params

  try {
    const model = await fetchGWCWorker(slug)

    return {
      title: `${model.name} | AI Worker`,
      description: model.description || `${model.name} - rbee AI Worker`,
      openGraph: {
        title: `${model.name} | AI Worker`,
        description: model.description || `${model.name} - rbee AI Worker`,
        images: model.imageUrl ? [{ url: model.imageUrl }] : undefined,
      },
    }
  } catch {
    return { title: 'Worker Not Found' }
  }
}

export default async function WorkerDetailPage({ params }: Props) {
  const { slug } = await params

  try {
    // TEAM-501: Fetch both worker data and README in parallel (like HuggingFace)
    const [model, readme] = await Promise.all([fetchGWCWorker(slug), fetchGWCWorkerReadme(slug)])

    // Extract all backends from tags (skip first tag which is implementation)
    const backends = model.tags
      .slice(1)
      .filter((tag): tag is 'cpu' | 'cuda' | 'metal' | 'rocm' => ['cpu', 'cuda', 'metal', 'rocm'].includes(tag))

    // Extract platforms and architectures
    const platforms = model.tags.filter((t: string) => ['linux', 'macos', 'windows'].includes(t))
    const architectures = model.tags.filter((t: string) => ['x86_64', 'aarch64'].includes(t))

    // Convert MarketplaceModel to WorkerDetailData
    const workerData = {
      id: model.id,
      name: model.name,
      description: model.description || '',
      version: (model.metadata?.version as string) || '0.1.0',
      author: model.author,
      license: model.license || 'Unknown',

      // Backends (explicitly typed)
      backends: (backends.length > 0 ? backends : ['cpu']) as Array<'cpu' | 'cuda' | 'metal' | 'rocm'>,

      // Capabilities
      supportedFormats: (model.metadata?.supportedFormats as string)?.split(', ') || [],
      supportsStreaming: (model.metadata?.supportsStreaming as boolean) || false,
      supportsBatching: (model.metadata?.supportsBatching as boolean) || false,
      ...(model.metadata?.maxContextLength ? { maxContextLength: model.metadata.maxContextLength as number } : {}),

      // Platforms
      platforms: platforms.length > 0 ? platforms : ['linux', 'macos', 'windows'],
      architectures: architectures.length > 0 ? architectures : ['x86_64', 'aarch64'],

      // Implementation
      implementation: (model.tags[0] as 'rust' | 'python' | 'cpp') || 'rust',
      buildSystem: (model.metadata?.buildSystem as 'cargo' | 'cmake' | 'pip' | 'npm') || 'cargo',

      // URLs
      externalUrl: model.url,
      externalLabel: 'View on GitHub',
      ...(model.imageUrl ? { coverImage: model.imageUrl } : {}),

      // README markdown
      ...(readme ? { readmeMarkdown: readme } : {}),
    }

    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <WorkerDetail worker={workerData} />
      </div>
    )
  } catch (error) {
    console.error('[Worker Detail] Error:', error)
    notFound()
  }
}
