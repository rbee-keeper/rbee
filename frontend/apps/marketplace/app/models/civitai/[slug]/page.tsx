// TEAM-477: CivitAI model detail page - CORRECT implementation
// Uses @rbee/marketplace-core adapters (not marketplace-node)
// Uses fetchCivitAIModel (returns normalized MarketplaceModel)
// Uses CivitAIModelDetail template (3-column design)

import { fetchCivitAIModel } from '@rbee/marketplace-core'
import { CivitAIModelDetail } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

// Cache for 1 hour
export const revalidate = 3600

interface Props {
  params: Promise<{ slug: string }>
}

// Helper to convert slug to model ID
function slugToModelId(slug: string): number {
  // Slug format: "civitai-12345-model-name" -> 12345
  // OR just "12345" -> 12345
  const decoded = decodeURIComponent(slug)

  // Try to extract number from slug
  const match = decoded.match(/(\d+)/)
  if (match?.[1]) {
    return parseInt(match[1], 10)
  }

  // Fallback: try to parse as number
  const id = parseInt(decoded, 10)
  if (!Number.isNaN(id)) {
    return id
  }

  throw new Error(`Invalid CivitAI model ID: ${slug}`)
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params

  try {
    const modelId = slugToModelId(slug)
    const model = await fetchCivitAIModel(modelId)

    return {
      title: `${model.name} | CivitAI Model`,
      description: model.description || `${model.name} - ${model.downloads.toLocaleString()} downloads`,
      openGraph: {
        title: `${model.name} | CivitAI Model`,
        description: model.description || `${model.name} - ${model.downloads.toLocaleString()} downloads`,
        images: model.imageUrl ? [{ url: model.imageUrl }] : undefined,
      },
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function CivitAIModelDetailPage({ params }: Props) {
  const { slug } = await params

  try {
    const modelId = slugToModelId(slug)
    
    // TEAM-477: Use marketplace-core adapter (returns normalized MarketplaceModel)
    const model = await fetchCivitAIModel(modelId)

    // TEAM-477: Convert MarketplaceModel to CivitAIModelDetailProps format
    const civitaiModelData = {
      id: model.id,
      name: model.name,
      description: model.description || '',
      author: model.author,
      downloads: model.downloads,
      likes: model.likes,
      rating: typeof model.metadata?.rating === 'number' ? model.metadata.rating : 0,
      size: model.sizeBytes ? formatBytes(model.sizeBytes) : 'Unknown',
      tags: model.tags,
      type: model.type,
      baseModel: (model.metadata?.baseModel as string) || 'Unknown',
      version: (model.metadata?.version as string) || 'Latest',
      images: model.imageUrl && !model.nsfw ? [{ url: model.imageUrl, nsfw: false, width: 1024, height: 1024 }] : [],
      files: [], // Files would come from metadata if available
      trainedWords: (model.metadata?.trainedWords as string[]) || undefined,
      allowCommercialUse: model.metadata?.allowCommercialUse || 'Unknown',
      externalUrl: model.url,
      externalLabel: 'View on CivitAI',
    }

    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <CivitAIModelDetail model={civitaiModelData} />
      </div>
    )
  } catch (error) {
    console.error('[CivitAI Detail] Error:', error)
    notFound()
  }
}

// Helper function
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`
}
