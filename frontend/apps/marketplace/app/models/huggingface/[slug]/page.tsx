// TEAM-477: HuggingFace model detail page - CORRECT implementation
// Uses @rbee/marketplace-core adapters (not marketplace-node)
// Uses fetchHuggingFaceModel (returns normalized MarketplaceModel)
// Uses HFModelDetail template (3-column design)
// TEAM-478: Added README markdown parser above the fold

import { fetchHuggingFaceModel, fetchHuggingFaceModelReadme } from '@rbee/marketplace-core'
import { HFModelDetail } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

// Cache for 1 hour
export const revalidate = 3600

interface Props {
  params: Promise<{ slug: string }>
}

// Helper to convert slug to model ID
function slugToModelId(slug: string): string {
  // Slug format: "meta-llama-llama-2-7b-hf" -> "meta-llama/Llama-2-7b-hf"
  // URL decode first
  const decoded = decodeURIComponent(slug)
  
  // If it already contains /, return as-is
  if (decoded.includes('/')) {
    return decoded
  }
  
  // Otherwise, try to parse from slug format
  // Common pattern: org-name-model-name -> org-name/model-name
  const parts = decoded.split('-')
  if (parts.length >= 2) {
    const org = parts[0]
    const modelName = parts.slice(1).join('-')
    return `${org}/${modelName}`
  }
  
  return decoded
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)

  try {
    const model = await fetchHuggingFaceModel(modelId)

    return {
      title: `${model.name} | HuggingFace Model`,
      description: model.description || `${model.name} - ${model.downloads.toLocaleString()} downloads`,
      openGraph: {
        title: `${model.name} | HuggingFace Model`,
        description: model.description || `${model.name} - ${model.downloads.toLocaleString()} downloads`,
        images: model.imageUrl ? [{ url: model.imageUrl }] : undefined,
      },
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function HuggingFaceModelDetailPage({ params }: Props) {
  const { slug } = await params
  const modelId = slugToModelId(slug)

  try {
    // TEAM-477: Use marketplace-core adapter (returns normalized MarketplaceModel)
    // TEAM-478: Fetch both model data and README in parallel
    const [model, readme] = await Promise.all([
      fetchHuggingFaceModel(modelId),
      fetchHuggingFaceModelReadme(modelId),
    ])

    // TEAM-477: Convert MarketplaceModel to HFModelDetailData for template
    // TEAM-478: Include README in model data for 3-column layout
    const hfModelData = {
      id: model.id,
      name: model.name,
      description: model.description || '',
      author: model.author,
      downloads: model.downloads,
      likes: model.likes,
      size: model.sizeBytes ? formatBytes(model.sizeBytes) : 'Unknown',
      tags: model.tags,
      
      // HF-specific metadata
      pipeline_tag: model.metadata?.pipeline_tag as string | undefined,
      library_name: model.metadata?.library_name as string | undefined,
      sha: model.metadata?.sha as string | undefined,
      
      // Dates
      createdAt: model.createdAt.toISOString(),
      lastModified: model.updatedAt.toISOString(),
      
      // External URL
      externalUrl: model.url,
      externalLabel: 'View on HuggingFace',
      
      // TEAM-478: README markdown (displayed in first 2 columns)
      ...(readme ? { readmeMarkdown: readme } : {}),
    }

    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <HFModelDetail model={hfModelData} />
      </div>
    )
  } catch (error) {
    console.error('[HuggingFace Detail] Error:', error)
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
