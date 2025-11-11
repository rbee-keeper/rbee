// TEAM-477: HuggingFace model detail page for Tauri
// Uses @rbee/marketplace-core adapters (client-side fetching)
// Uses HFModelDetail template (3-column design)

import { fetchHuggingFaceModel } from '@rbee/marketplace-core'
import { HFModelDetail } from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'

export function HFDetailsPage() {
  const { modelId } = useParams<{ modelId: string }>()

  // Fetch model data using marketplace-core adapter
  const { data: model, isLoading, error } = useQuery({
    queryKey: ['huggingface-model', modelId],
    queryFn: async () => {
      if (!modelId) throw new Error('Model ID is required')
      // Decode URL-encoded model ID (e.g., "meta-llama%2FLlama-2-7b-hf" -> "meta-llama/Llama-2-7b-hf")
      const decodedId = decodeURIComponent(modelId)
      return fetchHuggingFaceModel(decodedId)
    },
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  // Loading state
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-12 max-w-7xl">
        <div className="text-center py-12 text-muted-foreground">Loading model details...</div>
      </div>
    )
  }

  // Error state
  if (error || !model) {
    return (
      <div className="container mx-auto px-4 py-12 max-w-7xl">
        <div className="text-center py-12 text-destructive">
          Error loading model: {error ? String(error) : 'Model not found'}
        </div>
      </div>
    )
  }

  // Convert MarketplaceModel to HFModelDetailData
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
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <HFModelDetail model={hfModelData} />
    </div>
  )
}

// Helper function
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`
}
