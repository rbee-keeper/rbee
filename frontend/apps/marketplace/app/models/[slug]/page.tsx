// TEAM-415: SSG model detail page with slugified URLs
// TEAM-410: Added compatibility integration
import { ModelDetailPageTemplate } from '@rbee/ui/marketplace'
import { getHuggingFaceModel, listHuggingFaceModels } from '@rbee/marketplace-node'
import { modelIdToSlug, slugToModelId } from '@/lib/slugify'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  const models = await listHuggingFaceModels({ limit: 100 })
  return models.map((model) => ({ 
    slug: modelIdToSlug(model.id) 
  }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const model = await getHuggingFaceModel(modelId)
    
    return {
      title: `${model.name} | AI Model`,
      description: model.description || `${model.name} - ${model.downloads.toLocaleString()} downloads`,
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function ModelPage({ params }: Props) {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const model = await getHuggingFaceModel(modelId)
    
    // TEAM-410: Get compatible workers (build time)
    // Note: In a real implementation, this would call marketplace-node's
    // checkModelCompatibility function for each worker type.
    // For now, we'll pass undefined and let the component handle it gracefully.
    const compatibleWorkers = undefined
    
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <ModelDetailPageTemplate 
          model={model} 
          showBackButton={false}
          compatibleWorkers={compatibleWorkers}
        />
      </div>
    )
  } catch {
    notFound()
  }
}
