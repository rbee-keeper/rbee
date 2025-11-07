// TEAM-460: HuggingFace model detail page (migrated from /models/[slug])
// TEAM-415: SSG model detail page with slugified URLs
// TEAM-410: Added compatibility integration
// TEAM-413: Added InstallButton integration
// TEAM-421: Pre-build top 100 popular models (WASM filtering doesn't work in SSG)
import { 
  getHuggingFaceModel, 
  listHuggingFaceModels
} from '@rbee/marketplace-node'
import { modelIdToSlug, slugToModelId } from '@/lib/slugify'
import { ModelDetailWithInstall } from '@/components/ModelDetailWithInstall'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  // TEAM-421: WASM doesn't work in Next.js SSG build context
  // Solution: Pre-build top 100 models, filter client-side at runtime
  const FETCH_LIMIT = 100
  
  console.log(`[SSG] Pre-building top ${FETCH_LIMIT} most popular HuggingFace models (compatibility check happens at runtime)`)
  
  const models = await listHuggingFaceModels({ limit: FETCH_LIMIT })
  
  console.log(`[SSG] Pre-building ${models.length} HuggingFace model pages`)
  
  return models.map((model) => ({ 
    slug: modelIdToSlug((model as {id: string}).id) 
  }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const model = await getHuggingFaceModel(modelId)
    
    return {
      title: `${model.name} | HuggingFace Model`,
      description: model.description || `${model.name} - ${model.downloads.toLocaleString()} downloads`,
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function HuggingFaceModelPage({ params }: Props) {
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
        <ModelDetailWithInstall 
          model={model} 
          compatibleWorkers={compatibleWorkers}
        />
      </div>
    )
  } catch {
    notFound()
  }
}
