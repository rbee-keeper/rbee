// TEAM-405: SSG model detail page with slugified URLs
import { ModelDetailPageTemplate } from '@rbee/ui/marketplace'
import { fetchModel, transformToModelDetailData, getStaticModelIds } from '@/lib/huggingface'
import { modelIdToSlug, slugToModelId } from '@/lib/slugify'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  const modelIds = await getStaticModelIds(100)
  return modelIds.map((id: string) => ({ 
    slug: modelIdToSlug(id) 
  }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const hfModel = await fetchModel(modelId)
    const model = transformToModelDetailData(hfModel)
    
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
    const hfModel = await fetchModel(modelId)
    const model = transformToModelDetailData(hfModel)
    
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <ModelDetailPageTemplate model={model} showBackButton={false} />
      </div>
    )
  } catch {
    notFound()
  }
}
