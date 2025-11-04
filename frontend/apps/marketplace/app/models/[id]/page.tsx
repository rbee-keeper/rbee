// TEAM-405: SSG model detail page
import { ModelDetailPageTemplate } from '@rbee/ui/marketplace'
import { fetchModel, transformToModelDetailData, getStaticModelIds } from '@/lib/huggingface'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

interface Props {
  params: Promise<{ id: string }>
}

export async function generateStaticParams() {
  const modelIds = await getStaticModelIds(100)
  return modelIds.map((id: string) => ({ id: encodeURIComponent(id) }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params
  const modelId = decodeURIComponent(id)
  
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
  const { id } = await params
  const modelId = decodeURIComponent(id)
  
  try {
    const hfModel = await fetchModel(modelId)
    const model = transformToModelDetailData(hfModel)
    
    return <ModelDetailPageTemplate model={model} />
  } catch {
    notFound()
  }
}
