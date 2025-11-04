// TEAM-405: HuggingFace API client for SSG
export async function fetchTopModels(limit: number = 100) {
  const response = await fetch(
    `https://huggingface.co/api/models?sort=downloads&direction=-1&limit=${limit}&full=true`,
    { next: { revalidate: 3600 } }
  )
  
  if (!response.ok) {
    throw new Error('Failed to fetch models')
  }
  
  return response.json()
}

export async function fetchModel(modelId: string) {
  const response = await fetch(
    `https://huggingface.co/api/models/${modelId}`,
    { next: { revalidate: 3600 } }
  )
  
  if (!response.ok) {
    throw new Error('Model not found')
  }
  
  return response.json()
}

export async function getStaticModelIds(limit: number = 100) {
  const models = await fetchTopModels(limit)
  return models.map((m: any) => m.id)
}

export function transformToModelDetailData(hfModel: any) {
  return {
    id: hfModel.id,
    name: hfModel.id.split('/').pop() || hfModel.id,
    author: hfModel.author || null,
    description: hfModel.cardData?.model_description || hfModel.description || '',
    downloads: hfModel.downloads || 0,
    likes: hfModel.likes || 0,
    tags: hfModel.tags || [],
    createdAt: hfModel.createdAt,
    lastModified: hfModel.lastModified,
    config: hfModel.config,
    siblings: hfModel.siblings || [],
  }
}
