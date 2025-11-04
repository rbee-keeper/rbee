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
  const models = await fetchTopModels(limit) as any[]
  return models.map((m) => m.id)
}

export function transformToModelDetailData(hfModel: any) {
  // Calculate total size from siblings (model files)
  const totalBytes = hfModel.siblings?.reduce((sum: number, file: any) => {
    return sum + (file.size || 0)
  }, 0) || 0
  
  // Format size as human-readable string
  const formatSize = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
  }
  
  return {
    id: hfModel.id,
    name: hfModel.id.split('/').pop() || hfModel.id,
    author: hfModel.author || null,
    description: hfModel.cardData?.model_description || hfModel.description || '',
    downloads: hfModel.downloads || 0,
    likes: hfModel.likes || 0,
    size: formatSize(totalBytes),
    tags: hfModel.tags || [],
    createdAt: hfModel.createdAt,
    lastModified: hfModel.lastModified,
    config: hfModel.config,
    siblings: hfModel.siblings || [],
  }
}
