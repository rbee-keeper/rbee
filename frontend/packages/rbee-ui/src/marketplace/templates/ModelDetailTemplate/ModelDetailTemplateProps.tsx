// TEAM-401: All model detail data (for SSG)
import type { ModelCardProps } from '../../organisms/ModelCard'

export interface ModelDetailTemplateProps {
  model: {
    id: string
    name: string
    description: string
    longDescription?: string
    author?: string
    authorUrl?: string
    imageUrl?: string
    tags: string[]
    downloads: number
    likes: number
    size: string
    quantization?: string
    parameters?: string
    contextLength?: string
    architecture?: string
    license?: string
    createdAt?: string
    updatedAt?: string
  }
  installButton?: React.ReactNode
  relatedModels?: Array<ModelCardProps['model']>
  onRelatedModelAction?: (modelId: string) => void
}
