// TEAM-415: Shared types for marketplace-node

export interface ModelFile {
  rfilename: string
  size: number
}

export interface Model {
  id: string
  name: string
  author?: string
  description: string
  downloads: number
  likes: number
  size: string
  tags: string[]
  source: 'huggingface' | 'civitai'
  createdAt?: string
  lastModified?: string
  config?: any
  siblings?: ModelFile[]
}

export interface SearchOptions {
  limit?: number
  sort?: 'popular' | 'recent' | 'trending'
}

export interface Worker {
  id: string
  name: string
  type: 'cpu' | 'cuda' | 'metal'
  platform: 'linux' | 'macos' | 'windows'
  version: string
  downloadUrl: string
}
