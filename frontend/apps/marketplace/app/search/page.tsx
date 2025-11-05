// TEAM-415: Dynamic search page
//! Client-side search using Next.js API routes
//! This page is NOT pre-rendered - it's for dynamic search

'use client'

import { useState, useEffect } from 'react'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { useSearchParams } from 'next/navigation'

export default function SearchPage() {
  const searchParams = useSearchParams()
  const [models, setModels] = useState<ModelTableItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string>()
  
  // Get initial search from URL
  const initialSearch = searchParams.get('q') || ''
  
  useEffect(() => {
    async function fetchModels() {
      setIsLoading(true)
      setError(undefined)
      
      try {
        const params = new URLSearchParams()
        if (initialSearch) params.set('query', initialSearch)
        params.set('limit', '100')
        
        const response = await fetch(`/api/models?${params}`)
        
        if (!response.ok) {
          throw new Error('Failed to fetch models')
        }
        
        const data = await response.json()
        
        // Transform to ModelTableItem format
        interface HFModel {
          id: string
          author?: string
          pipeline_tag?: string
          downloads: number
          likes: number
          tags: string[]
        }
        const transformedModels: ModelTableItem[] = (data as HFModel[]).map((model) => ({
          id: model.id,
          name: model.id.split('/').pop() || model.id,
          description: `${model.author || 'Community'} - ${model.pipeline_tag || 'text-generation'}`,
          author: model.author || null,
          downloads: model.downloads,
          likes: model.likes,
          tags: model.tags.slice(0, 10)
        }))
        
        setModels(transformedModels)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load models')
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchModels()
  }, [initialSearch])
  
  return (
    <main className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Search Models</h1>
        <p className="text-muted-foreground text-lg">
          {initialSearch ? `Results for "${initialSearch}"` : 'Search AI language models'}
        </p>
      </div>

      {isLoading ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Loading models...</p>
        </div>
      ) : error ? (
        <div className="text-center py-12">
          <p className="text-destructive">{error}</p>
        </div>
      ) : models.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No models found</p>
          <p className="text-sm text-muted-foreground">Try adjusting your search query</p>
        </div>
      ) : (
        <div className="rounded-lg border border-border bg-card p-6">
          <ModelTableWithRouting models={models} />
        </div>
      )}
    </main>
  )
}
