// TEAM-463: CivitAI model details page - Separated from generic ModelDetailsPage
// Created by: TEAM-463
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: ModelDetailPageTemplate from rbee-ui

import { Button, Card, CardContent } from '@rbee/ui/atoms'
import type { ModelDetailData } from '@rbee/ui/marketplace'
import { ModelDetailPageTemplate, useArtifactActions } from '@rbee/ui/marketplace'
import { PageContainer } from '@rbee/ui/molecules'
import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { ArrowLeft, Sparkles } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import type { Model } from '@/generated/bindings'

export function ModelDetailsCivitAIPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const navigate = useNavigate()

  // Environment-aware actions
  const actions = useArtifactActions({
    onActionSuccess: (action) => {
      console.log(`✅ ${action} started successfully`)
    },
    onActionError: (action, error) => {
      console.error(`❌ ${action} failed:`, error)
    },
  })

  // Fetch the specific model by ID
  const {
    data: rawModel,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['marketplace', 'civitai', 'model', modelId],
    queryFn: async () => {
      if (!modelId) throw new Error('No model ID provided')
      const result = await invoke<Model>('marketplace_get_model', {
        modelId: decodeURIComponent(modelId),
      })
      return result
    },
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000,
  })

  // Transform Model to ModelDetailData format
  const model: ModelDetailData | undefined = rawModel
    ? {
        id: rawModel.id,
        name: rawModel.name,
        description: rawModel.description,
        author: rawModel.author,
        downloads: rawModel.downloads,
        likes: rawModel.likes,
        size: rawModel.size,
        tags: rawModel.tags,
        // Pass through all CivitAI-specific fields
        ...(rawModel as any),
      }
    : undefined

  if (isLoading) {
    return (
      <PageContainer title="Loading..." description="Fetching model details" padding="default">
        <div className="animate-pulse space-y-6">
          <div className="h-64 bg-muted/20 rounded-lg" />
          <div className="h-32 bg-muted/20 rounded-lg" />
          <div className="h-48 bg-muted/20 rounded-lg" />
        </div>
      </PageContainer>
    )
  }

  if (error || !model) {
    return (
      <PageContainer title="Model Not Found" description="The requested model could not be found" padding="default">
        <Card>
          <CardContent className="p-12 text-center">
            <Sparkles className="size-16 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-xl font-semibold mb-2">Model not found</h3>
            <p className="text-muted-foreground mb-6">
              {error ? `Error: ${String(error)}` : "The model you're looking for doesn't exist or has been removed."}
            </p>
            <Button onClick={() => navigate('/marketplace/civitai')}>
              <ArrowLeft className="size-4 mr-2" />
              Back to CivitAI Models
            </Button>
          </CardContent>
        </Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer title="" padding="default">
      <ModelDetailPageTemplate
        model={model}
        onBack={() => navigate('/marketplace/civitai')}
        onDownload={() => actions.downloadModel(model.id)}
        isLoading={isLoading}
        // CivitAI models are Stable Diffusion, not LLMs - different worker compatibility
        // For now, pass undefined and let the component handle it gracefully
        compatibleWorkers={undefined}
      />
    </PageContainer>
  )
}
