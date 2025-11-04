// TEAM-381: Model Management - Main component with clean composition
// TEAM-405: Removed "Search HuggingFace" tab - marketplace search moved to separate component
// TEAM-405: Now focuses on LOCAL CATALOG management (Downloaded, Loaded)

import { useState } from 'react'
import { HardDrive, Play } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from '@rbee/ui/atoms'
import { useModels, useModelOperations } from '@rbee/rbee-hive-react'
import { DownloadedModelsView } from './DownloadedModelsView'
import { LoadedModelsView } from './LoadedModelsView'
import { ModelDetailsPanel } from './ModelDetailsPanel'
import type { ViewMode, ModelInfo } from './types'

export function ModelManagement() {
  const { models, loading, error } = useModels()
  const [viewMode, setViewMode] = useState<ViewMode>('downloaded')
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  
  const { loadModel, unloadModel, deleteModel, isPending } = useModelOperations()

  // Filter models by state
  const downloadedModels = models.filter((m: ModelInfo) => !m.loaded)
  const loadedModels = models.filter((m: ModelInfo) => m.loaded)

  // Model operations
  const handleLoadModel = (modelId: string, device: string = 'cuda:0') => {
    loadModel({ modelId, device })
  }

  const handleUnloadModel = (modelId: string) => {
    unloadModel({ modelId })
  }

  const handleDeleteModel = (modelId: string) => {
    deleteModel({ modelId })
  }

  return (
    <Card className="col-span-2">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <HardDrive className="h-5 w-5" />
              Model Management
            </CardTitle>
            <CardDescription>
              Download models, load to RAM, and deploy to workers
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Badge variant="secondary">{downloadedModels.length} Downloaded</Badge>
            <Badge variant="default">{loadedModels.length} Loaded</Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* TEAM-405: Removed "Search HuggingFace" tab - use MarketplaceSearch component instead */}
        {/* View Mode Tabs */}
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="downloaded">
              <HardDrive className="h-4 w-4 mr-2" />
              Downloaded ({downloadedModels.length})
            </TabsTrigger>
            <TabsTrigger value="loaded">
              <Play className="h-4 w-4 mr-2" />
              Loaded in RAM ({loadedModels.length})
            </TabsTrigger>
          </TabsList>

          {/* Downloaded Models Tab */}
          <TabsContent value="downloaded" className="space-y-4">
            <DownloadedModelsView
              models={downloadedModels}
              loading={loading}
              error={error}
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
              onLoad={handleLoadModel}
              onDelete={handleDeleteModel}
            />
          </TabsContent>

          {/* Loaded Models Tab */}
          <TabsContent value="loaded" className="space-y-4">
            <LoadedModelsView
              models={loadedModels}
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
              onUnload={handleUnloadModel}
            />
          </TabsContent>
        </Tabs>

        {/* Model Details Panel (always visible when model selected) */}
        {selectedModel && (
          <ModelDetailsPanel
            model={selectedModel}
            onLoad={handleLoadModel}
            onUnload={handleUnloadModel}
            onDelete={handleDeleteModel}
            isPending={isPending}
          />
        )}
      </CardContent>
    </Card>
  )
}

// Re-export types for convenience
export type { ViewMode, ModelInfo } from './types'
