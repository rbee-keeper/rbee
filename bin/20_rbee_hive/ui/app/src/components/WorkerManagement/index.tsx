// TEAM-382: Worker Management - Main component with clean composition
// Updated to focus on worker installation lifecycle first

import { useState } from 'react'
import { Server, Plus, Activity, Package } from 'lucide-react'
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
import { parseNarrationLine } from '@rbee/ui/organisms'
import { useWorkers, useWorkerOperations } from '@rbee/rbee-hive-react'
import { useModels } from '@rbee/rbee-hive-react'
import { ActiveWorkersView } from './ActiveWorkersView'
import { SpawnWorkerView } from './SpawnWorkerView'
import { WorkerCatalogView } from './WorkerCatalogView'
import { InstalledWorkersView } from './InstalledWorkersView'
import type { ViewMode, SpawnFormState } from './types'

export function WorkerManagement() {
  const { workers, loading, error } = useWorkers()
  const { models } = useModels()
  
  // TEAM-384: Parse SSE messages for better formatting
  const [installProgress, setInstallProgress] = useState<string[]>([])
  
  const { spawnWorker, installWorker, terminateWorker, isPending, isSuccess, isError, error: installError, reset } = useWorkerOperations({
    onSSEMessage: (line: string) => {
      // TEAM-384: Parse the line to extract just the message
      const parsed = parseNarrationLine(line)
      setInstallProgress(prev => [...prev, parsed.message])
    }
  })
  
  const handleReset = () => {
    reset()
    setInstallProgress([])
  }
  
  const [viewMode, setViewMode] = useState<ViewMode>('catalog') // Start with catalog - install workers first!

  // Separate idle and active workers
  const idleWorkers = workers.filter((w: any) => w.gpu_util_pct === 0.0)
  const activeWorkers = workers.filter((w: any) => w.gpu_util_pct > 0.0)

  const handleSpawnWorker = (params: SpawnFormState) => {
    spawnWorker({
      modelId: params.modelId,
      workerType: params.workerType,
      deviceId: params.deviceId,
    })
  }
  
  const handleInstallWorker = async (workerId: string) => {
    // TEAM-378: Call hive backend via SDK → JobClient → Job Server
    console.log('[WorkerManagement] 📥 Received install request for:', workerId)
    console.log('[WorkerManagement] 🚀 Calling useWorkerOperations.installWorker()...')
    installWorker(workerId)
    console.log('[WorkerManagement] ✓ installWorker() called (async operation started)')
  }
  
  const handleRemoveWorker = async (workerId: string) => {
    // TODO: Call hive backend to remove worker
    console.log('Removing worker:', workerId)
    // DELETE /v1/workers/{workerId}
  }
  
  // TEAM-384: Handle worker termination (stop running worker)
  const handleTerminateWorker = (pid: number) => {
    console.log('[WorkerManagement] 🛑 Terminating worker PID:', pid)
    terminateWorker(pid)
  }

  return (
    <Card className="col-span-2">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Worker Management
            </CardTitle>
            <CardDescription>
              Install workers, monitor performance, and manage processes
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Badge variant="secondary">{idleWorkers.length} Idle</Badge>
            <Badge variant="default">{activeWorkers.length} Active</Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* View Mode Tabs */}
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="catalog">
              <Package className="h-4 w-4 mr-2" />
              Catalog
            </TabsTrigger>
            <TabsTrigger value="installed">
              <Server className="h-4 w-4 mr-2" />
              Installed
            </TabsTrigger>
            <TabsTrigger value="active">
              <Activity className="h-4 w-4 mr-2" />
              Active ({workers.length})
            </TabsTrigger>
            <TabsTrigger value="spawn">
              <Plus className="h-4 w-4 mr-2" />
              Spawn
            </TabsTrigger>
          </TabsList>

          {/* Worker Catalog Tab - FIRST PRIORITY */}
          <TabsContent value="catalog" className="space-y-4">
            <WorkerCatalogView
              onInstall={handleInstallWorker}
              onRemove={handleRemoveWorker}
              installProgress={installProgress}
              isInstalling={isPending}
              installSuccess={isSuccess}
              installError={isError ? installError : null}
              onResetInstall={handleReset}
            />
          </TabsContent>

          {/* Installed Workers Tab */}
          <TabsContent value="installed" className="space-y-4">
            <InstalledWorkersView
              onUninstall={handleRemoveWorker}
            />
          </TabsContent>

          {/* Active Workers Tab */}
          <TabsContent value="active" className="space-y-4">
            <ActiveWorkersView
              workers={workers}
              loading={loading}
              error={error}
              onTerminate={handleTerminateWorker}
            />
          </TabsContent>

          {/* Spawn Worker Tab - REQUIRES INSTALLED WORKERS + MODELS */}
          <TabsContent value="spawn" className="space-y-4">
            <SpawnWorkerView
              models={models}
              onSpawn={handleSpawnWorker}
              isPending={isPending}
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

// Re-export types for convenience
export type { ViewMode, SpawnFormState } from './types'
