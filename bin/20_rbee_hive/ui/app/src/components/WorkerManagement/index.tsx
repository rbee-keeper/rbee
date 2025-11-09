// TEAM-382: Worker Management - Main component with clean composition
// TEAM-405: Removed "Catalog" tab - marketplace search moved to separate component
// TEAM-405: Now focuses on LOCAL CATALOG management (Installed, Active, Spawn)

import { useModels, useWorkerOperations, useWorkers } from '@rbee/rbee-hive-react'
import {
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@rbee/ui/atoms'
import { Activity, Plus, Server } from 'lucide-react'
import { useState } from 'react'
import { ActiveWorkersView } from './ActiveWorkersView'
import { InstalledWorkersView } from './InstalledWorkersView'
import { SpawnWorkerView } from './SpawnWorkerView'
import type { SpawnFormState, ViewMode } from './types'

export function WorkerManagement() {
  const { workers, loading, error } = useWorkers()
  const { models } = useModels()

  const { spawnWorker, terminateWorker, isPending } = useWorkerOperations()

  const [viewMode, setViewMode] = useState<ViewMode>('installed') // Start with installed workers

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

  // TEAM-405: Removed handleInstallWorker and handleRemoveWorker - use MarketplaceSearch component

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
            <CardDescription>Install workers, monitor performance, and manage processes</CardDescription>
          </div>
          <div className="flex gap-2">
            <Badge variant="secondary">{idleWorkers.length} Idle</Badge>
            <Badge variant="default">{activeWorkers.length} Active</Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* TEAM-405: Removed "Catalog" tab - use MarketplaceSearch component instead */}
        {/* View Mode Tabs */}
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList className="grid w-full grid-cols-3">
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

          {/* Installed Workers Tab */}
          <TabsContent value="installed" className="space-y-4">
            <InstalledWorkersView />
          </TabsContent>

          {/* Active Workers Tab */}
          <TabsContent value="active" className="space-y-4">
            <ActiveWorkersView workers={workers} loading={loading} error={error} onTerminate={handleTerminateWorker} />
          </TabsContent>

          {/* Spawn Worker Tab - REQUIRES INSTALLED WORKERS + MODELS */}
          <TabsContent value="spawn" className="space-y-4">
            <SpawnWorkerView models={models} onSpawn={handleSpawnWorker} isPending={isPending} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

// Re-export types for convenience
export type { SpawnFormState, ViewMode } from './types'
