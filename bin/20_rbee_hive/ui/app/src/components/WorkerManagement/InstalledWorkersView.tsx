// Installed Workers View - Shows table of installed worker binaries
// Different from ActiveWorkersView which shows running processes
// TEAM-382: Wired up to useInstalledWorkers hook

import { useState } from 'react'
import { 
  Card, 
  CardContent
} from '@rbee/ui/atoms/Card'
import { Button } from '@rbee/ui/atoms/Button'
import { Badge } from '@rbee/ui/atoms/Badge'
import { 
  Trash2, 
  CheckCircle, 
  AlertCircle,
  Package,
  Cpu,
  Zap,
  Apple,
  Loader2
} from 'lucide-react'
import { useInstalledWorkers, type InstalledWorker as ApiInstalledWorker } from '@rbee/rbee-hive-react'

interface InstalledWorker {
  id: string
  name: string
  version: string
  workerType: 'cpu' | 'cuda' | 'metal'
  installedAt: string
  binaryPath: string
  size: string
}

interface InstalledWorkersViewProps {
  onUninstall?: (workerId: string) => void
}

// TEAM-382: Helper to convert API response to UI format
function convertWorker(apiWorker: ApiInstalledWorker): InstalledWorker {
  // Extract worker type from worker_type string (e.g., "CpuLlm" -> "cpu")
  const workerType = apiWorker.worker_type.toLowerCase().replace('llm', '') as 'cpu' | 'cuda' | 'metal'
  
  // Format size from bytes to human-readable
  const sizeInMB = (apiWorker.size / (1024 * 1024)).toFixed(1)
  
  return {
    id: apiWorker.id,
    name: apiWorker.name,
    version: apiWorker.version,
    workerType,
    installedAt: apiWorker.added_at,
    binaryPath: apiWorker.path,
    size: `${sizeInMB} MB`
  }
}

export function InstalledWorkersView({ onUninstall }: InstalledWorkersViewProps) {
  const [removingWorker, setRemovingWorker] = useState<string | null>(null)
  
  // TEAM-382: Fetch installed workers from catalog
  const { data: apiWorkers = [], isLoading, error } = useInstalledWorkers()
  
  // Convert API format to UI format
  const installedWorkers = apiWorkers.map(convertWorker)
  
  const getWorkerIcon = (workerType: string) => {
    switch (workerType) {
      case 'cpu':
        return <Cpu className="h-4 w-4" />
      case 'cuda':
        return <Zap className="h-4 w-4" />
      case 'metal':
        return <Apple className="h-4 w-4" />
      default:
        return <Package className="h-4 w-4" />
    }
  }
  
  const getWorkerTypeBadge = (workerType: string) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      cpu: 'secondary',
      cuda: 'default',
      metal: 'outline'
    }
    return (
      <Badge variant={variants[workerType] || 'secondary'} className="flex items-center gap-1">
        {getWorkerIcon(workerType)}
        {workerType.toUpperCase()}
      </Badge>
    )
  }
  
  const handleUninstall = async (workerId: string) => {
    setRemovingWorker(workerId)
    try {
      await onUninstall?.(workerId)
    } finally {
      setRemovingWorker(null)
    }
  }
  
  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }
  
  // TEAM-382: Show loading state
  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12">
            <Loader2 className="h-12 w-12 text-muted-foreground mb-4 animate-spin" />
            <p className="text-sm text-muted-foreground">Loading installed workers...</p>
          </div>
        </CardContent>
      </Card>
    )
  }
  
  // TEAM-382: Show error state
  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="h-12 w-12 text-destructive mb-4" />
            <h3 className="text-lg font-semibold mb-2">Failed to Load Workers</h3>
            <p className="text-sm text-muted-foreground">{error.message}</p>
          </div>
        </CardContent>
      </Card>
    )
  }
  
  if (installedWorkers.length === 0) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Package className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Workers Installed</h3>
            <p className="text-sm text-muted-foreground mb-4 max-w-md">
              Install worker binaries from the Worker Catalog to get started.
              Workers are compiled binaries that can spawn processes to handle inference requests.
            </p>
            <p className="text-xs text-muted-foreground">
              Go to <strong>Worker Catalog</strong> tab to install workers
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }
  
  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h3 className="text-lg font-semibold">Installed Worker Binaries</h3>
        <p className="text-sm text-muted-foreground">
          Compiled worker binaries ready to spawn processes
        </p>
      </div>
      
      {/* Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b bg-muted/50">
                <tr>
                  <th className="text-left p-4 font-medium text-sm">Worker</th>
                  <th className="text-left p-4 font-medium text-sm">Type</th>
                  <th className="text-left p-4 font-medium text-sm">Version</th>
                  <th className="text-left p-4 font-medium text-sm">Installed</th>
                  <th className="text-left p-4 font-medium text-sm">Size</th>
                  <th className="text-left p-4 font-medium text-sm">Binary Path</th>
                  <th className="text-right p-4 font-medium text-sm">Actions</th>
                </tr>
              </thead>
              <tbody>
                {installedWorkers.map((worker) => (
                  <tr key={worker.id} className="border-b last:border-0 hover:bg-muted/30">
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="font-medium">{worker.name}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      {getWorkerTypeBadge(worker.workerType)}
                    </td>
                    <td className="p-4">
                      <code className="text-xs bg-muted px-2 py-1 rounded">
                        v{worker.version}
                      </code>
                    </td>
                    <td className="p-4 text-sm text-muted-foreground">
                      {formatDate(worker.installedAt)}
                    </td>
                    <td className="p-4 text-sm text-muted-foreground">
                      {worker.size}
                    </td>
                    <td className="p-4">
                      <code className="text-xs text-muted-foreground">
                        {worker.binaryPath}
                      </code>
                    </td>
                    <td className="p-4 text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleUninstall(worker.id)}
                        disabled={removingWorker === worker.id}
                      >
                        {removingWorker === worker.id ? (
                          <>
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            Removing...
                          </>
                        ) : (
                          <>
                            <Trash2 className="h-4 w-4 mr-2" />
                            Uninstall
                          </>
                        )}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
      
      {/* Info Card */}
      <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <AlertCircle className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
                About Installed Workers
              </p>
              <p className="text-xs text-blue-700 dark:text-blue-300">
                These are compiled binaries that can spawn worker processes. To start a worker process,
                go to the <strong>Spawn Worker</strong> tab and select a model.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
