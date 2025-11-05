// TEAM-410: Worker compatibility list component
// Shows list of workers with compatibility status

import { CompatibilityBadge } from '../atoms/CompatibilityBadge'
import type { CompatibilityResult, Worker } from '../types/compatibility'

interface WorkerCompatibilityListProps {
  workers: Array<{
    worker: Worker
    compatibility: CompatibilityResult
  }>
  className?: string
}

export function WorkerCompatibilityList({ workers, className = '' }: WorkerCompatibilityListProps) {
  // Group by compatibility
  const compatible = workers.filter(w => w.compatibility.compatible)
  const incompatible = workers.filter(w => !w.compatibility.compatible)
  
  return (
    <div className={`space-y-6 ${className}`}>
      {/* Compatible Workers */}
      {compatible.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Compatible Workers ({compatible.length})</h3>
          <div className="grid gap-3">
            {compatible.map(({ worker, compatibility }) => (
              <div
                key={worker.id}
                className="border rounded-lg p-4 hover:border-green-500 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{worker.name}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {worker.worker_type.toUpperCase()} • {worker.platform}
                    </p>
                  </div>
                  <CompatibilityBadge 
                    result={compatibility} 
                    workerName={worker.name} 
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Incompatible Workers */}
      {incompatible.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Incompatible Workers ({incompatible.length})</h3>
          <div className="grid gap-3 opacity-60">
            {incompatible.map(({ worker, compatibility }) => (
              <div
                key={worker.id}
                className="border rounded-lg p-4"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{worker.name}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {worker.worker_type.toUpperCase()} • {worker.platform}
                    </p>
                  </div>
                  <CompatibilityBadge 
                    result={compatibility} 
                    workerName={worker.name} 
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* No workers */}
      {workers.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <p>No worker compatibility data available</p>
        </div>
      )}
    </div>
  )
}
