// TEAM-502: Worker Filter Component
// Shows available workers as checkboxes

import React from 'react'
import type { HFFilterWorker } from './HFFilterSidebar'
import { Checkbox } from '@rbee/ui/atoms/Checkbox'
import { Label } from '@rbee/ui/atoms/Label'

interface WorkerFilterProps {
  workers: HFFilterWorker[]
  selectedWorkers: string[]
  onWorkersChange: (workers: string[]) => void
}

/**
 * Worker checkbox list component
 */
export const WorkerFilter: React.FC<WorkerFilterProps> = ({
  workers,
  selectedWorkers,
  onWorkersChange
}) => {
  const handleWorkerToggle = (workerId: string) => {
    if (selectedWorkers.includes(workerId)) {
      // Remove worker
      onWorkersChange(selectedWorkers.filter(id => id !== workerId))
    } else {
      // Add worker
      onWorkersChange([...selectedWorkers, workerId])
    }
  }

  const getWorkerDescription = (worker: HFFilterWorker) => {
    if (worker.marketplaceCompatibility.huggingface) {
      const compat = worker.marketplaceCompatibility.huggingface
      const taskCount = compat.tasks.length
      const formatCount = compat.formats.length
      
      if (taskCount > 0 && formatCount > 0) {
        return `${taskCount} task${taskCount > 1 ? 's' : ''}, ${formatCount} format${formatCount > 1 ? 's' : ''}`
      }
    }
    return worker.description
  }

  const getWorkerIcon = (worker: HFFilterWorker) => {
    // Return appropriate icon based on worker type
    if (worker.id.includes('llm')) {
      return '🤖' // LLM icon
    } else if (worker.id.includes('sd')) {
      return '🎨' // Image generation icon
    } else if (worker.id.includes('audio')) {
      return '🎵' // Audio icon
    } else {
      return '⚙️' // Generic worker icon
    }
  }

  return (
    <div className="space-y-2">
      {workers.map((worker) => {
        const isSelected = selectedWorkers.includes(worker.id)
        
        return (
          <div
            key={worker.id}
            className={`
              flex items-start gap-3 p-3 rounded-lg transition-all
              ${isSelected 
                ? 'bg-blue-50 border border-blue-200 hover:bg-blue-100' 
                : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
              }
            `}
          >
            <Checkbox
              id={`worker-${worker.id}`}
              checked={isSelected}
              onCheckedChange={() => handleWorkerToggle(worker.id)}
            />
            <Label
              htmlFor={`worker-${worker.id}`}
              className="flex-1 min-w-0 cursor-pointer"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-lg">{getWorkerIcon(worker)}</span>
                <span className="font-medium text-gray-900 text-sm">
                  {worker.name}
                </span>
              </div>
              <p className="text-xs text-gray-600 leading-relaxed">
                {getWorkerDescription(worker)}
              </p>
              {worker.marketplaceCompatibility.huggingface && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {worker.marketplaceCompatibility.huggingface.tasks.slice(0, 3).map((task: string) => (
                    <span
                      key={task}
                      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800"
                    >
                      {task}
                    </span>
                  ))}
                  {worker.marketplaceCompatibility.huggingface.formats.slice(0, 2).map((format: string) => (
                    <span
                      key={format}
                      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800"
                    >
                      {format}
                    </span>
                  ))}
                </div>
              )}
            </Label>
          </div>
        )
      })}
      
      {workers.length === 0 && (
        <div className="text-center py-4 text-gray-500 text-sm">
          No workers available
        </div>
      )}
    </div>
  )
}

export default WorkerFilter
