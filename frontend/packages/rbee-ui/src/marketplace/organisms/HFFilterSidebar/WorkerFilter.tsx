// TEAM-502: Worker Filter Component
// Shows available workers as checkboxes

import React from 'react'
import { CheckCircle, Circle } from 'lucide-react'
import type { GWCWorker } from '@rbee/marketplace-core'

interface WorkerFilterProps {
  workers: GWCWorker[]
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

  const getWorkerDescription = (worker: GWCWorker) => {
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

  const getWorkerIcon = (worker: GWCWorker) => {
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
        const Icon = isSelected ? CheckCircle : Circle
        
        return (
          <label
            key={worker.id}
            className={`
              flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-all
              ${isSelected 
                ? 'bg-blue-50 border border-blue-200 hover:bg-blue-100' 
                : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
              }
            `}
          >
            <input
              type="checkbox"
              checked={isSelected}
              onChange={() => handleWorkerToggle(worker.id)}
              className="sr-only"
            />
            <Icon 
              className={`
                w-5 h-5 mt-0.5 flex-shrink-0 transition-colors
                ${isSelected ? 'text-blue-600' : 'text-gray-400'}
              `} 
            />
            <div className="flex-1 min-w-0">
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
            </div>
          </label>
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
