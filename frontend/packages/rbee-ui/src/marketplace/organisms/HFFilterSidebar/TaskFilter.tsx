// TEAM-502: Task Filter Component
// Shows available HuggingFace tasks as checkboxes

import React, { useState } from 'react'
import { CheckCircle, Circle, ChevronDown, ChevronUp } from 'lucide-react'

interface TaskFilterProps {
  tasks: string[]
  selectedTasks: string[]
  onTasksChange: (tasks: string[]) => void
}

/**
 * Task checkbox list component
 */
export const TaskFilter: React.FC<TaskFilterProps> = ({
  tasks,
  selectedTasks,
  onTasksChange
}) => {
  const [showAll, setShowAll] = useState(false)
  
  // Common tasks to show by default
  const commonTasks = [
    'text-generation',
    'text-to-image',
    'image-to-text',
    'text-to-speech',
    'automatic-speech-recognition',
    'summarization',
    'translation',
    'text-classification'
  ]
  
  // Sort tasks: common tasks first, then alphabetical
  const sortedTasks = [...tasks].sort((a, b) => {
    const aIsCommon = commonTasks.includes(a)
    const bIsCommon = commonTasks.includes(b)
    
    if (aIsCommon && !bIsCommon) return -1
    if (!aIsCommon && bIsCommon) return 1
    if (aIsCommon && bIsCommon) return commonTasks.indexOf(a) - commonTasks.indexOf(b)
    
    return a.localeCompare(b)
  })
  
  // Show first 8 tasks by default, or all if showAll is true
  const displayedTasks = showAll ? sortedTasks : sortedTasks.slice(0, 8)
  const hasMore = sortedTasks.length > 8
  
  const handleTaskToggle = (task: string) => {
    if (selectedTasks.includes(task)) {
      // Remove task
      onTasksChange(selectedTasks.filter(t => t !== task))
    } else {
      // Add task
      onTasksChange([...selectedTasks, task])
    }
  }
  
  const getTaskIcon = (task: string) => {
    // Return appropriate icon based on task type
    if (task.includes('text-generation')) {
      return '💬'
    } else if (task.includes('text-to-image')) {
      return '🖼️'
    } else if (task.includes('image-to-text')) {
      return '👁️'
    } else if (task.includes('text-to-speech')) {
      return '🔊'
    } else if (task.includes('speech-recognition')) {
      return '🎤'
    } else if (task.includes('summarization')) {
      return '📝'
    } else if (task.includes('translation')) {
      return '🌐'
    } else if (task.includes('classification')) {
      return '🏷️'
    } else if (task.includes('question-answering')) {
      return '❓'
    } else if (task.includes('fill-mask')) {
      return '🔤'
    } else {
      return '📋'
    }
  }
  
  const getTaskDisplayName = (task: string) => {
    // Convert kebab-case to readable format
    return task
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }
  
  const getTaskDescription = (task: string) => {
    // Add helpful descriptions for common tasks
    const descriptions: Record<string, string> = {
      'text-generation': 'Generate text, chat, complete prompts',
      'text-to-image': 'Generate images from text descriptions',
      'image-to-text': 'Describe images, OCR, visual understanding',
      'text-to-speech': 'Convert text to spoken audio',
      'automatic-speech-recognition': 'Convert speech to text',
      'summarization': 'Summarize long documents or text',
      'translation': 'Translate between languages',
      'text-classification': 'Classify text into categories',
      'question-answering': 'Answer questions based on context',
      'fill-mask': 'Fill in missing words in text',
      'token-classification': 'Identify entities, parts of speech',
      'conversational': 'Chat and dialogue systems',
      'feature-extraction': 'Extract features from text',
      'sentence-similarity': 'Compare sentence similarity',
      'zero-shot-classification': 'Classify without training data'
    }
    
    return descriptions[task] || ''
  }

  return (
    <div className="space-y-2">
      {displayedTasks.map((task) => {
        const isSelected = selectedTasks.includes(task)
        const Icon = isSelected ? CheckCircle : Circle
        const description = getTaskDescription(task)
        
        return (
          <label
            key={task}
            className={`
              flex items-start gap-3 p-2 rounded-lg cursor-pointer transition-all
              ${isSelected 
                ? 'bg-blue-50 border border-blue-200 hover:bg-blue-100' 
                : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
              }
            `}
          >
            <input
              type="checkbox"
              checked={isSelected}
              onChange={() => handleTaskToggle(task)}
              className="sr-only"
            />
            <Icon 
              className={`
                w-4 h-4 mt-0.5 flex-shrink-0 transition-colors
                ${isSelected ? 'text-blue-600' : 'text-gray-400'}
              `} 
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm">{getTaskIcon(task)}</span>
                <span className="font-medium text-gray-900 text-sm">
                  {getTaskDisplayName(task)}
                </span>
              </div>
              {description && (
                <p className="text-xs text-gray-600 mt-0.5">
                  {description}
                </p>
              )}
            </div>
          </label>
        )
      })}
      
      {hasMore && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors mt-2"
        >
          {showAll ? (
            <>
              <ChevronUp className="w-4 h-4" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="w-4 h-4" />
              Show {sortedTasks.length - 8} more tasks
            </>
          )}
        </button>
      )}
      
      {tasks.length === 0 && (
        <div className="text-center py-4 text-gray-500 text-sm">
          No tasks available. Select a worker first.
        </div>
      )}
    </div>
  )
}

export default TaskFilter
