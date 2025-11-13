# HFFilterSidebar Usage Example

**TEAM-502** | **Date:** 2025-11-13

## Quick Start

```tsx
import React, { useState, useEffect } from 'react'
import { HFFilterSidebar, type HFFilterState, type HFFilterOptions } from '@rbee/rbee-ui'
import type { GWCWorker } from '@rbee/marketplace-core'

export const HuggingFaceModelsPage: React.FC = () => {
  // State management
  const [filters, setFilters] = useState<HFFilterState>({
    workers: [],
    tasks: [],
    libraries: [],
    formats: [],
    languages: [],
    licenses: [],
    minParameters: undefined,
    maxParameters: undefined,
    sort: 'downloads',
    direction: -1
  })
  
  const [searchQuery, setSearchQuery] = useState('')
  const [workers, setWorkers] = useState<GWCWorker[]>([])
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  
  // Fetch workers from GWC API
  useEffect(() => {
    const fetchWorkers = async () => {
      try {
        const response = await fetch('/api/gwc/workers')
        const data = await response.json()
        setWorkers(data.workers)
      } catch (error) {
        console.error('Failed to fetch workers:', error)
      }
    }
    
    fetchWorkers()
  }, [])
  
  // Build filter options from workers
  const filterOptions: HFFilterOptions = {
    availableWorkers: workers,
    availableTasks: getAvailableTasks(workers),
    availableLibraries: getAvailableLibraries(workers),
    availableFormats: getAvailableFormats(workers),
    availableLanguages: getAvailableLanguages(workers),
    availableLicenses: getAvailableLicenses(workers)
  }
  
  // Fetch models when filters change
  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true)
      try {
        const queryParams = buildHuggingFaceQuery(filters, searchQuery)
        const response = await fetch(`/api/huggingface/models?${queryParams}`)
        const data = await response.json()
        
        // Apply client-side filtering for languages/parameters
        const filteredModels = applyClientSideFilters(data.models, filters)
        setModels(filteredModels)
      } catch (error) {
        console.error('Failed to fetch models:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchModels()
  }, [filters, searchQuery, workers])
  
  // Handle URL parameters (for sharing)
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const workerParam = urlParams.get('worker')
    
    if (workerParam) {
      setFilters(prev => ({
        ...prev,
        workers: [workerParam]
      }))
    }
  }, [])
  
  // Update URL when filters change
  useEffect(() => {
    const url = new URL(window.location.href)
    
    // Clear existing params
    url.searchParams.delete('worker')
    url.searchParams.delete('task')
    url.searchParams.delete('format')
    
    // Add current params
    if (filters.workers.length > 0) {
      url.searchParams.set('worker', filters.workers[0])
    }
    
    window.history.replaceState({}, '', url.toString())
  }, [filters])
  
  return (
    <div className="flex h-screen bg-gray-50">
      {/* Filter Sidebar */}
      <HFFilterSidebar
        filters={filters}
        options={filterOptions}
        searchQuery={searchQuery}
        onFiltersChange={setFilters}
        onSearchChange={setSearchQuery}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
      />
      
      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        {/* Header with active filters */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold text-gray-900">
              HuggingFace Models
            </h1>
            
            {/* Mobile toggle */}
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="lg:hidden p-2 hover:bg-gray-100 rounded-lg"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
          
          {/* Active Filters Bar */}
          <ActiveFiltersBar
            filters={filters}
            workers={workers}
            onFilterRemove={(filter, value) => {
              setFilters(prev => ({
                ...prev,
                [filter]: Array.isArray(prev[filter as keyof HFFilterState])
                  ? (prev[filter as keyof HFFilterState] as string[]).filter(v => v !== value)
                  : undefined
              }))
            }}
          />
        </div>
        
        {/* Models Grid */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {models.map((model) => (
                <ModelCard key={model.id} model={model} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Helper functions
function getAvailableTasks(workers: GWCWorker[]): string[] {
  const tasks = new Set<string>()
  workers.forEach(worker => {
    if (worker.marketplaceCompatibility.huggingface) {
      worker.marketplaceCompatibility.huggingface.tasks.forEach(task => 
        tasks.add(task)
      )
    }
  })
  return Array.from(tasks)
}

function getAvailableLibraries(workers: GWCWorker[]): string[] {
  const libraries = new Set<string>()
  workers.forEach(worker => {
    if (worker.marketplaceCompatibility.huggingface) {
      worker.marketplaceCompatibility.huggingface.libraries.forEach(library => 
        libraries.add(library)
      )
    }
  })
  return Array.from(libraries)
}

function getAvailableFormats(workers: GWCWorker[]): string[] {
  const formats = new Set<string>()
  workers.forEach(worker => {
    if (worker.marketplaceCompatibility.huggingface) {
      worker.marketplaceCompatibility.huggingface.formats.forEach(format => 
        formats.add(format)
      )
    }
  })
  return Array.from(formats)
}

function getAvailableLanguages(workers: GWCWorker[]): string[] {
  const languages = new Set<string>()
  workers.forEach(worker => {
    if (worker.marketplaceCompatibility.huggingface?.languages) {
      worker.marketplaceCompatibility.huggingface.languages.forEach(language => 
        languages.add(language)
      )
    }
  })
  return Array.from(languages)
}

function getAvailableLicenses(workers: GWCWorker[]): string[] {
  const licenses = new Set<string>()
  workers.forEach(worker => {
    if (worker.marketplaceCompatibility.huggingface?.licenses) {
      worker.marketplaceCompatibility.huggingface.licenses.forEach(license => 
        licenses.add(license)
      )
    }
  })
  return Array.from(licenses)
}

function buildHuggingFaceQuery(filters: HFFilterState, searchQuery: string): string {
  const params = new URLSearchParams()
  
  // Add search
  if (searchQuery) {
    params.set('search', searchQuery)
  }
  
  // Add sorting
  params.set('sort', filters.sort)
  params.set('direction', filters.direction.toString())
  
  // Add filters from selected workers
  if (filters.workers.length > 0) {
    const selectedWorkers = workers.filter(w => filters.workers.includes(w.id))
    
    // Combine all compatible filters from selected workers
    const allTasks = new Set<string>()
    const allLibraries = new Set<string>()
    const allFormats = new Set<string>()
    
    selectedWorkers.forEach(worker => {
      if (worker.marketplaceCompatibility.huggingface) {
        worker.marketplaceCompatibility.huggingface.tasks.forEach(task => 
          allTasks.add(task)
        )
        worker.marketplaceCompatibility.huggingface.libraries.forEach(library => 
          allLibraries.add(library)
        )
        worker.marketplaceCompatibility.huggingface.formats.forEach(format => 
          allFormats.add(format)
        )
      }
    })
    
    // Add to query if user has selected specific filters
    if (filters.tasks.length > 0) {
      params.set('pipeline_tag', filters.tasks.join(','))
    } else if (allTasks.size > 0) {
      params.set('pipeline_tag', Array.from(allTasks).join(','))
    }
    
    if (filters.libraries.length > 0) {
      params.set('library', filters.libraries.join(','))
    } else if (allLibraries.size > 0) {
      params.set('library', Array.from(allLibraries).join(','))
    }
    
    if (filters.formats.length > 0) {
      params.set('filter', filters.formats.join(','))
    } else if (allFormats.size > 0) {
      params.set('filter', Array.from(allFormats).join(','))
    }
    
    // Add license filters
    if (filters.licenses && filters.licenses.length > 0) {
      params.set('filter', `${params.get('filter') || ''},${filters.licenses.join(',')}`)
    }
  }
  
  params.set('limit', '50')
  
  return params.toString()
}

function applyClientSideFilters(models: any[], filters: HFFilterState): any[] {
  return models.filter(model => {
    // Filter by languages (client-side)
    if (filters.languages && filters.languages.length > 0) {
      const modelLanguages = model.tags?.filter((tag: string) => 
        filters.languages!.some(lang => tag.includes(lang))
      ) || []
      
      if (modelLanguages.length === 0) {
        return false
      }
    }
    
    // Filter by parameters (client-side)
    if (filters.minParameters !== undefined || filters.maxParameters !== undefined) {
      // Extract parameter count from model ID or tags
      const paramMatch = model.id.match(/(\d+(?:\.\d+)?)b/i)
      if (!paramMatch) return true // Skip if can't determine size
      
      const modelParams = parseFloat(paramMatch[1])
      
      if (filters.minParameters !== undefined && modelParams < filters.minParameters) {
        return false
      }
      
      if (filters.maxParameters !== undefined && modelParams > filters.maxParameters) {
        return false
      }
    }
    
    return true
  })
}
```

## Active Filters Bar Component

```tsx
interface ActiveFiltersBarProps {
  filters: HFFilterState
  workers: GWCWorker[]
  onFilterRemove: (filter: string, value: string) => void
}

const ActiveFiltersBar: React.FC<ActiveFiltersBarProps> = ({
  filters,
  workers,
  onFilterRemove
}) => {
  const hasActiveFilters = Object.entries(filters).some(([key, value]) => {
    if (key === 'sort' || key === 'direction') return false
    if (Array.isArray(value)) return value.length > 0
    return value !== undefined && value !== null
  })
  
  if (!hasActiveFilters) {
    return (
      <div className="text-sm text-gray-500">
        Showing all models. Use filters to narrow results.
      </div>
    )
  }
  
  const getFilterDisplay = (filter: string, value: string): string => {
    switch (filter) {
      case 'workers':
        const worker = workers.find(w => w.id === value)
        return worker?.name || value
      case 'tasks':
        return value.split('-').map(word => 
          word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ')
      case 'formats':
        return value.toUpperCase()
      case 'languages':
        const languageNames: Record<string, string> = {
          'en': 'English',
          'zh': 'Chinese',
          'es': 'Spanish',
          'fr': 'French',
          'de': 'German',
          'ja': 'Japanese',
          'ko': 'Korean',
          'multilingual': 'Multilingual'
        }
        return languageNames[value] || value
      default:
        return value
    }
  }
  
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm text-gray-500">Active filters:</span>
      
      {filters.workers.map(workerId => (
        <span
          key={`worker-${workerId}`}
          className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm"
        >
          🔧 {getFilterDisplay('workers', workerId)}
          <button
            onClick={() => onFilterRemove('workers', workerId)}
            className="hover:text-blue-900"
          >
            ×
          </button>
        </span>
      ))}
      
      {filters.tasks.map(task => (
        <span
          key={`task-${task}`}
          className="inline-flex items-center gap-1 px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm"
        >
          📝 {getFilterDisplay('tasks', task)}
          <button
            onClick={() => onFilterRemove('tasks', task)}
            className="hover:text-green-900"
          >
            ×
          </button>
        </span>
      ))}
      
      {filters.formats.map(format => (
        <span
          key={`format-${format}`}
          className="inline-flex items-center gap-1 px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm"
        >
          📦 {getFilterDisplay('formats', format)}
          <button
            onClick={() => onFilterRemove('formats', format)}
            className="hover:text-purple-900"
          >
            ×
          </button>
        </span>
      ))}
      
      {(filters.minParameters !== undefined || filters.maxParameters !== undefined) && (
        <span className="inline-flex items-center gap-1 px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm">
          📊 {filters.minParameters || 0}B - {filters.maxParameters || '∞'}B
          <button
            onClick={() => {
              onFilterRemove('minParameters', '')
              onFilterRemove('maxParameters', '')
            }}
            className="hover:text-orange-900"
          >
            ×
          </button>
        </span>
      )}
      
      {filters.languages?.map(language => (
        <span
          key={`language-${language}`}
          className="inline-flex items-center gap-1 px-3 py-1 bg-cyan-100 text-cyan-700 rounded-full text-sm"
        >
          🌍 {getFilterDisplay('languages', language)}
          <button
            onClick={() => onFilterRemove('languages', language)}
            className="hover:text-cyan-900"
          >
            ×
          </button>
        </span>
      ))}
    </div>
  )
}
```

## Key Integration Points

1. **Fetch Workers from GWC API** - Get available workers and their compatibility
2. **Build Filter Options** - Extract available tasks, formats, etc. from workers
3. **Handle Filter Changes** - Update URL params and fetch models
4. **Build HuggingFace Query** - Convert filters to API parameters
5. **Client-Side Filtering** - Apply language/parameter filters
6. **Active Filters Bar** - Show selected filters with remove buttons

## Mobile Responsiveness

- Use `collapsed` prop for mobile view
- Add toggle button in header
- Sidebar becomes icon-only when collapsed
- Filters are preserved when collapsing/expanding

## URL Sharing

- Filters are encoded in URL parameters
- `?worker=llm-worker-rbee&task=text-generation&format=gguf`
- Page loads with filters applied from URL
- URL updates when filters change
