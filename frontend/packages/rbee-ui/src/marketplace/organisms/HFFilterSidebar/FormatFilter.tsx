// TEAM-502: Format Filter Component
// Shows available formats and libraries as checkboxes

import { CheckCircle, ChevronDown, ChevronUp, Circle, Info } from 'lucide-react'
import type React from 'react'
import { useState } from 'react'

interface FormatFilterProps {
  formats: string[]
  libraries: string[]
  selectedFormats: string[]
  selectedLibraries: string[]
  onFormatsChange: (formats: string[]) => void
  onLibrariesChange: (libraries: string[]) => void
}

/**
 * Format and Library checkbox list component
 */
export const FormatFilter: React.FC<FormatFilterProps> = ({
  formats,
  libraries,
  selectedFormats,
  selectedLibraries,
  onFormatsChange,
  onLibrariesChange,
}) => {
  const [showAllFormats, setShowAllFormats] = useState(false)
  const [showAllLibraries, setShowAllLibraries] = useState(false)

  // Common formats to show by default
  const commonFormats = ['safetensors', 'gguf', 'pytorch']

  // Common libraries to show by default
  const commonLibraries = ['transformers', 'diffusers', 'sentence-transformers']

  // Sort: common items first, then alphabetical
  const sortedFormats = [...formats].sort((a, b) => {
    const aIsCommon = commonFormats.includes(a)
    const bIsCommon = commonFormats.includes(b)

    if (aIsCommon && !bIsCommon) return -1
    if (!aIsCommon && bIsCommon) return 1
    if (aIsCommon && bIsCommon) return commonFormats.indexOf(a) - commonFormats.indexOf(b)

    return a.localeCompare(b)
  })

  const sortedLibraries = [...libraries].sort((a, b) => {
    const aIsCommon = commonLibraries.includes(a)
    const bIsCommon = commonLibraries.includes(b)

    if (aIsCommon && !bIsCommon) return -1
    if (!aIsCommon && bIsCommon) return 1
    if (aIsCommon && bIsCommon) return commonLibraries.indexOf(a) - commonLibraries.indexOf(b)

    return a.localeCompare(b)
  })

  // Show first 6 items by default
  const displayedFormats = showAllFormats ? sortedFormats : sortedFormats.slice(0, 6)
  const displayedLibraries = showAllLibraries ? sortedLibraries : sortedLibraries.slice(0, 6)

  const hasMoreFormats = sortedFormats.length > 6
  const hasMoreLibraries = sortedLibraries.length > 6

  const handleFormatToggle = (format: string) => {
    if (selectedFormats.includes(format)) {
      onFormatsChange(selectedFormats.filter((f) => f !== format))
    } else {
      onFormatsChange([...selectedFormats, format])
    }
  }

  const handleLibraryToggle = (library: string) => {
    if (selectedLibraries.includes(library)) {
      onLibrariesChange(selectedLibraries.filter((l) => l !== library))
    } else {
      onLibrariesChange([...selectedLibraries, library])
    }
  }

  const getFormatInfo = (format: string) => {
    const info: Record<string, { icon: string; description: string; color: string }> = {
      safetensors: {
        icon: '🔒',
        description: 'Safe, fast loading format',
        color: 'green',
      },
      gguf: {
        icon: '🦙',
        description: 'llama.cpp compatible, quantized',
        color: 'purple',
      },
      pytorch: {
        icon: '🔥',
        description: 'PyTorch model format',
        color: 'orange',
      },
      onnx: {
        icon: '⚡',
        description: 'Cross-platform inference',
        color: 'blue',
      },
      tensorflow: {
        icon: '🧠',
        description: 'TensorFlow SavedModel',
        color: 'red',
      },
      jax: {
        icon: '🦋',
        description: 'JAX/Flax format',
        color: 'cyan',
      },
    }

    return (
      info[format] || {
        icon: '📄',
        description: 'Model format',
        color: 'gray',
      }
    )
  }

  const getLibraryInfo = (library: string) => {
    const info: Record<string, { icon: string; description: string; color: string }> = {
      transformers: {
        icon: '🤗',
        description: 'HuggingFace Transformers',
        color: 'yellow',
      },
      diffusers: {
        icon: '🎨',
        description: 'Diffusion models library',
        color: 'pink',
      },
      'sentence-transformers': {
        icon: '📝',
        description: 'Sentence embeddings',
        color: 'indigo',
      },
      timm: {
        icon: '🖼️',
        description: 'Image models library',
        color: 'teal',
      },
      'adapter-transformers': {
        icon: '🔌',
        description: 'Adapter models',
        color: 'lime',
      },
    }

    return (
      info[library] || {
        icon: '📚',
        description: 'Model library',
        color: 'gray',
      }
    )
  }

  const getColorClasses = (color: string, isSelected: boolean) => {
    const colors: Record<string, { bg: string; text: string; border: string }> = {
      green: {
        bg: isSelected ? 'bg-green-50' : 'bg-gray-50',
        text: isSelected ? 'text-green-700' : 'text-gray-700',
        border: isSelected ? 'border-green-200' : 'border-gray-200',
      },
      purple: {
        bg: isSelected ? 'bg-purple-50' : 'bg-gray-50',
        text: isSelected ? 'text-purple-700' : 'text-gray-700',
        border: isSelected ? 'border-purple-200' : 'border-gray-200',
      },
      orange: {
        bg: isSelected ? 'bg-orange-50' : 'bg-gray-50',
        text: isSelected ? 'text-orange-700' : 'text-gray-700',
        border: isSelected ? 'border-orange-200' : 'border-gray-200',
      },
      blue: {
        bg: isSelected ? 'bg-blue-50' : 'bg-gray-50',
        text: isSelected ? 'text-blue-700' : 'text-gray-700',
        border: isSelected ? 'border-blue-200' : 'border-gray-200',
      },
      red: {
        bg: isSelected ? 'bg-red-50' : 'bg-gray-50',
        text: isSelected ? 'text-red-700' : 'text-gray-700',
        border: isSelected ? 'border-red-200' : 'border-gray-200',
      },
      cyan: {
        bg: isSelected ? 'bg-cyan-50' : 'bg-gray-50',
        text: isSelected ? 'text-cyan-700' : 'text-gray-700',
        border: isSelected ? 'border-cyan-200' : 'border-gray-200',
      },
      yellow: {
        bg: isSelected ? 'bg-yellow-50' : 'bg-gray-50',
        text: isSelected ? 'text-yellow-700' : 'text-gray-700',
        border: isSelected ? 'border-yellow-200' : 'border-gray-200',
      },
      pink: {
        bg: isSelected ? 'bg-pink-50' : 'bg-gray-50',
        text: isSelected ? 'text-pink-700' : 'text-gray-700',
        border: isSelected ? 'border-pink-200' : 'border-gray-200',
      },
      indigo: {
        bg: isSelected ? 'bg-indigo-50' : 'bg-gray-50',
        text: isSelected ? 'text-indigo-700' : 'text-gray-700',
        border: isSelected ? 'border-indigo-200' : 'border-gray-200',
      },
      teal: {
        bg: isSelected ? 'bg-teal-50' : 'bg-gray-50',
        text: isSelected ? 'text-teal-700' : 'text-gray-700',
        border: isSelected ? 'border-teal-200' : 'border-gray-200',
      },
      lime: {
        bg: isSelected ? 'bg-lime-50' : 'bg-gray-50',
        text: isSelected ? 'text-lime-700' : 'text-gray-700',
        border: isSelected ? 'border-lime-200' : 'border-gray-200',
      },
      gray: {
        bg: isSelected ? 'bg-gray-100' : 'bg-gray-50',
        text: isSelected ? 'text-gray-800' : 'text-gray-700',
        border: isSelected ? 'border-gray-300' : 'border-gray-200',
      },
    }

    return colors[color] || colors.gray
  }

  return (
    <div className="space-y-4">
      {/* Formats Section */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-sm font-medium text-gray-900">File Formats</h4>
          <Info className="w-3 h-3 text-gray-400" title="Model file formats" />
        </div>

        <div className="space-y-2">
          {displayedFormats.map((format) => {
            const isSelected = selectedFormats.includes(format)
            const Icon = isSelected ? CheckCircle : Circle
            const info = getFormatInfo(format)
            const colors = getColorClasses(info.color, isSelected)

            return (
              <label
                key={format}
                className={`
                  flex items-start gap-3 p-2 rounded-lg cursor-pointer transition-all
                  ${colors.bg} ${colors.border} border hover:opacity-80
                `}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => handleFormatToggle(format)}
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
                    <span className="text-sm">{info.icon}</span>
                    <span className={`font-medium text-sm ${colors.text}`}>{format}</span>
                  </div>
                  <p className="text-xs text-gray-600 mt-0.5">{info.description}</p>
                </div>
              </label>
            )
          })}

          {hasMoreFormats && (
            <button
              onClick={() => setShowAllFormats(!showAllFormats)}
              className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors"
            >
              {showAllFormats ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show {sortedFormats.length - 6} more formats
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Libraries Section */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-sm font-medium text-gray-900">Libraries</h4>
          <Info className="w-3 h-3 text-gray-400" title="Model libraries/frameworks" />
        </div>

        <div className="space-y-2">
          {displayedLibraries.map((library) => {
            const isSelected = selectedLibraries.includes(library)
            const Icon = isSelected ? CheckCircle : Circle
            const info = getLibraryInfo(library)
            const colors = getColorClasses(info.color, isSelected)

            return (
              <label
                key={library}
                className={`
                  flex items-start gap-3 p-2 rounded-lg cursor-pointer transition-all
                  ${colors.bg} ${colors.border} border hover:opacity-80
                `}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => handleLibraryToggle(library)}
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
                    <span className="text-sm">{info.icon}</span>
                    <span className={`font-medium text-sm ${colors.text}`}>{library}</span>
                  </div>
                  <p className="text-xs text-gray-600 mt-0.5">{info.description}</p>
                </div>
              </label>
            )
          })}

          {hasMoreLibraries && (
            <button
              onClick={() => setShowAllLibraries(!showAllLibraries)}
              className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors"
            >
              {showAllLibraries ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show {sortedLibraries.length - 6} more libraries
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {formats.length === 0 && libraries.length === 0 && (
        <div className="text-center py-4 text-gray-500 text-sm">
          No formats or libraries available. Select a worker first.
        </div>
      )}
    </div>
  )
}

export default FormatFilter
