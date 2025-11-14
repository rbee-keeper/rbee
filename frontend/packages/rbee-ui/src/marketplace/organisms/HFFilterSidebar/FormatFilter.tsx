// TEAM-502: Format Filter Component
// Shows available formats and libraries as checkboxes

import { ChevronDown, ChevronUp, Info } from 'lucide-react'
import type React from 'react'
import { useState } from 'react'
import { Checkbox } from '@rbee/ui/atoms/Checkbox'
import { Label } from '@rbee/ui/atoms/Label'
import { Button } from '@rbee/ui/atoms/Button'

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

  const getColorClasses = (isSelected: boolean) => {
    if (isSelected) {
      return {
        bg: 'bg-sidebar-accent/10',
        text: 'text-sidebar-foreground',
        border: 'border-sidebar-accent',
      }
    }

    return {
      bg: 'bg-muted',
      text: 'text-sidebar-foreground',
      border: 'border-sidebar-border',
    }
  }

  return (
    <div className="space-y-4">
      {/* Formats Section */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-sm font-medium text-sidebar-foreground">File Formats</h4>
          <Info className="w-3 h-3 text-muted-foreground" />
        </div>

        <div className="space-y-2">
          {displayedFormats.map((format) => {
            const isSelected = selectedFormats.includes(format)
            const info = getFormatInfo(format)
            const colors = getColorClasses(isSelected)

            return (
              <div
                key={format}
                className={`
                  flex items-start gap-3 p-2 rounded-lg transition-all
                  ${colors?.bg || ''} ${colors?.border || ''} border hover:opacity-80
                `}
              >
                <Checkbox
                  id={`format-${format}`}
                  checked={isSelected}
                  onCheckedChange={() => handleFormatToggle(format)}
                />
                <Label
                  htmlFor={`format-${format}`}
                  className="flex-1 min-w-0 cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{info.icon}</span>
                    <span className={`font-medium text-sm ${colors?.text || ''}`}>{format}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">{info.description}</p>
                </Label>
              </div>
            )
          })}

          {hasMoreFormats && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setShowAllFormats(!showAllFormats)}
              className="flex items-center gap-2 text-sm text-primary hover:text-primary/80"
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
            </Button>
          )}
        </div>
      </div>

      {/* Libraries Section */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-sm font-medium text-sidebar-foreground">Libraries</h4>
          <Info className="w-3 h-3 text-muted-foreground" />
        </div>

        <div className="space-y-2">
          {displayedLibraries.map((library) => {
            const isSelected = selectedLibraries.includes(library)
            const info = getLibraryInfo(library)
            const colors = getColorClasses(isSelected)

            return (
              <div
                key={library}
                className={`
                  flex items-start gap-3 p-2 rounded-lg transition-all
                  ${colors?.bg || ''} ${colors?.border || ''} border hover:opacity-80
                `}
              >
                <Checkbox
                  id={`library-${library}`}
                  checked={isSelected}
                  onCheckedChange={() => handleLibraryToggle(library)}
                />
                <Label
                  htmlFor={`library-${library}`}
                  className="flex-1 min-w-0 cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{info.icon}</span>
                    <span className={`font-medium text-sm ${colors?.text || ''}`}>{library}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">{info.description}</p>
                </Label>
              </div>
            )
          })}

          {hasMoreLibraries && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setShowAllLibraries(!showAllLibraries)}
              className="flex items-center gap-2 text-sm text-primary hover:text-primary/80"
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
            </Button>
          )}
        </div>
      </div>

      {formats.length === 0 && libraries.length === 0 && (
        <div className="text-center py-4 text-muted-foreground text-sm">
          No formats or libraries available. Select a worker first.
        </div>
      )}
    </div>
  )
}

export default FormatFilter
