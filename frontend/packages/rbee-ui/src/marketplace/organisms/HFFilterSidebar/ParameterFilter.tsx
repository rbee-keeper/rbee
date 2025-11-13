// TEAM-502: Parameter Filter Component
// Shows parameter size range slider

import React, { useState, useCallback } from 'react'

interface ParameterFilterProps {
  min: number
  max: number
  selectedMin?: number
  selectedMax?: number
  onParametersChange: (min?: number, max?: number) => void
}

/**
 * Parameter size range slider component
 */
export const ParameterFilter: React.FC<ParameterFilterProps> = ({
  min,
  max,
  selectedMin,
  selectedMax,
  onParametersChange
}) => {
  // Initialize with full range if no selection
  const [localMin, setLocalMin] = useState(selectedMin ?? min)
  const [localMax, setLocalMax] = useState(selectedMax ?? max)
  
  // Common parameter size presets
  const presets = [
    { label: 'Tiny (<1B)', min: min, max: 1 },
    { label: 'Small (1-3B)', min: 1, max: 3 },
    { label: 'Medium (3-7B)', min: 3, max: 7 },
    { label: 'Large (7-13B)', min: 7, max: 13 },
    { label: 'XL (13-30B)', min: 13, max: 30 },
    { label: 'XXL (30B+)', min: 30, max: max }
  ]
  
  // Format number for display
  const formatNumber = (num: number) => {
    if (num < 1) {
      return `${(num * 1000).toFixed(0)}M`
    } else if (num < 1000) {
      return `${num.toFixed(1)}B`
    } else {
      return `${(num / 1000).toFixed(1)}T`
    }
  }
  
  // Handle slider change
  const handleSliderChange = useCallback((values: [number, number]) => {
    const [newMin, newMax] = values
    setLocalMin(newMin)
    setLocalMax(newMax)
    
    // Only call onChange if values are different from defaults
    if (newMin !== min || newMax !== max) {
      onParametersChange(newMin, newMax)
    } else {
      // Reset to undefined if back to full range
      onParametersChange(undefined, undefined)
    }
  }, [min, max, onParametersChange])
  
  // Handle preset click
  const handlePresetClick = (presetMin: number, presetMax: number) => {
    setLocalMin(presetMin)
    setLocalMax(presetMax)
    
    if (presetMin === min && presetMax === max) {
      onParametersChange(undefined, undefined)
    } else {
      onParametersChange(presetMin, presetMax)
    }
  }
  
  // Handle manual input
  const handleMinInputChange = (value: string) => {
    const num = parseFloat(value)
    if (!isNaN(num) && num >= min && num <= max) {
      setLocalMin(num)
      onParametersChange(num, localMax)
    }
  }
  
  const handleMaxInputChange = (value: string) => {
    const num = parseFloat(value)
    if (!isNaN(num) && num >= min && num <= max) {
      setLocalMax(num)
      onParametersChange(localMin, num)
    }
  }
  
  // Check if current selection matches a preset
  const getActivePreset = () => {
    return presets.findIndex(p => 
      Math.abs(p.min - localMin) < 0.1 && Math.abs(p.max - localMax) < 0.1
    )
  }
  
  const activePresetIndex = getActivePreset()
  const isActive = selectedMin !== undefined || selectedMax !== undefined

  return (
    <div className="space-y-4">
      {/* Preset Buttons */}
      <div>
        <div className="grid grid-cols-2 gap-2">
          {presets.map((preset, index) => (
            <button
              key={preset.label}
              onClick={() => handlePresetClick(preset.min, preset.max)}
              className={`
                px-3 py-2 text-xs font-medium rounded-lg transition-all
                ${index === activePresetIndex
                  ? 'bg-blue-100 text-blue-700 border border-blue-200'
                  : 'bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200'
                }
              `}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* Range Slider */}
      <div>
        <div className="mb-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-gray-900">Parameter Range</span>
            <span className={`
              text-xs px-2 py-1 rounded
              ${isActive ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}
            `}>
              {formatNumber(localMin)} - {formatNumber(localMax)}
            </span>
          </div>
        </div>
        
        {/* Custom range slider implementation */}
        <div className="relative h-2 bg-gray-200 rounded-full">
          {/* Track */}
          <div 
            className="absolute h-2 bg-blue-500 rounded-full"
            style={{
              left: `${((localMin - min) / (max - min)) * 100}%`,
              right: `${100 - ((localMax - min) / (max - min)) * 100}%`
            }}
          />
          
          {/* Min thumb */}
          <div
            className="absolute w-4 h-4 bg-white border-2 border-blue-500 rounded-full shadow-sm cursor-pointer hover:scale-110 transition-transform"
            style={{
              left: `${((localMin - min) / (max - min)) * 100}%`,
              transform: 'translateX(-50%)'
            }}
            title={`Min: ${formatNumber(localMin)}`}
          />
          
          {/* Max thumb */}
          <div
            className="absolute w-4 h-4 bg-white border-2 border-blue-500 rounded-full shadow-sm cursor-pointer hover:scale-110 transition-transform"
            style={{
              left: `${((localMax - min) / (max - min)) * 100}%`,
              transform: 'translateX(-50%)'
            }}
            title={`Max: ${formatNumber(localMax)}`}
          />
        </div>
        
        {/* Range labels */}
        <div className="flex justify-between mt-1 text-xs text-gray-500">
          <span>{formatNumber(min)}</span>
          <span>{formatNumber(max)}</span>
        </div>
      </div>
      
      {/* Manual Input */}
      <div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Min (B)
            </label>
            <input
              type="number"
              step="0.1"
              min={min}
              max={localMax}
              value={localMin}
              onChange={(e) => handleMinInputChange(e.target.value)}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500 outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Max (B)
            </label>
            <input
              type="number"
              step="0.1"
              min={localMin}
              max={max}
              value={localMax}
              onChange={(e) => handleMaxInputChange(e.target.value)}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500 outline-none"
            />
          </div>
        </div>
      </div>
      
      {/* Info Text */}
      <div className="text-xs text-gray-500">
        <p>Parameter size affects model performance and memory requirements:</p>
        <ul className="mt-1 space-y-1">
          <li>• &lt;1B: Fast, low memory, good for simple tasks</li>
          <li>• 1-7B: Good balance of speed and capability</li>
          <li>• 7-30B: High quality, requires more memory</li>
          <li>• &gt;30B: Best quality, high memory requirements</li>
        </ul>
      </div>
    </div>
  )
}

export default ParameterFilter
