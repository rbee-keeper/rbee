// TEAM-410: Compatibility badge component
// Shows compatibility status with tooltip details

import type { CompatibilityResult } from '../types/compatibility'

interface CompatibilityBadgeProps {
  result: CompatibilityResult
  workerName: string
  className?: string
}

export function CompatibilityBadge({ result, workerName, className = '' }: CompatibilityBadgeProps) {
  const badgeClass = result.compatible
    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  
  const label = result.compatible ? 'Compatible' : 'Incompatible'
  
  return (
    <div className={`group relative inline-block ${className}`}>
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${badgeClass}`}>
        {label}
      </span>
      
      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-50">
        <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 max-w-sm shadow-lg">
          <p className="font-semibold mb-1">{workerName}</p>
          <p className="text-gray-300 text-xs mb-1">Confidence: {result.confidence}</p>
          
          {result.reasons.length > 0 && (
            <div className="mt-2">
              <p className="font-medium text-xs">Reasons:</p>
              <ul className="list-disc list-inside text-xs space-y-0.5">
                {result.reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
          
          {result.warnings.length > 0 && (
            <div className="mt-2">
              <p className="font-medium text-xs text-yellow-400">Warnings:</p>
              <ul className="list-disc list-inside text-xs space-y-0.5">
                {result.warnings.map((warning, i) => (
                  <li key={i}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          
          {result.recommendations.length > 0 && (
            <div className="mt-2">
              <p className="font-medium text-xs text-blue-400">Recommendations:</p>
              <ul className="list-disc list-inside text-xs space-y-0.5">
                {result.recommendations.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Arrow */}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
            <div className="border-4 border-transparent border-t-gray-900"></div>
          </div>
        </div>
      </div>
    </div>
  )
}
