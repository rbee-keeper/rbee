// TEAM-411: Compatibility badge for Keeper
// Shows compatibility status with tooltip

import { useQuery } from '@tanstack/react-query'
import { checkModelCompatibility, type CompatibilityResult } from '@/api/compatibility'
import { Badge } from '@rbee/ui/atoms'

interface CompatibilityBadgeProps {
  modelId: string
  workerType: string
  className?: string
}

export function CompatibilityBadge({ modelId, workerType, className = '' }: CompatibilityBadgeProps) {
  const { data: compat, isLoading } = useQuery({
    queryKey: ['compatibility', modelId, workerType],
    queryFn: () => checkModelCompatibility(modelId, workerType),
    staleTime: 60 * 60 * 1000, // Cache for 1 hour
  })

  if (isLoading) {
    return (
      <Badge variant="secondary" className={className}>
        Checking...
      </Badge>
    )
  }

  if (!compat) {
    return null
  }

  const variant = compat.compatible ? 'default' : 'destructive'
  const label = compat.compatible ? '✅ Compatible' : '❌ Incompatible'

  return (
    <div className={`group relative inline-block ${className}`}>
      <Badge variant={variant}>
        {label}
      </Badge>
      
      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-50 w-64">
        <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 shadow-lg">
          <p className="font-semibold mb-1">{workerType.toUpperCase()} Worker</p>
          <p className="text-gray-300 text-xs mb-1">Confidence: {compat.confidence}</p>
          
          {compat.reasons.length > 0 && (
            <div className="mt-2">
              <p className="font-medium text-xs">Reasons:</p>
              <ul className="list-disc list-inside text-xs space-y-0.5">
                {compat.reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
          
          {compat.warnings.length > 0 && (
            <div className="mt-2">
              <p className="font-medium text-xs text-yellow-400">Warnings:</p>
              <ul className="list-disc list-inside text-xs space-y-0.5">
                {compat.warnings.map((warning, i) => (
                  <li key={i}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          
          {compat.recommendations.length > 0 && (
            <div className="mt-2">
              <p className="font-medium text-xs text-blue-400">Recommendations:</p>
              <ul className="list-disc list-inside text-xs space-y-0.5">
                {compat.recommendations.map((rec, i) => (
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
