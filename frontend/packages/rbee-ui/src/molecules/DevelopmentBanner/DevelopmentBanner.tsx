// TEAM-477: Reusable development/MVP banner component
import { AlertTriangle, Hammer, Rocket } from 'lucide-react'
import * as React from 'react'

export interface DevelopmentBannerProps {
  /** Banner variant */
  variant?: 'development' | 'mvp' | 'beta'
  
  /** Custom message (overrides default) */
  message?: string
  
  /** Optional additional details */
  details?: string
  
  /** Icon to display (default: variant-specific) */
  icon?: 'warning' | 'hammer' | 'rocket' | React.ReactNode
  
  /** Background color override */
  className?: string
}

const variantConfig = {
  development: {
    icon: 'warning' as const,
    bgClass: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
    textClass: 'text-yellow-800 dark:text-yellow-200',
    defaultMessage: '🚧 This website is currently under active development. Features and content may change.',
  },
  mvp: {
    icon: 'hammer' as const,
    bgClass: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    textClass: 'text-blue-800 dark:text-blue-200',
    defaultMessage: '🔨 MVP Release: Core features are functional. More capabilities coming soon.',
  },
  beta: {
    icon: 'rocket' as const,
    bgClass: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
    textClass: 'text-purple-800 dark:text-purple-200',
    defaultMessage: '🚀 Beta Release: Testing in progress. Report issues on GitHub.',
  },
}

const iconComponents = {
  warning: AlertTriangle,
  hammer: Hammer,
  rocket: Rocket,
}

export function DevelopmentBanner({
  variant = 'development',
  message,
  details,
  icon,
  className,
}: DevelopmentBannerProps) {
  const config = variantConfig[variant]
  
  // Determine icon to render
  const IconComponent = React.useMemo(() => {
    if (React.isValidElement(icon)) return null
    if (typeof icon === 'string' && icon in iconComponents) {
      return iconComponents[icon as keyof typeof iconComponents]
    }
    return iconComponents[config.icon]
  }, [icon, config.icon])

  return (
    <div className={className || `${config.bgClass} border-b -mt-16 pt-16`}>
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-start justify-center gap-3 text-sm">
          {/* Icon */}
          <div className={`flex-shrink-0 ${config.textClass}`}>
            {React.isValidElement(icon) ? (
              icon
            ) : IconComponent ? (
              <IconComponent className="h-5 w-5" aria-hidden="true" />
            ) : null}
          </div>

          {/* Content */}
          <div className={`flex-1 ${config.textClass}`}>
            <p className="font-medium">{message || config.defaultMessage}</p>
            {details && <p className="mt-1 text-xs opacity-90">{details}</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
