'use client'

import { Toaster as Sonner, type ToasterProps } from 'sonner'
import { useTheme } from '../../providers/ThemeProvider/ThemeProvider'

// TEAM-472: Handle theme properly with exactOptionalPropertyTypes
const Toaster = ({ theme: _theme, className, ...restProps }: ToasterProps) => {
  const { theme = 'system', resolvedTheme } = useTheme()
  const effectiveTheme = (theme === 'system' ? resolvedTheme : theme) ?? 'system'

  return (
    <Sonner
      theme={effectiveTheme as ToasterProps['theme']}
      className={className || 'toaster group'}
      style={
        {
          '--normal-bg': 'var(--popover)',
          '--normal-text': 'var(--popover-foreground)',
          '--normal-border': 'var(--border)',
        } as React.CSSProperties
      }
      {...(restProps as any)}
    />
  )
}

export { Toaster }
