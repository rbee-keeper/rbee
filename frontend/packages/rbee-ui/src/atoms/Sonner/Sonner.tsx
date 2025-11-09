'use client'

import { Toaster as Sonner, type ToasterProps } from 'sonner'
import { useTheme } from '../../providers/ThemeProvider/ThemeProvider'

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = 'system', resolvedTheme } = useTheme()
  const effectiveTheme = theme === 'system' ? resolvedTheme : theme

  return (
    <Sonner
      theme={effectiveTheme as ToasterProps['theme']}
      className="toaster group"
      style={
        {
          '--normal-bg': 'var(--popover)',
          '--normal-text': 'var(--popover-foreground)',
          '--normal-border': 'var(--border)',
        } as React.CSSProperties
      }
      {...props}
    />
  )
}

export { Toaster }
