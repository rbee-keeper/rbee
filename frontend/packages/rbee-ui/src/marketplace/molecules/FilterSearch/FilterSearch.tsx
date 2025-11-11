// TEAM-476: Search filter component with debounce
import { Input } from '@rbee/ui/atoms/Input'
import { Label } from '@rbee/ui/atoms/Label'
import { Search, X } from 'lucide-react'
import * as React from 'react'

export interface FilterSearchProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  debounceMs?: number
}

export function FilterSearch({ label, value, onChange, placeholder = 'Search...', debounceMs = 300 }: FilterSearchProps) {
  const [localValue, setLocalValue] = React.useState(value)
  const debounceTimerRef = React.useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Debounce search input
  React.useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    debounceTimerRef.current = setTimeout(() => {
      onChange(localValue)
    }, debounceMs)

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [localValue, onChange, debounceMs])

  // Sync external changes
  React.useEffect(() => {
    setLocalValue(value)
  }, [value])

  const handleClear = () => {
    setLocalValue('')
  }

  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">{label}</Label>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground pointer-events-none" />
        <Input
          type="search"
          placeholder={placeholder}
          value={localValue}
          onChange={(e) => setLocalValue(e.target.value)}
          className="pl-9 pr-9"
        />
        {localValue && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        )}
      </div>
    </div>
  )
}
