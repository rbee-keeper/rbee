// TEAM-476: Sort dropdown component - RIGHT SIDE of FilterBar
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@rbee/ui/atoms/Select'
import { ArrowDownAZ, ArrowUpAZ } from 'lucide-react'

export interface SortOption {
  value: string
  label: string
}

export interface SortDropdownProps {
  value: string
  onChange: (value: string) => void
  options: SortOption[]
  placeholder?: string
  className?: string
}

export function SortDropdown({ value, onChange, options, placeholder = 'Sort by', className }: SortDropdownProps) {
  return (
    <div className={`flex items-center gap-2 ${className || ''}`}>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-[200px]">
          <div className="flex items-center gap-2">
            {value ? <ArrowDownAZ className="size-4" /> : <ArrowUpAZ className="size-4" />}
            <SelectValue placeholder={placeholder} />
          </div>
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
