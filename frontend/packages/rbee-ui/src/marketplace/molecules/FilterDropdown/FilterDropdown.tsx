// TEAM-476: Single-select dropdown filter component
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@rbee/ui/atoms/Select'
import { Label } from '@rbee/ui/atoms/Label'

export interface FilterOption {
  value: string
  label: string
}

export interface FilterDropdownProps {
  label: string
  value: string | undefined
  onChange: (value: string | undefined) => void
  options: FilterOption[]
  placeholder?: string
  allowClear?: boolean
}

export function FilterDropdown({
  label,
  value,
  onChange,
  options,
  placeholder = 'Select...',
  allowClear = true,
}: FilterDropdownProps) {
  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">{label}</Label>
      <Select
        value={value || ''}
        onValueChange={(val) => {
          if (val === '' && allowClear) {
            onChange(undefined)
          } else {
            onChange(val)
          }
        }}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {allowClear && value && (
            <SelectItem value="">
              <span className="text-muted-foreground">Clear selection</span>
            </SelectItem>
          )}
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
