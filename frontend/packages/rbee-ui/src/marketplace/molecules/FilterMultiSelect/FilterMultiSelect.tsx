// TEAM-476: Multi-select filter component with checkboxes
import { Button } from '@rbee/ui/atoms/Button'
import { Checkbox } from '@rbee/ui/atoms/Checkbox'
import { Label } from '@rbee/ui/atoms/Label'
import { Popover, PopoverContent, PopoverTrigger } from '@rbee/ui/atoms/Popover'
import { Badge } from '@rbee/ui/atoms/Badge'
import { ChevronDown, X } from 'lucide-react'
import * as React from 'react'

export interface MultiSelectOption {
  value: string
  label: string
}

export interface FilterMultiSelectProps {
  label: string
  values: string[]
  onChange: (values: string[]) => void
  options: MultiSelectOption[]
  placeholder?: string
  maxDisplay?: number
}

export function FilterMultiSelect({
  label,
  values,
  onChange,
  options,
  placeholder = 'Select...',
  maxDisplay = 3,
}: FilterMultiSelectProps) {
  const [open, setOpen] = React.useState(false)

  const handleToggle = (value: string) => {
    const newValues = values.includes(value) ? values.filter((v) => v !== value) : [...values, value]
    onChange(newValues)
  }

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation()
    onChange([])
  }

  const selectedLabels = values.map((v) => options.find((o) => o.value === v)?.label || v)

  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">{label}</Label>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" className="w-full justify-between font-normal">
            {values.length === 0 ? (
              <span className="text-muted-foreground">{placeholder}</span>
            ) : (
              <div className="flex items-center gap-1 flex-wrap">
                {selectedLabels.slice(0, maxDisplay).map((label, i) => (
                  <Badge key={i} variant="secondary" className="text-xs">
                    {label}
                  </Badge>
                ))}
                {selectedLabels.length > maxDisplay && (
                  <Badge variant="secondary" className="text-xs">
                    +{selectedLabels.length - maxDisplay}
                  </Badge>
                )}
              </div>
            )}
            <div className="flex items-center gap-1">
              {values.length > 0 && (
                <X className="size-4 hover:text-destructive" onClick={handleClear} />
              )}
              <ChevronDown className="size-4" />
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[300px] p-3" align="start">
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {options.map((option) => (
              <div key={option.value} className="flex items-center space-x-2">
                <Checkbox
                  id={`${label}-${option.value}`}
                  checked={values.includes(option.value)}
                  onCheckedChange={() => handleToggle(option.value)}
                />
                <label
                  htmlFor={`${label}-${option.value}`}
                  className="text-sm font-normal leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer flex-1"
                >
                  {option.label}
                </label>
              </div>
            ))}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
