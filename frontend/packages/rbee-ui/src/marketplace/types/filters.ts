// Generic filter system types - reusable across workers, models, etc.

export interface FilterOption<T = string> {
  /** Display label for the filter option */
  label: string
  /** Value used in URLs and filtering logic */
  value: T
}

export interface FilterGroup<T = string> {
  /** Unique identifier for this filter group (e.g., "category", "backend") */
  id: string
  /** Display label for the filter group */
  label: string
  /** Available options for this filter */
  options: FilterOption<T>[]
}

export interface FilterConfig<T = Record<string, string>> {
  /** Current filter values */
  filters: T
  /** URL path for this filter combination (e.g., "filter/llm/cuda") */
  path: string
}
