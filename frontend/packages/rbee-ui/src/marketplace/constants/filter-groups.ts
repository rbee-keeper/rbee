// TEAM-467 RULE ZERO: UI-ready filter groups using API enum values
// Shared between Next.js app and Tauri Keeper app
// NO MAGIC STRINGS - uses constants from filter-constants.ts

import type { FilterGroup } from '../types/filters'
import {
  CIVITAI_BASE_MODELS,
  CIVITAI_MODEL_TYPES,
  CIVITAI_SORTS,
  CIVITAI_TIME_PERIODS,
  HF_LICENSES,
  HF_SIZES,
  HF_SORTS,
} from './filter-constants'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HUGGINGFACE FILTER GROUPS (using API enum values)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export const HUGGINGFACE_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'size',
    label: 'Model Size',
    options: [
      { label: 'All Sizes', value: 'All' },
      { label: 'Small (<7B)', value: 'Small' },
      { label: 'Medium (7B-13B)', value: 'Medium' },
      { label: 'Large (>13B)', value: 'Large' },
    ],
  },
  {
    id: 'license',
    label: 'License',
    options: [
      { label: 'All Licenses', value: 'All' },
      { label: 'Apache 2.0', value: 'Apache' },
      { label: 'MIT', value: 'MIT' },
      { label: 'Other', value: 'Other' },
    ],
  },
]

export const HUGGINGFACE_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'Downloads' },
    { label: 'Most Likes', value: 'Likes' },
  ],
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CIVITAI FILTER GROUPS (using API enum values)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export const CIVITAI_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'timePeriod',
    label: 'Time Period',
    options: [
      { label: 'All Time', value: 'AllTime' },
      { label: 'Month', value: 'Month' },
      { label: 'Week', value: 'Week' },
      { label: 'Day', value: 'Day' },
    ],
  },
  {
    id: 'modelType',
    label: 'Model Type',
    options: [
      { label: 'All Types', value: 'All' },
      { label: 'Checkpoint', value: 'Checkpoint' },
      { label: 'LORA', value: 'LORA' },
    ],
  },
  {
    id: 'baseModel',
    label: 'Base Model',
    options: [
      { label: 'All Models', value: 'All' },
      { label: 'SDXL 1.0', value: 'SDXL 1.0' },
      { label: 'SD 1.5', value: 'SD 1.5' },
      { label: 'SD 2.1', value: 'SD 2.1' },
    ],
  },
]

export const CIVITAI_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'Most Downloaded' },
    { label: 'Highest Rated', value: 'Highest Rated' },
    { label: 'Newest', value: 'Newest' },
  ],
}
