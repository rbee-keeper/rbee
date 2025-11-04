// TEAM-401: ALL page data (perfect for SSG)
import type { WorkerListTemplateProps } from '../../templates/WorkerListTemplate'

export interface WorkersPageProps {
  seo?: {
    title: string
    description: string
  }
  template: WorkerListTemplateProps
}

// Example props for SSG
export const defaultWorkersPageProps: WorkersPageProps = {
  seo: {
    title: 'Browse Workers | rbee Marketplace',
    description:
      'Download optimized inference workers for CPU, CUDA, and Metal. Run AI models efficiently on your hardware.',
  },
  template: {
    title: 'Inference Workers',
    description: 'Download workers optimized for your hardware',
    workers: [],
    filters: { search: '', sort: 'name' },
  },
}
