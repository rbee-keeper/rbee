import type { Meta, StoryObj } from '@storybook/nextjs'
import { BlogComparisonTable } from './BlogComparisonTable'

const meta: Meta<typeof BlogComparisonTable> = {
  title: 'Organisms/BlogComparisonTable',
  component: BlogComparisonTable,
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof BlogComparisonTable>

export const DeploymentComparison: Story = {
  args: {
    title: 'Deployment Method Comparison',
    columns: ['rbee (SSH)', 'Kubernetes', 'Docker Swarm', 'Manual Scripts'],
    highlightColumn: 0,
    rows: [
      { feature: 'Setup Time', values: ['5 minutes', '2-6 months', '1-2 weeks', 'Varies'] },
      { feature: 'Complexity', values: ['Low', 'Very High', 'Medium', 'High'] },
      { feature: 'Prerequisites', values: ['SSH access', 'K8s cluster, DevOps expertise', 'Docker registry, networking', 'Custom tooling, maintenance'] },
      { feature: 'Multi-GPU Support', values: [true, true, true, false] },
      { feature: 'Heterogeneous Hardware', values: [true, false, false, false] },
    ],
  },
}

export const FeatureComparison: Story = {
  args: {
    title: 'rbee vs Cloud APIs',
    columns: ['rbee', 'OpenAI', 'Anthropic', 'Cohere'],
    highlightColumn: 0,
    rows: [
      { feature: 'Data Privacy', values: ['100% local', 'Cloud-based', 'Cloud-based', 'Cloud-based'] },
      { feature: 'Monthly Cost', values: ['$0 + power', '$100-3,000', '$100-2,500', '$100-2,000'] },
      { feature: 'GDPR Compliant', values: [true, false, false, false] },
      { feature: 'Custom Models', values: [true, false, false, false] },
      { feature: 'Offline Support', values: [true, false, false, false] },
    ],
  },
}
