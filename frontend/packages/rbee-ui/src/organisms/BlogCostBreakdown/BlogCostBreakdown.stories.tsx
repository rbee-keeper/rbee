import type { Meta, StoryObj } from '@storybook/nextjs'
import { BlogCostBreakdown } from './BlogCostBreakdown'

const meta: Meta<typeof BlogCostBreakdown> = {
  title: 'Organisms/BlogCostBreakdown',
  component: BlogCostBreakdown,
  tags: ['autodocs'],
  parameters: {
    layout: 'padded',
  },
}

export default meta
type Story = StoryObj<typeof BlogCostBreakdown>

export const SaaSStartup: Story = {
  args: {
    title: 'SaaS Startup Use Case',
    subtitle: 'Customer support chatbot serving 100K requests/month',
    before: 'Spending $800/month on OpenAI GPT-3.5 for customer support chatbot',
    after: 'Deployed on 1× RTX 3090 ($800 hardware + €129 license)',
    items: [
      { label: 'Hardware', value: '$800', oneTime: true },
      { label: 'rbee License', value: '€129', oneTime: true },
      { label: 'Power', value: '~$30/month' },
    ],
    summary: [
      { label: 'Monthly savings', value: '$770', isSavings: true },
      { label: 'Annual savings', value: '$9,240', isSavings: true },
      { label: 'Break-even', value: '1.2 months', isSavings: true },
    ],
  },
}

export const EnterpriseDeployment: Story = {
  args: {
    title: 'Enterprise Deployment',
    subtitle: 'Multi-department AI infrastructure',
    before: 'Spending $15,000/month on Anthropic Claude API across 5 departments',
    after: 'Deployed on 10× A100 GPUs with rbee orchestration',
    items: [
      { label: 'Hardware (10× A100)', value: '$80,000', oneTime: true },
      { label: 'rbee Enterprise License', value: '€2,499', oneTime: true },
      { label: 'Power & Cooling', value: '~$1,200/month' },
      { label: 'Maintenance', value: '~$500/month' },
    ],
    summary: [
      { label: 'Monthly savings', value: '$13,300', isSavings: true },
      { label: 'Annual savings', value: '$159,600', isSavings: true },
      { label: 'Break-even', value: '6.2 months', isSavings: true },
      { label: '3-year ROI', value: '483%', isSavings: true },
    ],
  },
}

export const HomelabSetup: Story = {
  args: {
    title: 'Homelab Setup',
    before: 'Using ChatGPT Plus ($20/month) + occasional API calls ($50/month)',
    after: 'Repurposed gaming PC with RTX 4090',
    items: [
      { label: 'Hardware (already owned)', value: '$0', oneTime: true },
      { label: 'rbee License', value: 'Free (GPL-3.0)', oneTime: true },
      { label: 'Power', value: '~$15/month' },
    ],
    summary: [
      { label: 'Monthly savings', value: '$55', isSavings: true },
      { label: 'Annual savings', value: '$660', isSavings: true },
    ],
  },
}

export const ResearchLab: Story = {
  args: {
    title: 'University Research Lab',
    subtitle: 'AI research with mixed GPU infrastructure',
    before: 'Spending $5,000/month on cloud GPU instances (AWS p3.8xlarge)',
    after: 'Deployed on 4× RTX 4090 + 2× A6000 + 3× Mac Studio M2',
    items: [
      { label: 'Hardware (4× RTX 4090)', value: '$7,200', oneTime: true },
      { label: 'Hardware (2× A6000)', value: '$9,000', oneTime: true },
      { label: 'Hardware (3× Mac Studio)', value: '$12,000', oneTime: true },
      { label: 'rbee License', value: 'Free (Academic)', oneTime: true },
      { label: 'Power', value: '~$400/month' },
    ],
    summary: [
      { label: 'Monthly savings', value: '$4,600', isSavings: true },
      { label: 'Annual savings', value: '$55,200', isSavings: true },
      { label: 'Break-even', value: '6.1 months', isSavings: true },
    ],
  },
}

export const MinimalExample: Story = {
  args: {
    title: 'Quick ROI Example',
    items: [
      { label: 'Initial investment', value: '$1,000', oneTime: true },
      { label: 'Monthly cost', value: '$50' },
    ],
    summary: [
      { label: 'Monthly savings', value: '$450', isSavings: true },
    ],
  },
}

export const WithoutBeforeAfter: Story = {
  args: {
    title: 'Cost Breakdown',
    subtitle: 'Simple cost analysis',
    items: [
      { label: 'Hardware', value: '$2,000', oneTime: true },
      { label: 'Software', value: '$500', oneTime: true },
      { label: 'Operating costs', value: '$100/month' },
    ],
  },
}
