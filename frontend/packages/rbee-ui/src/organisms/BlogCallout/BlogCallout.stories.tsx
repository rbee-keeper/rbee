import type { Meta, StoryObj } from '@storybook/nextjs'
import { BlogCallout } from './BlogCallout'

const meta: Meta<typeof BlogCallout> = {
  title: 'Organisms/BlogCallout',
  component: BlogCallout,
  tags: ['autodocs'],
  parameters: {
    layout: 'padded',
  },
}

export default meta
type Story = StoryObj<typeof BlogCallout>

export const Info: Story = {
  args: {
    variant: 'info',
    title: 'Important Information',
    children: (
      <p>This is an informational callout. Use it to highlight important details that readers should know about.</p>
    ),
  },
}

export const Success: Story = {
  args: {
    variant: 'success',
    title: 'Success Story',
    children: (
      <p>This company reduced their AI infrastructure costs by 90% after switching to rbee. Monthly savings: $8,500.</p>
    ),
  },
}

export const Warning: Story = {
  args: {
    variant: 'warning',
    title: 'Important Warning',
    children: (
      <p>
        Make sure to backup your configuration files before upgrading. This version includes breaking changes to the
        routing system.
      </p>
    ),
  },
}

export const Danger: Story = {
  args: {
    variant: 'danger',
    title: 'The Pain Points',
    children: (
      <ul className="space-y-2 mt-2">
        <li>
          <strong>Ollama:</strong> Great for single machines, but can't orchestrate multiple GPUs across your network
        </li>
        <li>
          <strong>Kubernetes + Ray/KServe:</strong> 6 months of setup, requires a dedicated DevOps team
        </li>
        <li>
          <strong>Cloud APIs:</strong> $100-3,000/month when you already own the hardware
        </li>
      </ul>
    ),
  },
}

export const Tip: Story = {
  args: {
    variant: 'tip',
    title: 'Pro Tip',
    children: (
      <p>
        Use worker metadata to tag workers with custom properties like <code>region</code>, <code>gpu_type</code>, or{' '}
        <code>cost_tier</code>. This enables advanced routing logic in your Rhai scripts.
      </p>
    ),
  },
}

export const Example: Story = {
  args: {
    variant: 'example',
    title: 'Real-World Example',
    children: (
      <div className="space-y-2">
        <p>
          <strong>Scenario:</strong> A startup with 3 machines (RTX 4090, Mac Studio M2, old server with Tesla K80)
        </p>
        <p>
          <strong>Setup time:</strong> 5 minutes
        </p>
        <p>
          <strong>Result:</strong> Unified API serving 10K requests/day, $0 cloud costs
        </p>
      </div>
    ),
  },
}

export const Pricing: Story = {
  args: {
    variant: 'pricing',
    title: 'Transparent Pricing Model',
    children: (
      <div className="space-y-3 mt-2">
        <div>
          <p className="font-semibold">Free Forever (GPL-3.0)</p>
          <p className="text-sm">Core orchestration, multi-machine support, OpenAI API, basic routing</p>
        </div>
        <div>
          <p className="font-semibold">Premium Modules (€129-499 one-time)</p>
          <p className="text-sm">Advanced scheduling, Rhai scripting, telemetry, GDPR auditing, priority support</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span>✓ No subscriptions</span>
          <span>✓ No per-token fees</span>
          <span>✓ No vendor lock-in</span>
        </div>
      </div>
    ),
  },
}

export const WithComplexContent: Story = {
  args: {
    variant: 'info',
    title: 'Advanced Configuration',
    children: (
      <div className="space-y-3">
        <p>You can configure rbee using multiple methods:</p>
        <ol className="list-decimal list-inside space-y-1">
          <li>Environment variables</li>
          <li>Configuration files (~/.config/rbee/)</li>
          <li>CLI flags</li>
          <li>API endpoints</li>
        </ol>
        <p className="text-sm text-muted-foreground">
          Priority order: CLI flags → Environment variables → Config files → Defaults
        </p>
      </div>
    ),
  },
}
