import type { Meta, StoryObj } from '@storybook/nextjs'
import { FeaturesHero } from './FeaturesHero'

const meta: Meta<typeof FeaturesHero> = {
  title: 'Templates/FeaturesHero',
  component: FeaturesHero,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof FeaturesHero>

export const OnFeaturesPage: Story = {}
