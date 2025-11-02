import type { Meta, StoryObj } from '@storybook/nextjs'
import FeaturesPage from './FeaturesPage'

const meta: Meta<typeof FeaturesPage> = {
  title: 'Pages/FeaturesPage',
  component: FeaturesPage,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof FeaturesPage>

export const Default: Story = {}
