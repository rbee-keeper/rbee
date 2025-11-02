import type { Meta, StoryObj } from '@storybook/nextjs'
import ProvidersPage from './ProvidersPage'

const meta = {
  title: 'Pages/ProvidersPage',
  component: ProvidersPage,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersPage>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}
