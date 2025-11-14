import type { Meta, StoryObj } from '@storybook/nextjs'
import { Checkbox } from '@rbee/ui/atoms/Checkbox'
import { Label } from '@rbee/ui/atoms/Label'
import { SidebarFilterItem } from './SidebarFilterItem'

const meta: Meta<typeof SidebarFilterItem> = {
  title: 'Marketplace/Molecules/SidebarFilterItem',
  component: SidebarFilterItem,
  parameters: {
    layout: 'centered',
  },
}

export default meta

type Story = StoryObj<typeof SidebarFilterItem>

export const Default: Story = {
  args: {
    children: (
      <div className="text-sm text-muted-foreground">
        Default sidebar filter item
      </div>
    ),
  },
}

export const Selected: Story = {
  args: {
    selected: true,
    children: (
      <div className="text-sm text-sidebar-foreground">
        Selected sidebar filter item
      </div>
    ),
  },
}

export const WithCheckbox: Story = {
  render: () => (
    <SidebarFilterItem selected size="md">
      <Checkbox id="sidebar-filter-item-example" checked />
      <Label
        htmlFor="sidebar-filter-item-example"
        className="flex-1 min-w-0 cursor-pointer"
      >
        <div className="flex flex-col gap-0.5">
          <span className="text-sm font-medium text-sidebar-foreground">
            Text Generation
          </span>
          <span className="text-xs text-muted-foreground">
            Generate text, chat, complete prompts
          </span>
        </div>
      </Label>
    </SidebarFilterItem>
  ),
}

export const Sizes: Story = {
  render: () => (
    <div className="space-y-3 max-w-md">
      <SidebarFilterItem size="sm">
        <div className="text-sm text-muted-foreground">
          Small padding (sm)
        </div>
      </SidebarFilterItem>
      <SidebarFilterItem size="md" selected>
        <div className="text-sm text-sidebar-foreground">
          Medium padding (md, selected)
        </div>
      </SidebarFilterItem>
    </div>
  ),
}
