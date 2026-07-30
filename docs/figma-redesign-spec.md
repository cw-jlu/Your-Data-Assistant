# Data Agent Workspace - Figma Redesign Spec

## Design read

这是一个高频使用的本地数据 Agent 客户端，不是营销页面。界面采用浅色桌面工具语言，优先保证文件、Query、结果和 Trace 的连续操作关系。

- Design variance: 5
- Motion intensity: 2
- Visual density: 7
- Primary accent: cobalt blue
- Product font: Segoe UI Variable / Noto Sans fallback
- Code stack: vanilla HTML, CSS and JavaScript

## Desktop frame

- Frame: 1440 x 960
- Canvas: `#F4F5F7`
- Top app bar: 64 px
- View heading: 86 px
- Workspace minimum height: viewport minus 182 px
- Workspace radius: 14 px
- Workspace columns:
  - Context rail: 236 px
  - Analysis canvas: fill
  - Inspector preview: 296 px

At widths below 1180 px, hide the Inspector preview and keep Context + Analysis. Below 840 px, collapse to one column and use a fixed bottom navigation.

## Semantic tokens

| Figma variable | CSS variable | Value |
| --- | --- | --- |
| Color/Canvas | `--canvas` | `#F4F5F7` |
| Color/Surface | `--surface` | `#FCFCFD` |
| Color/Surface Subtle | `--surface-subtle` | `#F0F2F5` |
| Color/Surface Hover | `--surface-hover` | `#E9ECF1` |
| Color/Border | `--border` | `#D9DDE5` |
| Color/Border Strong | `--border-strong` | `#C5CAD4` |
| Color/Text | `--text` | `#17191D` |
| Color/Muted | `--muted` | `#6D7380` |
| Color/Quiet | `--quiet` | `#9298A4` |
| Color/Accent | `--accent` | `#315CE8` |
| Color/Accent Hover | `--accent-hover` | `#264BC8` |
| Color/Accent Soft | `--accent-soft` | `#E8EDFF` |
| Color/Success | `--success` | `#168862` |
| Color/Warning | `--warning` | `#9B6417` |
| Color/Danger | `--danger` | `#C43D4B` |

Radii:

- Small: 7 px
- Medium: 10 px
- Large: 14 px

## Screen anatomy

### App bar

- Left: product mark, Data Agent, Local workspace
- Center: 工作区, 运行, 融合路径
- Right: local service status and refresh
- Navigation height: 34 px
- App bar uses a single bottom border, no outer shadow

### Context rail

Files:

- Section padding: 18 x 16 px
- Upload target: 112 px minimum height
- Uploaded file rows: 28 px icon, file name and size, remove action
- “查看支持格式” opens the same upload popover as the composer plus button

Agents:

- Four selectable engine rows
- Selected state uses Accent Soft background, Accent border and a semantic check
- Unready state is disabled at the native control level
- “全选” remains a secondary action

### Analysis canvas

Toolbar:

- 54 px high
- Left: Workspace / 新分析, 多引擎分析
- Right: current model and settings

Empty state:

- Maximum width: 680 px
- Heading: 34 px, weight 640, two lines maximum
- Supporting copy: 13 px, line-height 1.75
- Three workflow cells: 添加文件, 输入 Query, 检查结果与 Trace

Composer:

- Maximum width: 760 px
- Radius: 14 px
- Textarea minimum height: 72 px
- Left actions: upload, settings, selected engine summary
- Right action: cobalt Run button

### Inspector preview

- Describes the three trace sources:
  - Process log
  - JSON artifacts
  - SQLite tracing
- A single “用户可见” semantic label explains the product promise
- This column disappears below 1180 px; complete Trace remains available through the Run Inspector drawer

## Required states

### Empty

- No files
- Mamba selected by default
- Upload target visible
- Query composer visible without scrolling on desktop

### Upload popover

- Shows selected engine names
- Shows the common supported extension count
- Groups formats by data, document and media
- Expands into an engine-by-engine matrix
- Contains the actual file picker action

### Conversation and result

- User Query is right aligned
- Each engine response is its own bordered result block
- Result block footer exposes both “查看结果” and “完整 Trace”
- Running state shows progress feedback and bounded runtime logs

### Run Inspector

- Width: min(680 px, 94 viewport width)
- Tabs: 输出结果 and 完整 Trace
- Trace pane exposes runtime log, JSON artifacts and SQLite-derived artifacts
- Source paths remain visible
- Stop action is shown only while running

## Figma library discovery

The connected file exposes Figma Simple Design System. Relevant discovered assets:

- AI Chat Box
- AI Chat Sidebar
- AI Chat User Message
- AI Chat Response
- Button
- Textarea Field
- Tabs
- Body Base, Body Small and Heading text styles

The library assets can supply behavioral structure. The local Data Agent semantic tokens above remain authoritative for color, radius and spacing so the Figma screen stays aligned with `static/styles.css`.

## Source of truth

- Markup: `static/index.html`
- Visual tokens and layout: `static/styles.css`
- Interaction states: `static/app.js`
- Desktop reference: `.runtime/ui-redesign.png`
- Mobile reference: `.runtime/ui-redesign-mobile.png`
