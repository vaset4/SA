# FRONTEND UI Guide

本文档用于约束 `StudyAssistant` 当前这轮前端重构后的页面职责、组件边界和验证入口。

## 1. 页面结构

### 1.1 项目列表页

- 路由：`app/(workspace)/projects/page.jsx`
- 角色：`项目列表页`
- 只负责：
  - 展示项目卡片
  - 创建项目
  - 进入项目详细页

### 1.2 项目详细页

- 路由：`app/(workspace)/projects/[projectId]/page.jsx`
- 角色：`唯一项目工作面`
- 顶部标签固定为：
  - `总览`
  - `论文`
  - `研究笔记`
  - `活动`

项目详细页核心组件：

- `ProjectHeroPanel`
- `ProjectResearchDesk`
- `ProjectPaperStatusList`
- `ProjectNotesPanel`
- `ProjectActivityFeed`

### 1.3 论文详情页

- 路由：`app/(workspace)/papers/[paperId]/page.jsx`
- 角色：研究材料准备页
- 标签顺序：`总览 / AI 分析 / 原文`
- 主标签固定为：
  - `总览`
  - `AI 分析`
  - `原文`

对应组件：

- `PaperOverviewTab`
- `PaperAnalysisTab`
- `PaperOriginalTab`

### 1.4 研究页

- `app/(workspace)/chat/page.jsx` 是 `内部研究执行页`
- `app/(workspace)/research/page.jsx` 是 `外部研究执行页`
- 不再保留统一研究工作区总页

## 2. 组件分层

### 2.1 页面层

- 页面只负责组织版面和组合组件
- 页面不直接散落业务判断和接口细节

### 2.2 项目工作面层

- `ProjectHeroPanel`
  - 只展示项目总览与研究范围提示
- `ProjectResearchDesk`
  - 只负责研究动作与最近研究
- `ProjectPaperStatusList`
  - 只负责论文范围与分析状态
- `ProjectNotesPanel`
  - 只负责项目笔记沉淀
- `ProjectActivityFeed`
  - 只负责项目动态

### 2.3 论文详情层

- `PaperOverviewTab`
  - 研究问题、核心方法、主要结论
- `PaperAnalysisTab`
  - 中文分析内容和参考文献
- `PaperOriginalTab`
  - 原始 PDF、Markdown 解析文件、原始材料

## 3. 状态与 API 边界

### 3.1 Project Store

- `stores/project-store.js`
- 需要稳定提供：
  - `projects`
  - `papers`
  - `sessions`
  - `activities`
  - `latestSession`
  - `summary.totalPapers`
  - `summary.analyzedPapers`
  - `summary.pendingPapers`

### 3.2 Library Store

- `stores/library-store.js`
- 继续负责论文列表与单篇详情
- 需要支持详情页更新后回写当前 store

### 3.3 API Wrapper

- 项目相关请求走 `lib/api/projects.js`
- 论文相关请求走 `lib/api/papers.js`
- 研究相关请求走 `lib/api/research.js`
- 页面和组件不直接写 `fetch`

## 4. 视觉与交互约束

- 左侧只保留一套全局导航
- `研究` 在导航里只作为分组，下面只保留 `内部研究` 和 `外部研究`
- 项目详细页使用顶部标签，不再增加第二个左侧栏
- 以暖白、暖灰和轻边框建立层级
- 不用表情符号充当图标
- 主要状态必须有文字，不只靠颜色
- 空状态必须明确告诉用户下一步该做什么
- 所有主要按钮和卡片要有清楚的 hover / focus 状态
- 对话页左侧会话列表和右侧消息区都要有稳定滚动容器

## 5. 回归验证入口

当前最小回归仍依赖 Node 原生测试：

- `lib/study-assistant-workspace.test.mjs`
- `lib/frontend-phase6-structure.test.mjs`

建议执行：

```bash
npm test
npm run build
```

## 6. 当前阶段边界

这一轮先把页面角色和布局骨架收稳，不在这里展开：

- 手动触发 `AI 分析` 的完整链路
- 活动流后端持久化
- 图表、公式、原文逐段阅读器
- 图谱能力扩张
