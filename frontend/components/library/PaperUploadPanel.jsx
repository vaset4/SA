'use client'

import { Button } from '@heroui/react'
import { StatusBadge } from '../shared/StatusBadge'
import { UploadTaskStatus } from './UploadTaskStatus'

const STAGE_LABELS = {
  idle: '待上传',
  uploading: '上传文件',
  queued: '任务排队中',
  parsing: '解析文档',
  vectorizing: '向量化与索引',
  graphing: '图谱增强',
  done: '上传完成',
  paused: '已暂停，进度已保存',
  failed: '处理失败',
  cancelled: '已取消',
  completed: '处理完成',
}

export function PaperUploadPanel({ tools, library, projectId = '' }) {
  const hasFile = Boolean(tools.uploadFile)
  const stageLabel = STAGE_LABELS[tools.uploadStage] || STAGE_LABELS.idle
  const isBusy = tools.uploadBusy || tools.deleteBusy
  const progressText = tools.deleteBusy
    ? '删除处理中'
    : (tools.uploadBusy || tools.uploadStage === 'done')
      ? `${stageLabel} · ${tools.uploadProgress}% · ${tools.formattedElapsed}`
      : stageLabel
  const deleteStatus = tools.deleteResult?.status || ''
  const deleteTone = deleteStatus === 'failed' ? 'danger' : deleteStatus === 'completed' ? 'success' : 'warning'
  const deleteStatusLabel = {
    queued: '排队中',
    running: '清理中',
    completed: '已完成',
    failed: '失败',
  }[deleteStatus] || '未开始'

  const onFileChange = (event) => {
    const file = event.target.files?.[0] || null
    tools.setUploadFile(file)
  }

  const onUpload = () => {
    tools.upload(library.refresh, projectId)
  }

  return (
    <section
      className="sa-section-card sa-section-card--compact sa-upload-panel"
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 15 }}>上传论文</h3>
          <p style={{ margin: '6px 0 0', color: 'var(--sa-text-soft)', fontSize: 12 }}>
            支持 PDF / DOCX / PPTX / Markdown，上传后自动入库并切分。
            图谱可稍后单独启用；论文信息补全不会阻止基本使用。
          </p>
        </div>
        <StatusBadge tone={isBusy ? 'warning' : 'neutral'}>
          {progressText}
        </StatusBadge>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <input type="file" onChange={onFileChange} disabled={isBusy} />
        <Button color="primary" size="sm" onPress={onUpload} isDisabled={!hasFile || isBusy}>
          {tools.uploadBusy ? '上传中' : tools.deleteBusy ? '删除处理中' : '开始上传'}
        </Button>
        {hasFile ? (
          <span style={{ color: 'var(--sa-text-soft)', fontSize: 12 }}>
            {tools.uploadFile.name}
          </span>
        ) : null}
      </div>
      <div style={{ display: 'grid', gap: 8 }}>
        <div
          style={{
            height: 8,
            borderRadius: 999,
            background: 'color-mix(in srgb, var(--sa-border) 72%, white)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${tools.uploadProgress || 0}%`,
              height: '100%',
              borderRadius: 999,
              background: 'linear-gradient(90deg, var(--sa-accent), #0f766e)',
              transition: 'width 160ms ease',
            }}
          />
        </div>
      </div>
      {tools.canCheckUpload && !tools.uploadBusy ? (
        <button type="button" className="sa-button sa-button--ghost" onClick={tools.checkUpload}>查回上传状态</button>
      ) : null}
      <UploadTaskStatus task={tools.uploadTask} busy={isBusy}
        onResume={() => tools.resumeUpload(library.refresh)} onRebuild={() => tools.rebuildUpload(library.refresh)} />
      {tools.deleteResult ? (
        <div
          style={{
            marginTop: 14,
            paddingTop: 12,
            borderTop: '1px solid var(--sa-border)',
            display: 'grid',
            gap: 8,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <div>
              <h4 style={{ margin: 0, fontSize: 14 }}>论文删除任务</h4>
              <p style={{ margin: '6px 0 0', color: 'var(--sa-text-soft)', fontSize: 12 }}>
                论文 {tools.deleteResult.payload?.paper_id || '-'} · {tools.deleteStatusMessage || '处理中'}
              </p>
            </div>
            <StatusBadge tone={deleteTone}>{deleteStatusLabel}</StatusBadge>
          </div>
          {tools.deleteResult.error ? (
            <p style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 12 }}>{tools.deleteResult.error}</p>
          ) : null}
        </div>
      ) : null}
      {tools.libraryError ? (
        <p style={{ margin: '10px 0 0', color: 'var(--sa-danger)', fontSize: 12 }}>{tools.libraryError}</p>
      ) : null}
    </section>
  )
}
