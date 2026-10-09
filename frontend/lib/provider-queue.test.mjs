/** 模型额度等待文案回归，作者：xxx */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createContext, SourceTextModule } from 'node:vm'

const context = createContext({})
const trace = new SourceTextModule(fs.readFileSync(new URL('./research-trace.js', import.meta.url), 'utf8'), { context })
const modes = new SourceTextModule(fs.readFileSync(new URL('./external-research-mode.js', import.meta.url), 'utf8'), { context })
await trace.link(() => modes)
await trace.evaluate()

test('模型等待显示估计时间与可取消，不把排队标为已发送或已完成', () => {
  assert.equal(trace.namespace.formatResearchTraceTitle({ stage: 'provider_queued' }), '等待模型服务额度')
  const detail = trace.namespace.formatResearchTraceDetail({ stage: 'provider_queued', estimated_wait_seconds: 12.2 })
  assert.match(detail, /预计等待约 13 秒/)
  assert.match(detail, /可能.*变化.*可取消/)
  assert.equal(trace.namespace.formatResearchTraceTitle({ stage: 'provider_cancelled' }), '已取消等待模型服务')
})
