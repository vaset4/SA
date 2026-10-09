// 【研究页面】【交付回归】真实页面模型隔离验证，作者：xxx
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createContext, SourceTextModule } from 'node:vm'

/** @param {string} relative 入口相对路径 @returns {Promise<object>} 真实前端模块导出 */
async function loadModule(relative) {
  const context = createContext({ console })
  const cache = new Map()
  /** @param {string} filename 本地路径 @returns {Promise<SourceTextModule>} 已关联模块 */
  async function load(filename) {
    if (cache.has(filename)) return cache.get(filename)
    const module = new SourceTextModule(await fs.readFile(filename, 'utf8'), { context, identifier: filename })
    cache.set(filename, module)
    await module.link((specifier, parent) => {
      const resolved = path.resolve(path.dirname(parent.identifier), specifier)
      return load(path.extname(resolved) ? resolved : `${resolved}.js`)
    })
    return module
  }
  const module = await load(fileURLToPath(new URL(relative, import.meta.url)))
  await module.evaluate()
  return module.namespace
}

const view = await loadModule('./research-session-view.js')

test('研究页只展示已提交产物，不展示较新的未提交checkpoint正文', () => {
  const answer = '  😀 已保存的结论\n\n来源：[E1]'
  const session = {
    status: 'running', current_run_id: 'operation-b', result_run_id: 'run-a', delivery_revision: 1,
    current_result: { answer, final_delivery_report: { status: 'partial' }, persistence_status: 'saved' },
    values: { report: '尚未提交的新正文', final_delivery_report: { status: 'complete' } },
  }
  const model = view.buildResearchSessionModel({ sessionId: 's', session })
  assert.equal(model.report, answer)
  assert.equal(model.values.final_delivery_report.status, 'partial')
})

test('新操作尚未提交时，checkpoint非空不能制造可展示报告', () => {
  const model = view.buildResearchSessionModel({
    sessionId: 's', session: { status: 'running', current_run_id: 'op', current_result: null, values: { report: '未保存正文' } },
  })
  assert.equal(model.report, '')
  assert.equal(model.stages.find((stage) => stage.id === 'report').done, false)
})

for (const [status, label] of [['complete', '回答完整'], ['partial', '部分回答'], ['insufficient', '资料不足'], ['failed', '交付失败']]) {
  test(`研究已结束时显示真实交付状态：${status}`, () => {
    const model = view.buildResearchSessionModel({ sessionId: 's', session: {
      status: 'completed', result_run_id: 'run-a', delivery_revision: 1,
      current_result: { answer: '实际正文', final_delivery_report: { status }, persistence_status: 'saved' },
    } })
    assert.equal(view.headerStatus({}, model).label, label)
    assert.equal(model.stages.find((stage) => stage.id === 'report').label, label)
  })
}

test('取消请求仅受理时显示正在停止，不制造已取消终态', () => {
  const model = view.buildResearchSessionModel({ sessionId: 's', session: { status: 'running', cancel_requested: true } })
  assert.equal(view.headerStatus({}, model).label, '正在停止')
})
