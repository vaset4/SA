/** 同次引用坐标和真实 Markdown 渲染回归；作者：xxx */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { fileURLToPath, pathToFileURL } from 'node:url'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { finalCitationRecords, prepareFinalCitations } from './final-citations.mjs'
import { pickMessageExtras } from './study-assistant-workspace.mjs'

const require = createRequire(import.meta.url)
const { transform } = require('next/dist/build/swc')

/** @param {URL} url 产品组件地址 @returns {Promise<object>} 实际组件模块 */
async function loadComponent(url) {
  const source = await readFile(url, 'utf8')
  const output = await transform(source, { filename: fileURLToPath(url),
    jsc: { parser: { syntax: 'ecmascript', jsx: true }, transform: { react: { runtime: 'automatic' } } },
    module: { type: 'es6' } })
  const code = output.code.replace(/from (["'])([^"']+)\1/g, (_, quote, specifier) => {
    const resolved = specifier.startsWith('.') ? new URL(specifier, url).href : pathToFileURL(require.resolve(specifier)).href
    return `from ${quote}${resolved}${quote}`
  })
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)
}
const { MessageContent } = await loadComponent(new URL('../components/chat/MessageContent.jsx', import.meta.url))
const { MessageResearchPlannerSummary } = await loadComponent(new URL('../components/chat/MessageResearchPlannerSummary.jsx', import.meta.url))

/** @param {string} content 原始正文 @param {number[]} starts 明确的码点位置 @returns {object} 同次消息 */
function message(content, starts) {
  const bindings = starts.map((start, index) => ({ start, end: start + 4, alias: 'E1',
    evidence_id: 'ev-r1-real', attempt_id: 'attempt-1', block_id: `block-${index}` }))
  return { id: 'm-1', role: 'assistant', content, renderMode: 'markdown',
    final_generation_attempt: { attempt_id: 'attempt-1' }, final_citation_bindings: bindings,
    final_delivery_report: { contract_version: 'research-planner-final-v3', attempt_id: 'attempt-1',
      status: 'partial', citation_bindings: bindings },
    evidence_records: [{ evidence_id: 'ev-r1-real', paper_id: 'paper-1', excerpt: '保存原文' }] }
}

/** @param {object} item 消息 @returns {string} 产品组件实际服务端渲染结果 */
function render(item) {
  return renderToStaticMarkup(React.createElement(MessageContent, { message: item }))
}

test('表情、组合字符及多块换行按码点转换，坏的同名标记不链接', () => {
  const content = '😀e\u0301 [E1]\n\n未授权 [E1]'
  const item = message(content, [4])
  assert.equal(prepareFinalCitations(item, content).bindings[0].start, 5)
  const html = render(item)
  assert.equal((html.match(/href="#evidence-m-1-ev-r1-real"/g) || []).length, 1)
  assert.match(html, /未授权 \[E1\]/)
  assert.deepEqual(finalCitationRecords(item).map((record) => [record.evidence_id, record.citation_alias]), [['ev-r1-real', 'E1']])
})

test('完整表格和列表保持结构，引用映射到全文位置', () => {
  const content = '| 条件 | 来源 |\n| --- | --- |\n| 😀 | [E1] |\n\n- 第二块'
  const item = message(content, [[...content.slice(0, content.indexOf('[E1]'))].length])
  const html = render(item)
  assert.match(html, /<table>/)
  assert.match(html, /<li>第二块<\/li>/)
  assert.match(html, /href="#evidence-m-1-ev-r1-real"/)
})

test('代码字面量、模型伪造内部锚点和危险协议保持安全', () => {
  const content = '`[E1]`\n\n```txt\n[E1]\n```\n\n[x](#evidence-m-1-ev-r1-real) [bad](javascript:alert%281%29)'
  const item = message(content, [1, 15])
  const html = render(item)
  assert.doesNotMatch(html, /<a /)
  assert.match(html, /<code>\[E1\]<\/code>/)
  assert.doesNotMatch(html, /javascript:/)
})

test('原始正文前导空白、CRLF及合法标记不在渲染前清洗', () => {
  const content = '  [paper_id=p1] 😀 [E1]\r\n后续'
  const item = message(content, [[...content.slice(0, content.indexOf('[E1]'))].length])
  assert.match(render(item), /paper_id=p1/)
  assert.match(render(item), /href="#evidence-m-1-ev-r1-real"/)
})

test('跨尝试或畸形坐标不生成链接且显示风险', () => {
  const item = message('答复 [E1]', [3])
  item.final_generation_attempt = { attempt_id: 'different' }
  const html = render(item)
  assert.doesNotMatch(html, /<a /)
  assert.match(html, /尝试身份不一致/)
  const malformed = message('答复 [E1]', [-1])
  assert.match(render(malformed), /无法核对/)
  const missingContract = { ...message('答复 [E1]', [3]), final_delivery_report: { contract_version: 'unknown' } }
  assert.doesNotMatch(render(missingContract), /<a /)
  assert.match(render(missingContract), /可识别的引用合同/)
})

test('Markdown转义和字符实体不会改变后方绑定位置', () => {
  const content = 'A &amp; B \\*字面\\* 😀 [E1]'
  const item = message(content, [[...content.slice(0, content.indexOf('[E1]'))].length])
  const html = render(item)
  assert.match(html, /A &amp; B \*字面\*/)
  assert.match(html, /href="#evidence-m-1-ev-r1-real"/)
})

test('消息历史白名单保留同次交付，历史无报告不补造完成事实', () => {
  const item = message('答复 [E1]', [3])
  const restored = pickMessageExtras({ metadata: item })
  assert.deepEqual(restored.final_delivery_report, item.final_delivery_report)
  assert.deepEqual(restored.final_citation_bindings, item.final_citation_bindings)
  assert.deepEqual(prepareFinalCitations({ content: '旧 [E1]' }, '旧 [E1]').bindings, [])
  assert.equal(pickMessageExtras({ content: '旧正文' }).final_delivery_report, undefined)
})

test('最终内容不足与取证不足独立显示，历史消息保持原有语义', () => {
  const item = { runtime_chain: 'planner', source_status: { planner: 'ok', planner_final: 'skipped' },
    final_delivery_report: { status: 'insufficient', valid_business_block_ids: [] } }
  const html = renderToStaticMarkup(React.createElement(MessageResearchPlannerSummary, { message: item }))
  assert.match(html, /取证 完成/)
  assert.match(html, /最终回答 内容不足/)
  assert.doesNotMatch(html, /最终回答 证据不足/)
  const legacy = { runtime_chain: 'planner', source_status: { planner: 'insufficient', planner_final: 'skipped' } }
  const oldHtml = renderToStaticMarkup(React.createElement(MessageResearchPlannerSummary, { message: legacy }))
  assert.match(oldHtml, /取证 证据不足/)
  assert.match(oldHtml, /最终回答 已跳过/)
})
