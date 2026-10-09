/** 证据来源展示与历史恢复回归；作者：xxx */
import test from 'node:test'
import assert from 'node:assert/strict'
import { evidenceSourceLabel } from './evidence-source.mjs'
import { pickMessageExtras, restoredMessages } from './study-assistant-workspace.mjs'

test('跨页证据显示全部实际页码，不把兼容锚点当作唯一来源', () => {
  const record = {
    read_contract_version: 1, section: '实验', page_num: 2, source_location_status: 'exact',
    source_spans: [
      { source_fragments: [{ page_num: 4 }, { page_num: 3 }] },
      { source_fragments: [{ page_num: 4 }, { page_num: 6 }] },
    ],
  }
  assert.equal(evidenceSourceLabel(record), '实验 · pp.3、4、6')
})

test('缺失物理页码与旧格式块级定位分别说明', () => {
  assert.equal(evidenceSourceLabel({ read_contract_version: 1, source_location_status: 'legacy_chunk', page_num: 2 }),
    '来源块定位，未保存完整页码范围')
  assert.equal(evidenceSourceLabel({ read_contract_version: 1, source_location_status: 'no_physical_location', page_num: 1 }),
    '未提供页码')
  assert.equal(evidenceSourceLabel({ read_contract_version: 1, source_spans: [
    { source_fragments: [{ page_num: 5 }, { page_num: null }] },
  ] }), 'p.5 · 部分来源无页码')
  assert.equal(evidenceSourceLabel({ section: '旧章节', page_num: 7 }), '旧章节 · p.7')
})

test('历史消息恢复完整保留新来源与证据身份', () => {
  const record = {
    evidence_id: 'ev-v2-unchanged', read_contract_version: 1, material_group_id: 'mg-1',
    source_location_status: 'exact',
    source_spans: [{ chunk_id: 'v3:batch:0', field: 'text', start: 0, end: 2,
      text_start: 0, text_end: 2, source_fragments: [{ page_num: 3 }] }],
  }
  const extras = { evidence_records: [record], citations: [record] }
  assert.deepEqual(pickMessageExtras({ metadata: extras }), extras)
  const messages = restoredMessages({ session: { session_id: 'history', chat_history: [
    { role: 'assistant', content: '已保存回答', metadata: extras },
  ] } })
  assert.deepEqual(messages.at(-1).evidence_records, [record])
  assert.deepEqual(messages.at(-1).citations, [record])
})
