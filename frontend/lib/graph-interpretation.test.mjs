import test from 'node:test'
import assert from 'node:assert/strict'

import { buildGraphInterpretation } from './graph-interpretation.mjs'
import { formatRelationLabel } from './relation-meanings.mjs'

test('graph interpretation summarizes relations without fabricating low-signal insights', () => {
  const low = buildGraphInterpretation({ nodes: [{ id: 'p1', paper_id: 'p1' }], edges: [], meta: {} })
  assert.equal(low.lowSignal, true)
  assert.deepEqual(low.insights, [])

  const graph = buildGraphInterpretation({
    nodes: [{ id: 'a', paper_id: 'pa', title: 'A' }, { id: 'b', paper_id: 'pb', title: 'B' }],
    edges: [{ source: 'a', target: 'b', type: 'USES_METHOD', shared_items: ['RAG'] }],
    meta: { relation_counts: { USES_METHOD: 1 } },
  })
  assert.equal(graph.byRelation[0].label, '使用相近方法')
  assert.equal(graph.topPapers[0].degree, 1)
  assert.equal(graph.topSharedItems[0].item, 'RAG')
  assert.equal(formatRelationLabel('UNKNOWN_EDGE'), 'unknown edge')
})
