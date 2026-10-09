import { formatResearchTraceDetail, formatResearchTraceTitle } from './research-trace'
import { externalResearchModeLabel, normalizeExternalResearchMode } from './external-research-mode'

function asList(values) {
  return Array.isArray(values) ? values.filter(Boolean) : []
}

function normalizeTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toISOString()
}

function metricCount(values) {
  return asList(values).length
}

function renderBulletList(items) {
  const values = asList(items)
  if (!values.length) return ['- (none)']
  return values.map((item) => `- ${item}`)
}

function renderObjectList(items, formatter) {
  const values = asList(items)
  if (!values.length) return ['- (none)']
  return values.map((item, index) => `- ${formatter(item, index)}`)
}

export function buildResearchTraceBundle({
  sessionId = '',
  session = null,
  values = {},
  researchMode = 'external_only',
  stages = [],
  traceEvents = [],
  report = '',
  selectedPapers = [],
  comparisonMatrix = [],
  evidenceCards = [],
  reasoningSummaries = [],
  sourceRecords = [],
  keyFindings = [],
  uncertainties = [],
  ideaCandidates = [],
  externalGaps = [],
  followUpQuestions = [],
  candidatePapersToImport = [],
}) {
  const normalizedResearchMode = normalizeExternalResearchMode(researchMode)
  return {
    export_version: 1,
    exported_at: new Date().toISOString(),
    session: {
      session_id: sessionId,
      topic: session?.topic || values.user_query || '',
      status: values.awaiting_clarification ? 'awaiting_clarification' : 'completed',
      created_at: session?.created_at || '',
      updated_at: session?.updated_at || '',
      intent: values.intent || '',
      research_mode: normalizedResearchMode,
      research_mode_label: externalResearchModeLabel(normalizedResearchMode),
    },
    metrics: {
      selected_paper_count: metricCount(selectedPapers),
      comparison_row_count: metricCount(comparisonMatrix),
      evidence_count: metricCount(evidenceCards),
      reasoning_summary_count: metricCount(reasoningSummaries),
      source_record_count: metricCount(sourceRecords),
      key_finding_count: metricCount(keyFindings),
      uncertainty_count: metricCount(uncertainties),
      idea_count: metricCount(ideaCandidates),
      external_gap_count: metricCount(externalGaps),
      follow_up_count: metricCount(followUpQuestions),
      candidate_paper_count: metricCount(candidatePapersToImport),
    },
    stages: asList(stages).map((stage) => ({
      id: stage.id || '',
      label: stage.label || '',
      done: Boolean(stage.done),
      active: Boolean(stage.active),
      detail: stage.detail || '',
    })),
    trace_events: asList(traceEvents).map((event, index) => ({
      index: index + 1,
      at: normalizeTime(event.at),
      stage: event.stage || event.type || 'event',
      title: formatResearchTraceTitle(event),
      detail: formatResearchTraceDetail(event),
      raw: event,
    })),
    report,
    selected_papers: selectedPapers,
    comparison_matrix: comparisonMatrix,
    evidence_cards: evidenceCards,
    reasoning_summaries: reasoningSummaries,
    source_records: sourceRecords,
    key_findings: keyFindings,
    uncertainties,
    idea_candidates: ideaCandidates,
    external_gaps: externalGaps,
    follow_up_questions: followUpQuestions,
    candidate_papers_to_import: candidatePapersToImport,
  }
}

export function renderResearchTraceAppendix(bundle) {
  const session = bundle?.session || {}
  const metrics = bundle?.metrics || {}
  const stages = asList(bundle?.stages)
  const events = asList(bundle?.trace_events)

  return [
    '# Research Trace Appendix',
    '',
    '## Session',
    `- session_id: ${session.session_id || '(unknown)'}`,
    `- topic: ${session.topic || '(unknown)'}`,
    `- status: ${session.status || '(unknown)'}`,
    `- intent: ${session.intent || '(unknown)'}`,
    `- research_mode: ${session.research_mode_label || session.research_mode || '(unknown)'}`,
    `- created_at: ${normalizeTime(session.created_at) || '(unknown)'}`,
    `- updated_at: ${normalizeTime(session.updated_at) || '(unknown)'}`,
    `- exported_at: ${bundle?.exported_at || '(unknown)'}`,
    '',
    '## Metrics',
    `- selected_paper_count: ${metrics.selected_paper_count || 0}`,
    `- comparison_row_count: ${metrics.comparison_row_count || 0}`,
    `- evidence_count: ${metrics.evidence_count || 0}`,
    `- reasoning_summary_count: ${metrics.reasoning_summary_count || 0}`,
    `- source_record_count: ${metrics.source_record_count || 0}`,
    `- key_finding_count: ${metrics.key_finding_count || 0}`,
    `- uncertainty_count: ${metrics.uncertainty_count || 0}`,
    `- idea_count: ${metrics.idea_count || 0}`,
    `- external_gap_count: ${metrics.external_gap_count || 0}`,
    `- follow_up_count: ${metrics.follow_up_count || 0}`,
    `- candidate_paper_count: ${metrics.candidate_paper_count || 0}`,
    '',
    '## Stages',
    ...renderObjectList(stages, (stage) => `[${stage.done ? 'x' : ' '}] ${stage.label || stage.id}${stage.detail ? ` — ${stage.detail}` : ''}`),
    '',
    '## Timeline',
    ...renderObjectList(events, (event) => `${event.at || '(no time)'} · ${event.title}${event.detail ? ` — ${event.detail}` : ''}`),
    '',
    '## Key Findings',
    ...renderBulletList(bundle?.key_findings),
    '',
    '## Uncertainties',
    ...renderBulletList(bundle?.uncertainties),
    '',
    '## Idea Candidates',
    ...renderBulletList(bundle?.idea_candidates),
    '',
    '## External Gaps',
    ...renderBulletList(bundle?.external_gaps),
    '',
    '## Follow-up Questions',
    ...renderBulletList(bundle?.follow_up_questions),
    '',
    '## Candidate Papers To Import',
    ...renderObjectList(bundle?.candidate_papers_to_import, (item) => `${item.title || item.url || 'paper'}${item.url ? ` — ${item.url}` : ''}`),
    '',
    '## Reasoning Summaries',
    ...renderObjectList(bundle?.reasoning_summaries, (item) => `${item.focus || 'summary'} — ${item.summary || 'N/A'}${item.next_action ? `；Next: ${item.next_action}` : ''}`),
    '',
    '## Source Records',
    ...renderObjectList(bundle?.source_records, (item) => `${item.source_type || 'source'} — ${item.detail || item.query || 'N/A'}`),
    '',
    '## Evidence Cards',
    ...renderObjectList(bundle?.evidence_cards, (item) => `${item.title || item.card_id || 'evidence'} — ${item.summary || 'N/A'}`),
    '',
    '## Background Papers',
    ...renderObjectList(bundle?.selected_papers, (item) => `${item.title || item.paper_id || 'paper'}${item.method ? ` — ${item.method}` : ''}`),
  ].join('\n')
}
