export const EXTERNAL_RESEARCH_MODE = 'external_only'

export function normalizeExternalResearchMode(mode) {
  const normalized = String(mode || '').trim().toLowerCase().replace(/[\s-]+/g, '_')
  if (
    normalized === EXTERNAL_RESEARCH_MODE
    || normalized === 'external'
    || normalized === 'plan'
    || normalized === 'paper_first'
    || normalized === 'hybrid'
    || normalized === 'web_first'
  ) {
    return EXTERNAL_RESEARCH_MODE
  }
  return EXTERNAL_RESEARCH_MODE
}

export function externalResearchModeLabel(mode) {
  const normalized = normalizeExternalResearchMode(mode)
  return normalized === EXTERNAL_RESEARCH_MODE ? 'External Only' : 'External Only'
}
