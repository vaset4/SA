'use client'

function evidenceTitle(card) {
  return card?.title || card?.card_id || card?.summary || '证据线索'
}

function evidenceSummary(card) {
  return card?.summary || card?.url || '暂无摘要'
}

function EvidenceCard({ card, index }) {
  return (
    <article className="sa-evidence-desk__card">
      <span className="sa-evidence-desk__label">{card?.card_type || card?.source_type || `Evidence ${index + 1}`}</span>
      <strong>{evidenceTitle(card)}</strong>
      <p>{evidenceSummary(card)}</p>
      {card?.url ? <a href={card.url} target="_blank" rel="noreferrer">查看来源</a> : null}
    </article>
  )
}

function TextCard({ label, text }) {
  return (
    <article className="sa-evidence-desk__card">
      <span className="sa-evidence-desk__label">{label}</span>
      <strong>{text}</strong>
    </article>
  )
}

export function EvidenceDeskSection({ model }) {
  const cards = model.evidenceCards.length ? model.evidenceCards : []
  const uncertaintyCards = model.uncertainties.slice(0, 2)
  if (!cards.length && !uncertaintyCards.length && !model.keyFindings.length) return null
  return (
    <section className="sa-evidence-desk sa-evidence-desk-panel">
      <header className="sa-evidence-desk__head">
        <h3 className="sa-evidence-desk__title">证据与不确定性</h3>
        <span className="sa-evidence-desk__badge">review</span>
      </header>
      <div className="sa-evidence-desk__grid">
        {model.keyFindings.slice(0, 1).map((item) => <TextCard key={item} label="Finding" text={item} />)}
        {cards.slice(0, 3).map((card, index) => <EvidenceCard key={card.card_id || card.title || index} card={card} index={index} />)}
        {uncertaintyCards.map((item) => <TextCard key={item} label="Uncertainty" text={item} />)}
      </div>
    </section>
  )
}
