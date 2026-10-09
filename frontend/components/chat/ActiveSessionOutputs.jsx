'use client'

import Link from 'next/link'

function OutputEntry({ item }) {
  const content = (
    <>
      <strong>{item.label}</strong>
      <span>{item.state}</span>
    </>
  )
  if (!item.href) return <div className="sa-active-output-item">{content}</div>
  return <Link href={item.href} className="sa-active-output-item">{content}</Link>
}

export function ActiveSessionOutputs({ items = [], label = '' }) {
  if (!items.length) return null
  return (
    <>
      {label ? <p className="sa-research-room-label">{label}</p> : null}
      <div className="sa-active-output-list">
        {items.map((item) => <OutputEntry key={item.id} item={item} />)}
      </div>
    </>
  )
}
