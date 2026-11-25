import { badge, button, card, el, emptyState, formatDate, icon, pageTitle } from '../components.js'

function recordCard(item, api) {
  const media = item.media_type === 'video' ? 'Video inspection' : 'Image analysis'
  const prediction = item.result?.assessment?.decision || item.result?.predicted_label || (item.result?.prediction_available === false ? 'No prediction' : 'Not returned')
  const detail = el('div', { className: 'hidden' })
  const detailsButton = button('Details', { className: 'button button-secondary button-small', onClick: async (event) => {
      const target = event.currentTarget
      target.disabled = true
      try {
        const record = await api.get(`/history/${encodeURIComponent(item.id)}`)
        detail.replaceChildren(el('pre', { className: 'report-json' }, JSON.stringify(record, null, 2)))
        detail.classList.toggle('hidden')
      } catch (error) {
        detail.replaceChildren(el('p', { className: 'settings-error' }, error.message)); detail.classList.remove('hidden')
      } finally { target.disabled = false }
    } })
  const actions = el('div', { className: 'toolbar' },
    detailsButton,
    el('a', { className: 'button button-secondary button-small', href: `${api.baseUrl}/reports/${encodeURIComponent(item.id)}.json`, target: '_blank', rel: 'noreferrer' }, icon('download', 13), el('span', {}, 'JSON')),
    el('a', { className: 'button button-secondary button-small', href: `${api.baseUrl}/reports/${encodeURIComponent(item.id)}.html`, target: '_blank', rel: 'noreferrer' }, icon('file', 13), el('span', {}, 'Report')),
  )
  const row = card(
    el('div', { className: 'recent-row' },
      el('div', { className: 'model-main' }, el('strong', {}, item.filename || 'Untitled file'), el('small', {}, `${media} · ${formatDate(item.created_at)} · ${prediction}`)),
      badge(item.status || 'complete', item.media_type === 'video' ? 'amber' : 'blue'),
    ),
    actions,
    detail,
  )
  row.style.padding = '13px'
  return row
}

export async function renderHistory(root, { api }) {
  const search = el('input', { className: 'text-input', type: 'search', placeholder: 'Search filename or analysis ID', 'aria-label': 'Search history' })
  const type = el('select', { className: 'select-input', 'aria-label': 'Filter media type' }, el('option', { value: '' }, 'All media'), el('option', { value: 'image' }, 'Images'), el('option', { value: 'video' }, 'Videos'))
  const from = el('input', { className: 'text-input', type: 'date', 'aria-label': 'From date' })
  const to = el('input', { className: 'text-input', type: 'date', 'aria-label': 'To date' })
  const total = el('span', { className: 'help-text' }, 'Loading…')
  const list = el('div', { className: 'grid' })
  let timer = null
  let active = true
  async function load() {
    if (!active) return
    list.replaceChildren(el('div', { className: 'loading-state' }, el('span', { className: 'spinner' }), 'Loading saved analyses…'))
    try {
      const params = new URLSearchParams({ query: search.value.trim(), media_type: type.value, limit: '500' })
      const data = await api.get(`/history?${params.toString()}`)
      if (!active) return
      const start = from.value ? new Date(`${from.value}T00:00:00`).getTime() : -Infinity
      const end = to.value ? new Date(`${to.value}T23:59:59.999`).getTime() : Infinity
      const items = (data.items || []).filter((item) => {
        const time = new Date(item.created_at).getTime()
        return time >= start && time <= end
      })
      total.textContent = `${items.length} shown · ${data.total} records returned (maximum 500)`
      if (!items.length) {
        list.replaceChildren(emptyState('No matching analyses', 'Try a different search or media filter. Uploads are not stored; history contains metadata, hashes, and model outputs.'))
      } else {
        list.replaceChildren(...items.map((item) => recordCard(item, api)))
      }
    } catch (error) {
      if (!active) return
      list.replaceChildren(el('div', { className: 'error-panel', role: 'alert' }, icon('alert', 20), el('div', {}, el('h2', {}, 'History is unavailable'), el('p', {}, error.message))))
      total.textContent = 'Could not load records'
    }
  }
  const schedule = () => { clearTimeout(timer); timer = setTimeout(load, 250) }
  search.addEventListener('input', schedule); type.addEventListener('change', load); from.addEventListener('change', load); to.addEventListener('change', load)
  root.append(pageTitle('Local records', 'Analysis history', 'Search and review saved metadata, model responses, and printable JSON/HTML reports.'),
    card(el('div', { className: 'toolbar' }, search, type, el('label', { className: 'field' }, el('span', { className: 'field-label' }, 'From'), from), el('label', { className: 'field' }, el('span', { className: 'field-label' }, 'To'), to), total)),
    el('section', { className: 'grid', style: { marginTop: '14px' } }, list))
  await load()
  return () => {
    active = false
    clearTimeout(timer)
  }
}
