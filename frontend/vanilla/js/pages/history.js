import { badge, button, card, el, emptyState, formatDate, icon, pageTitle } from '../components.js'

const PAGE_SIZE = 20

function safeRecord(value) {
  if (Array.isArray(value)) return value.map((item) => safeRecord(item))
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value).map(([key, item]) => {
    if (/path/i.test(key)) return [key, '[omitted]']
    if (/_base64$/i.test(key) && typeof item === 'string') return [key, `[base64 image omitted · ${item.length} chars]`]
    return [key, safeRecord(item)]
  }))
}

function recordResult(item) {
  return item.result?.assessment?.decision
    || item.result?.predicted_label
    || item.result?.prediction
    || (item.result?.prediction_available === false ? 'Metadata only' : 'Not returned')
}

function reportLink(item, api, kind, label, iconName) {
  return el('a', {
    className: 'button button-secondary button-small',
    href: api.baseUrl + '/reports/' + encodeURIComponent(item.id) + '.' + kind,
    target: '_blank',
    rel: 'noreferrer',
    'aria-label': label + ' for ' + (item.filename || 'analysis'),
  }, icon(iconName, 14), el('span', {}, label))
}

function rowFor(item, api) {
  const detailRow = el('tr', { className: 'history-detail-row hidden' }, el('td', { colspan: '7' },
    el('div', { className: 'loading-state' }, 'Select Details to load the saved record.')))
  let loadedRecord = null
  let open = false
  const detailButton = button('Details', {
    className: 'button button-secondary button-small',
    'aria-expanded': 'false',
    onClick: async (event) => {
      const control = event.currentTarget
      open = !open
      control.setAttribute('aria-expanded', String(open))
      detailRow.classList.toggle('hidden', !open)
      if (!open || loadedRecord) return
      control.disabled = true
      detailRow.firstElementChild.firstElementChild.replaceChildren(el('span', { className: 'spinner' }), 'Loading record…')
      try {
        const response = await api.get('/history/' + encodeURIComponent(item.id))
        loadedRecord = safeRecord(response)
        detailRow.firstElementChild.replaceChildren(el('pre', { className: 'report-json' }, JSON.stringify(loadedRecord, null, 2)))
      } catch (error) {
        detailRow.firstElementChild.replaceChildren(el('div', { className: 'error-panel', role: 'alert' },
          icon('alert', 18), el('div', {}, el('h2', {}, 'Record details are unavailable'), el('p', {}, error.message))))
      } finally {
        control.disabled = false
      }
    },
  })
  const actions = el('div', { className: 'table-actions' },
    detailButton,
    reportLink(item, api, 'json', 'JSON', 'download'),
    reportLink(item, api, 'html', 'Report', 'file'),
  )
  const row = el('tr', {},
    el('td', { 'data-label': 'File and analysis ID' }, el('strong', { title: item.filename || 'Untitled file' }, item.filename || 'Untitled file'), el('small', {}, 'ID ' + (item.id || 'not recorded'))),
    el('td', { 'data-label': 'Media' }, badge(item.media_type === 'video' ? 'Video' : 'Image', item.media_type === 'video' ? 'amber' : 'blue')),
    el('td', { 'data-label': 'Model' }, item.result?.model_identifier || item.model_id || '—'),
    el('td', { 'data-label': 'Result' }, recordResult(item)),
    el('td', { 'data-label': 'Created' }, formatDate(item.created_at)),
    el('td', { 'data-label': 'Status' }, badge(item.status || 'complete', item.status === 'failed' ? 'red' : 'green')),
    el('td', { 'data-label': 'Actions' }, actions),
  )
  return [row, detailRow]
}

export async function renderHistory(root, { api }) {
  const search = el('input', { className: 'text-input', type: 'search', placeholder: 'Filename or analysis ID', 'aria-label': 'Search history', autocomplete: 'off' })
  const type = el('select', { className: 'select-input', 'aria-label': 'Filter media type' },
    el('option', { value: '' }, 'All media'), el('option', { value: 'image' }, 'Images'), el('option', { value: 'video' }, 'Videos'))
  const from = el('input', { className: 'text-input date-control', type: 'date', 'aria-label': 'From date' })
  const to = el('input', { className: 'text-input date-control', type: 'date', 'aria-label': 'To date' })
  const order = el('select', { className: 'select-input', 'aria-label': 'Sort order' },
    el('option', { value: 'newest' }, 'Newest first'), el('option', { value: 'oldest' }, 'Oldest first'))
  const errorMessage = el('p', { className: 'settings-error', role: 'alert' })
  const summary = el('div', { className: 'history-summary', role: 'status' })
  const tableArea = el('div', { className: 'history-results' })
  const pagination = el('div', { className: 'pagination' })
  let timer = null
  let active = true
  let rows = []
  let apiTotal = 0
  let page = 1

  function visibleRows() {
    const start = from.value ? new Date(from.value + 'T00:00:00').getTime() : -Infinity
    const end = to.value ? new Date(to.value + 'T23:59:59.999').getTime() : Infinity
    return rows.filter((item) => {
      const time = new Date(item.created_at).getTime()
      return time >= start && time <= end
    }).sort((a, b) => {
      const delta = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      return order.value === 'oldest' ? delta : -delta
    })
  }

  function renderRows() {
    const filtered = visibleRows()
    const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
    page = Math.min(page, pages)
    const firstIndex = (page - 1) * PAGE_SIZE
    const current = filtered.slice(firstIndex, firstIndex + PAGE_SIZE)
    if (from.value && to.value && from.value > to.value) {
      errorMessage.textContent = 'The start date must be on or before the end date.'
      tableArea.replaceChildren(emptyState('Date range needs attention', 'Choose a valid date interval.'))
      summary.textContent = ''
      pagination.replaceChildren()
      return
    }
    errorMessage.textContent = ''
    if (!current.length) {
      tableArea.replaceChildren(emptyState('No matching analyses', 'Try another search or date range. History contains metadata, file hashes, and model outputs; uploaded media is not retained.'))
      summary.textContent = '0 records shown · ' + apiTotal + ' returned by the backend'
      pagination.replaceChildren()
      return
    }
    const table = el('table', { className: 'history-table' },
      el('thead', {}, el('tr', {},
        ...['File and analysis ID', 'Media', 'Model', 'Result', 'Created', 'Status', 'Actions'].map((label) => el('th', { scope: 'col' }, label)),
      )),
      el('tbody', {}, current.map((item) => rowFor(item, api))),
    )
    tableArea.replaceChildren(el('div', { className: 'history-table-wrap' }, table))
    const toIndex = Math.min(firstIndex + PAGE_SIZE, filtered.length)
    summary.replaceChildren(
      el('span', {}, 'Showing ' + (firstIndex + 1) + '–' + toIndex + ' of ' + filtered.length + ' matching records'),
      el('span', {}, 'Backend returned ' + apiTotal + (apiTotal >= 500 ? ' (maximum 500 loaded)' : '')),
    )
    pagination.replaceChildren(
      button('Previous', { className: 'button button-secondary button-small', disabled: page <= 1, onClick: () => { page -= 1; renderRows() } }),
      el('span', {}, 'Page ' + page + ' of ' + pages),
      button('Next', { className: 'button button-secondary button-small', disabled: page >= pages, onClick: () => { page += 1; renderRows() } }),
    )
  }

  async function load() {
    if (!active) return
    tableArea.replaceChildren(el('div', { className: 'loading-state' }, el('span', { className: 'spinner' }), 'Loading saved analyses…'))
    errorMessage.textContent = ''
    try {
      const params = new URLSearchParams({ query: search.value.trim(), media_type: type.value, limit: '500' })
      const data = await api.get('/history?' + params.toString())
      if (!active) return
      rows = Array.isArray(data.items) ? data.items : []
      apiTotal = Number(data.total || 0)
      page = 1
      renderRows()
    } catch (error) {
      if (!active) return
      tableArea.replaceChildren(el('div', { className: 'error-panel', role: 'alert' },
        icon('alert', 20), el('div', {}, el('h2', {}, 'History is unavailable'), el('p', {}, error.message))))
      summary.textContent = 'Could not load records'
      pagination.replaceChildren()
    }
  }

  const schedule = () => { clearTimeout(timer); timer = setTimeout(load, 250) }
  const clear = button('Clear filters', { variant: 'secondary', iconName: 'close', className: 'button button-secondary button-small', onClick: () => {
    search.value = ''; type.value = ''; from.value = ''; to.value = ''; order.value = 'newest'
    load()
  } })
  search.addEventListener('input', schedule)
  type.addEventListener('change', load)
  from.addEventListener('change', () => { page = 1; renderRows() })
  to.addEventListener('change', () => { page = 1; renderRows() })
  order.addEventListener('change', () => { page = 1; renderRows() })
  root.append(pageTitle('Saved local records', 'Analysis history', 'Search, filter, and inspect saved metadata and model outputs. Reports open in a separate tab.'),
    card(el('div', { className: 'history-controls' },
      el('label', { className: 'field search-control' }, el('span', { className: 'field-label' }, 'Search'), search),
      el('label', { className: 'field' }, el('span', { className: 'field-label' }, 'Media type'), type),
      el('label', { className: 'field date-control' }, el('span', { className: 'field-label' }, 'From'), from),
      el('label', { className: 'field date-control' }, el('span', { className: 'field-label' }, 'To'), to),
      el('label', { className: 'field' }, el('span', { className: 'field-label' }, 'Sort'), order),
      clear,
    ), errorMessage),
    summary, tableArea, pagination,
  )
  await load()
  return () => { active = false; clearTimeout(timer) }
}
