const SVG_NS = 'http://www.w3.org/2000/svg'

const ICONS = {
  dashboard: [['rect', { x: 3, y: 3, width: 8, height: 8, rx: 1 }], ['rect', { x: 13, y: 3, width: 8, height: 5, rx: 1 }], ['rect', { x: 13, y: 10, width: 8, height: 11, rx: 1 }], ['rect', { x: 3, y: 13, width: 8, height: 8, rx: 1 }]],
  image: [['rect', { x: 3, y: 3, width: 18, height: 18, rx: 3 }], ['circle', { cx: 8.5, cy: 8.5, r: 1.5 }], ['path', { d: 'm21 15-5-5L5 21' }]],
  video: [['rect', { x: 3, y: 5, width: 13, height: 14, rx: 2 }], ['path', { d: 'm16 10 5-3v10l-5-3' }]],
  history: [['path', { d: 'M3 12a9 9 0 1 0 2.6-6.4L3 8' }], ['path', { d: 'M3 3v5h5M12 7v5l3 2' }]],
  chart: [['path', { d: 'M3 3v18h18' }], ['path', { d: 'm7 14 4-4 4 3 6-7' }]],
  system: [['path', { d: 'M12 3v3m0 12v3M3 12h3m12 0h3M5.64 5.64l2.12 2.12m8.48 8.48 2.12 2.12m0-12.72-2.12 2.12m-8.48 8.48-2.12 2.12' }], ['circle', { cx: 12, cy: 12, r: 4 }]],
  settings: [['circle', { cx: 12, cy: 12, r: 3 }], ['path', { d: 'm19.4 15 .1.1 1.4 1.1-1.4 2.4-1.7-.7a8 8 0 0 1-1.6.9L16 20h-2.8l-.3-1.8a8 8 0 0 1-1.7-.9l-1.7.7-1.4-2.4 1.4-1.1a7 7 0 0 1 0-1.9l-1.4-1.1 1.4-2.4 1.7.7a8 8 0 0 1 1.7-.9L13.2 7H16l.3 1.8a8 8 0 0 1 1.6.9l1.7-.7 1.4 2.4-1.4 1.1a7 7 0 0 1-.2 2.5Z' }]],
  info: [['circle', { cx: 12, cy: 12, r: 9 }], ['path', { d: 'M12 11v5m0-8h.01' }]],
  shield: [['path', { d: 'M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z' }], ['path', { d: 'm9 12 2 2 4-4' }]],
  arrow: [['path', { d: 'M5 12h14m-7-7 7 7-7 7' }]],
  upload: [['path', { d: 'M12 16V4m-5 5 5-5 5 5' }], ['path', { d: 'M4 16v4h16v-4' }]],
  download: [['path', { d: 'M12 4v12m-5-5 5 5 5-5' }], ['path', { d: 'M4 17v3h16v-3' }]],
  check: [['path', { d: 'm5 12 4 4L19 6' }]],
  alert: [['path', { d: 'M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20h15.4a2 2 0 0 0 1.7-2.8L13.7 3.9a2 2 0 0 0-3.4 0Z' }], ['path', { d: 'M12 9v4m0 3h.01' }]],
  file: [['path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }], ['path', { d: 'M14 2v6h6M8 13h8m-8 4h8' }]],
  close: [['path', { d: 'm18 6-12 12M6 6l12 12' }]],
  menu: [['path', { d: 'M4 6h16M4 12h16M4 18h16' }]],
  chevron: [['path', { d: 'm9 18 6-6-6-6' }]],
  refresh: [['path', { d: 'M20 7v5h-5' }], ['path', { d: 'M20 12a8 8 0 1 1-2.35-5.65L20 9' }]],
  search: [['circle', { cx: 11, cy: 11, r: 7 }], ['path', { d: 'm20 20-4-4' }]],
  clock: [['circle', { cx: 12, cy: 12, r: 9 }], ['path', { d: 'M12 7v5l3 2' }]],
  cpu: [['rect', { x: 6, y: 6, width: 12, height: 12, rx: 2 }], ['path', { d: 'M9 1v3m6-3v3M9 20v3m6-3v3M20 9h3m-3 6h3M1 9h3m-3 6h3M9 9h6v6H9z' }]],
  database: [['ellipse', { cx: 12, cy: 5, rx: 8, ry: 3 }], ['path', { d: 'M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3' }]],
  external: [['path', { d: 'M14 3h7v7m-1-6-9 9' }], ['path', { d: 'M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6' }]],
  trash: [['path', { d: 'M3 6h18m-2 0-.9 14H5.9L5 6m3 0V4h8v2m-7 4v6m4-6v6' }]],
  imageStack: [['rect', { x: 3, y: 3, width: 14, height: 14, rx: 2 }], ['path', { d: 'm5 14 4-4 3 3 2-2 3 3M7 7h.01' }], ['path', { d: 'M7 19h12a2 2 0 0 0 2-2V7' }]],
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag)
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue
    if (key === 'className') node.className = value
    else if (key === 'text') node.textContent = String(value)
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value)
    else if (key === 'dataset' && typeof value === 'object') Object.assign(node.dataset, value)
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2).toLowerCase(), value)
    else if (key in node && key !== 'list' && key !== 'form') node[key] = value
    else node.setAttribute(key, value === true ? '' : String(value))
  }
  const append = (child) => {
    if (child === undefined || child === null || child === false) return
    if (Array.isArray(child)) child.forEach(append)
    else if (child instanceof Node) node.append(child)
    else node.append(document.createTextNode(String(child)))
  }
  children.forEach(append)
  return node
}

export function icon(name, size = 18) {
  const svg = document.createElementNS(SVG_NS, 'svg')
  svg.setAttribute('viewBox', '0 0 24 24')
  svg.setAttribute('width', String(size))
  svg.setAttribute('height', String(size))
  svg.setAttribute('fill', 'none')
  svg.setAttribute('stroke', 'currentColor')
  svg.setAttribute('stroke-width', '1.7')
  svg.setAttribute('stroke-linecap', 'round')
  svg.setAttribute('stroke-linejoin', 'round')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  for (const [tag, attrs] of ICONS[name] || ICONS.info) {
    const shape = document.createElementNS(SVG_NS, tag)
    for (const [key, value] of Object.entries(attrs)) shape.setAttribute(key, String(value))
    svg.append(shape)
  }
  return svg
}

export const button = (label, { variant = 'secondary', iconName, ...attrs } = {}) => el(
  'button', { className: `button button-${variant}`, type: 'button', ...attrs },
  iconName ? icon(iconName, 16) : null,
  el('span', {}, label),
)

export const card = (...children) => el('section', { className: 'card' }, ...children)
export const pageTitle = (eyebrow, title, description, actions = null) => el(
  'header', { className: 'page-title' },
  el('div', {}, el('p', { className: 'eyebrow' }, eyebrow), el('h1', {}, title), el('p', { className: 'page-description' }, description)),
  actions ? el('div', { className: 'page-actions' }, actions) : null,
)

export const badge = (label, tone = 'neutral') => el('span', { className: `badge badge-${tone}` }, label)

export function sectionHeading(title, description = '', action = null) {
  return el('div', { className: 'section-heading' },
    el('div', {}, el('h2', {}, title), description ? el('p', {}, description) : null),
    action,
  )
}

export function formatPercent(value, digits = 2) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(digits)}%` : 'Not recorded'
}

export function formatBytes(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric < 0) return 'Not reported'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = numeric
  let index = 0
  while (amount >= 1024 && index < units.length - 1) { amount /= 1024; index += 1 }
  return `${amount.toFixed(index ? 1 : 0)} ${units[index]}`
}

export function formatDate(value) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

export function metricTile(label, value, note = '') {
  return el('article', { className: 'metric-tile' }, el('p', {}, label), el('strong', {}, value), note ? el('span', {}, note) : null)
}

export function notice(title, body, tone = 'info') {
  return el('aside', { className: `notice notice-${tone}`, role: tone === 'warning' ? 'alert' : 'note' },
    icon(tone === 'warning' ? 'alert' : 'info', 18),
    el('div', {}, el('strong', {}, title), el('p', {}, body)),
  )
}

export function errorPanel(error, retry) {
  const children = [icon('alert', 20), el('div', {}, el('h2', {}, 'Unable to load this view'), el('p', {}, error?.message || 'The backend request failed.'))]
  if (retry) children.push(button('Retry', { iconName: 'arrow', onClick: retry }))
  return el('div', { className: 'error-panel', role: 'alert' }, ...children)
}

export function emptyState(title, message, action = null) {
  return el('div', { className: 'empty-state' }, icon('file', 24), el('h3', {}, title), el('p', {}, message), action)
}
