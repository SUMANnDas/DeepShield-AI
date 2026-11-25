import { createApiClient } from './api.js'
import { el, icon, errorPanel } from './components.js'
import { renderDashboard } from './pages/dashboard.js'
import { renderImageAnalysis } from './pages/images.js'
import { renderVideoAnalysis } from './pages/videos.js'
import { renderHistory } from './pages/history.js'
import { renderPerformance } from './pages/performance.js'
import { renderSystem } from './pages/system.js'
import { renderSettings } from './pages/settings.js'
import { renderAbout } from './pages/about.js'

const NAV = [
  ['dashboard', 'Overview', 'dashboard', 'Workspace'],
  ['images', 'Image analysis', 'image', 'Analyze'],
  ['videos', 'Video inspection', 'video', 'Analyze'],
  ['history', 'Analysis history', 'history', 'Review'],
  ['performance', 'Model performance', 'chart', 'Review'],
  ['system', 'System health', 'system', 'System'],
  ['settings', 'Settings', 'settings', 'System'],
  ['about', 'About DeepShield', 'info', 'System'],
]

const RENDERERS = {
  dashboard: renderDashboard,
  images: renderImageAnalysis,
  videos: renderVideoAnalysis,
  history: renderHistory,
  performance: renderPerformance,
  system: renderSystem,
  settings: renderSettings,
  about: renderAbout,
}

const host = document.querySelector('#app')
let activeCleanup = null
let renderToken = 0
let backendCheckPending = false
const BACKEND_HEALTH_INTERVAL_MS = 10_000
const nav = el('nav', { className: 'side-nav', 'aria-label': 'Main navigation' })
const page = el('main', { id: 'main-content', className: 'main-content', tabindex: '-1' })
const backendStatus = el('span', { className: 'connection-status', role: 'status' }, el('i', { className: 'status-dot' }), 'Checking backend')
const menuButton = el('button', { className: 'icon-button mobile-menu-button', type: 'button', 'aria-label': 'Open navigation', 'aria-expanded': 'false' }, icon('menu', 20))
const closeButton = el('button', { className: 'icon-button sidebar-close-button', type: 'button', 'aria-label': 'Close navigation' }, icon('close', 19))
const scrim = el('button', { className: 'sidebar-scrim', type: 'button', 'aria-label': 'Close navigation', tabindex: '-1' })
const sidebarHeading = el('div', { className: 'sidebar-heading' },
  el('a', { className: 'brand', href: '#/dashboard', 'aria-label': 'DeepShield AI overview' },
    el('span', { className: 'brand-mark' }, icon('shield', 22)),
    el('span', {}, el('strong', {}, 'DeepShield'), el('small', {}, 'MEDIA FORENSICS')),
  ), closeButton,
)

const sidebar = el('aside', { className: 'sidebar' },
  sidebarHeading,
  el('div', { className: 'sidebar-kicker' }, el('span', { className: 'workspace-indicator' }), el('span', {}, 'Local research workspace')),
  nav,
  el('div', { className: 'sidebar-caution' }, icon('info', 17), el('span', {}, 'Research indicators require independent review.')),
  el('div', { className: 'workspace-chip' }, el('span', { className: 'workspace-avatar' }, 'DS'), el('span', {}, el('strong', {}, 'Local session'), el('small', {}, 'Private workspace'))),
)

const frame = el('div', { className: 'app-frame' },
  scrim,
  sidebar,
  el('section', { className: 'workspace' },
    el('header', { className: 'topbar' },
      el('div', { className: 'topbar-leading' }, menuButton, el('div', { className: 'breadcrumbs' }, el('span', { className: 'breadcrumb-root' }, 'DeepShield AI'), icon('chevron', 14), el('span', { id: 'current-section' }, 'Overview'))),
      el('div', { className: 'topbar-actions' }, backendStatus, el('a', { id: 'api-docs-link', className: 'api-docs-link', href: 'http://127.0.0.1:8000/docs', target: '_blank', rel: 'noreferrer' }, 'API docs')),
    ),
    page,
  ),
)
host.replaceChildren(frame)

function updateNavigation(route) {
  const groups = [...new Set(NAV.map(([, , , group]) => group))]
  nav.replaceChildren(...groups.map((group) => el('div', { className: 'nav-group' },
    el('p', { className: 'nav-group-label' }, group),
    ...NAV.filter(([, , , itemGroup]) => itemGroup === group).map(([id, label, iconName]) => el('a', {
      className: `nav-link${route === id ? ' active' : ''}`,
      href: `#/${id}`,
      'aria-current': route === id ? 'page' : null,
    }, icon(iconName, 18), el('span', {}, label), id === 'images' ? el('span', { className: 'nav-tag' }, '32²') : null)),
  )))
  requestAnimationFrame(() => nav.querySelector('.active')?.scrollIntoView({ block: 'nearest' }))
}

function setSidebarOpen(open) {
  document.body.classList.toggle('sidebar-open', open)
  menuButton.setAttribute('aria-expanded', String(open))
  menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation')
}

menuButton.addEventListener('click', () => setSidebarOpen(!document.body.classList.contains('sidebar-open')))
closeButton.addEventListener('click', () => setSidebarOpen(false))
scrim.addEventListener('click', () => setSidebarOpen(false))
nav.addEventListener('click', (event) => {
  if (event.target.closest('a')) setSidebarOpen(false)
})
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') setSidebarOpen(false)
})

async function checkBackend() {
  if (backendCheckPending) return
  backendCheckPending = true
  const api = createApiClient()
  const docsLink = document.querySelector('#api-docs-link')
  if (docsLink) docsLink.href = `${api.baseUrl.replace(/\/api\/v1\/?$/, '')}/docs`
  try {
    await api.get('/health/live')
    backendStatus.classList.add('connected')
    backendStatus.classList.remove('disconnected')
    backendStatus.lastChild.textContent = 'Backend connected'
  } catch {
    backendStatus.classList.remove('connected')
    backendStatus.classList.add('disconnected')
    backendStatus.lastChild.textContent = 'Backend unavailable'
  } finally {
    backendCheckPending = false
  }
}

async function render() {
  const token = ++renderToken
  const route = location.hash.replace(/^#\/?/, '').split('/')[0] || 'dashboard'
  const renderPage = RENDERERS[route] || renderDashboard
  const title = NAV.find(([id]) => id === route)?.[1] || 'Overview'
  document.title = `${title} · DeepShield AI`
  document.querySelector('#current-section').textContent = title
  updateNavigation(route in RENDERERS ? route : 'dashboard')
  setSidebarOpen(false)
  if (activeCleanup) { activeCleanup(); activeCleanup = null }
  page.replaceChildren(el('div', { className: 'loading-state' }, el('span', { className: 'spinner' }), 'Loading workspace…'))
  const renderedPage = document.createElement('div')
  try {
    const cleanup = await renderPage(renderedPage, { api: createApiClient(), navigate: (id) => { location.hash = `#/${id}` } })
    if (token !== renderToken) { if (typeof cleanup === 'function') cleanup(); return }
    page.replaceChildren(...renderedPage.childNodes)
    if (typeof cleanup === 'function') activeCleanup = cleanup
  } catch (error) {
    if (token !== renderToken) return
    page.replaceChildren(errorPanel(error, render))
  }
  page.focus({ preventScroll: true })
}

window.addEventListener('hashchange', render)
window.addEventListener('storage', (event) => {
  if (event.key === 'deepshield.apiBase') { checkBackend(); render() }
})
window.addEventListener('focus', checkBackend)
window.setInterval(() => {
  if (document.visibilityState !== 'hidden') checkBackend()
}, BACKEND_HEALTH_INTERVAL_MS)
if (!location.hash) location.hash = '#/dashboard'
checkBackend()
render()
