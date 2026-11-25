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
  ['dashboard', 'Dashboard', 'dashboard'],
  ['images', 'Image analysis', 'image'],
  ['videos', 'Video analysis', 'video'],
  ['history', 'Analysis history', 'history'],
  ['performance', 'Model performance', 'chart'],
  ['system', 'System health', 'system'],
  ['settings', 'Settings', 'settings'],
  ['about', 'About DeepShield', 'info'],
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

const sidebar = el('aside', { className: 'sidebar' },
  el('a', { className: 'brand', href: '#/dashboard', 'aria-label': 'DeepShield AI dashboard' },
    el('span', { className: 'brand-mark' }, icon('shield', 22)),
    el('span', {}, el('strong', {}, 'DeepShield'), el('small', {}, 'MEDIA FORENSICS')),
  ),
  el('div', { className: 'sidebar-kicker' }, el('span', { className: 'status-dot' }), 'LOCAL RESEARCH WORKSPACE'),
  nav,
  el('div', { className: 'sidebar-caution' }, icon('alert', 17), el('span', {}, 'Model outputs are research indicators, not proof.')),
  el('div', { className: 'workspace-chip' }, el('span', { className: 'workspace-avatar' }, 'DS'), el('span', {}, el('strong', {}, 'Local session'), el('small', {}, 'Private workspace'))),
)

const frame = el('div', { className: 'app-frame' },
  sidebar,
  el('section', { className: 'workspace' },
    el('header', { className: 'topbar' },
      el('div', { className: 'breadcrumbs' }, 'DeepShield AI', el('span', {}, '›'), el('span', { id: 'current-section' }, 'Dashboard')),
      el('div', { className: 'topbar-actions' }, backendStatus, el('a', { id: 'api-docs-link', className: 'api-docs-link', href: 'http://127.0.0.1:8000/docs', target: '_blank', rel: 'noreferrer' }, 'API docs')),
    ),
    page,
  ),
)
host.replaceChildren(frame)

function updateNavigation(route) {
  nav.replaceChildren(...NAV.map(([id, label, iconName]) => el('a', {
    className: `nav-link${route === id ? ' active' : ''}`,
    href: `#/${id}`,
    'aria-current': route === id ? 'page' : null,
  }, icon(iconName, 17), el('span', {}, label), id === 'images' ? el('span', { className: 'nav-tag' }, '32×32') : null)))
}

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
  const title = NAV.find(([id]) => id === route)?.[1] || 'Dashboard'
  document.title = `${title} · DeepShield AI`
  document.querySelector('#current-section').textContent = title
  updateNavigation(route in RENDERERS ? route : 'dashboard')
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
