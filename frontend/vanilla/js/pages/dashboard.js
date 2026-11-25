import { badge, card, el, emptyState, formatDate, formatPercent, icon, metricTile, notice, pageTitle } from '../components.js'

function modelName(model) { return model?.name || model?.id || 'Unknown model' }

export async function renderDashboard(root, { api }) {
  const data = await api.get('/dashboard')
  const counts = data.counts || {}
  const models = data.models || []
  const system = data.system || {}
  const analyses = Number(counts.image || 0) + Number(counts.video || 0)

  root.append(pageTitle('Overview', 'Forensics dashboard', 'Local analysis activity, available research models, and system readiness.',
    el('a', { className: 'button button-primary', href: '#/images' }, icon('image', 16), el('span', {}, 'Analyze an image'))))
  root.append(el('section', { className: 'metric-grid', 'aria-label': 'Workspace summary' },
    metricTile('Total analyses', String(analyses), 'Saved in local history'),
    metricTile('Image analyses', String(counts.image || 0), 'CIFAKE image branch'),
    metricTile('Video analyses', String(counts.video || 0), 'Metadata and sampled frames only'),
    metricTile('CUDA', system.cuda_available ? 'Available' : 'Unavailable', system.gpu_name || 'CPU inference available'),
  ))
  root.append(notice('Research indicators only', 'Image models were evaluated on CIFAKE v3 at 32×32. They are not validated for unfamiliar generators, real-world authenticity decisions, facial manipulation, or video deepfakes.', 'warning'))

  const modelCard = card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Available image models'), el('p', {}, `${models.filter((m) => m.status === 'available').length} of ${models.length} saved checkpoints are available`)), el('span', { className: 'card-icon' }, icon('shield', 17))),
    models.length ? el('div', { className: 'model-list' }, models.map((model) => el('div', { className: 'model-row' },
      el('span', { className: 'model-symbol' }, icon('image', 17)),
      el('div', { className: 'model-main' }, el('strong', {}, modelName(model)), el('small', {}, `${model.architecture || model.id} · epoch ${model.checkpoint?.epoch ?? '—'} · ${model.dataset || 'Dataset not recorded'}`)),
      el('div', { className: 'model-stat' }, el('strong', {}, formatPercent(model.metrics?.accuracy)), el('small', {}, 'CIFAKE test accuracy')),
      badge(model.status === 'available' ? 'Research ready' : 'Unavailable', model.status === 'available' ? 'green' : 'red'),
    ))) : emptyState('No models registered', 'The backend did not report any available image models.'),
    el('a', { className: 'text-link', href: '#/performance' }, 'View benchmark details →'),
  )

  const recentItems = Array.isArray(data.recent) ? data.recent : []
  const recentCard = card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Recent activity'), el('p', {}, 'Latest local analysis records')), el('a', { className: 'text-link', href: '#/history' }, 'View history →')),
    recentItems.length ? recentItems.map((item) => el('div', { className: 'recent-row' },
      el('div', {}, el('strong', {}, item.filename || 'Untitled file'), el('small', {}, `${formatDate(item.created_at)} · ${item.media_type || 'media'}`)),
      badge(item.status || 'complete', item.media_type === 'video' ? 'amber' : 'blue'),
    )) : emptyState('No analyses yet', 'Run a local image or video workflow to create the first history record.', el('a', { className: 'button button-secondary', href: '#/images' }, 'Open image analysis')),
  )
  root.append(el('section', { className: 'grid dashboard-grid' }, modelCard, recentCard))
}
