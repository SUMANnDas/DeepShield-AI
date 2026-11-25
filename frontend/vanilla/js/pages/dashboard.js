import { badge, card, el, emptyState, formatDate, formatPercent, icon, metricTile, pageTitle, sectionHeading } from '../components.js'

function modelName(model) { return model?.name || model?.id || 'Unknown model' }

export async function renderDashboard(root, { api }) {
  const data = await api.get('/dashboard')
  const counts = data.counts || {}
  const models = data.models || []
  const system = data.system || {}
  const analyses = Number(counts.image || 0) + Number(counts.video || 0)
  const availableModels = models.filter((model) => model.status === 'available').length
  const maxCount = Math.max(Number(counts.image || 0), Number(counts.video || 0), 1)

  root.append(pageTitle('Workspace overview', 'Forensics dashboard', 'A live view of local analysis activity and the tools available in this workspace.',
    el('a', { className: 'button button-primary', href: '#/images' }, icon('image', 16), el('span', {}, 'Analyze an image'))))
  root.append(el('section', { className: 'metric-grid', 'aria-label': 'Workspace summary' },
    metricTile('Total analyses', String(analyses), 'Saved in local history'),
    metricTile('Image analyses', String(counts.image || 0), 'CIFAKE image branch'),
    metricTile('Video analyses', String(counts.video || 0), 'Metadata and sampled frames only'),
    metricTile('Image models', String(availableModels), 'Available local checkpoints'),
  ))

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
  const activityCard = card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Analysis activity'), el('p', {}, 'Saved record totals by media workflow')),
      el('span', { className: 'card-icon' }, icon('chart', 17))),
    analyses ? el('div', { className: 'activity-bars', role: 'img', 'aria-label': `${counts.image || 0} image analyses and ${counts.video || 0} video inspections saved` },
      ...[['Image analysis', Number(counts.image || 0), 'image'], ['Video inspection', Number(counts.video || 0), 'video']].map(([label, count, type]) =>
        el('div', { className: 'activity-row' },
          el('div', { className: 'activity-label' }, el('span', {}, label), el('strong', {}, String(count))),
          el('div', { className: `activity-track activity-${type}` }, el('span', { style: { width: `${Math.max(count ? 5 : 0, count / maxCount * 100)}%` } })),
        )),
      el('p', { className: 'help-text' }, 'Values come from local history. An empty workflow indicates no saved records.'),
    ) : emptyState('No activity to chart', 'Saved analyses will appear here after you run a local workflow.'),
  )
  const systemCard = card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Local services'), el('p', {}, 'Status reported by the connected backend')), badge('Live', 'green')),
    el('div', { className: 'service-list' },
      el('div', {}, el('span', {}, 'Compute'), el('strong', {}, system.cuda_available ? system.gpu_name || 'CUDA available' : 'CPU inference')),
      el('div', {}, el('span', {}, 'Video workflow'), el('strong', {}, system.video_prediction_status || 'Metadata and frame inspection')),
      el('div', {}, el('span', {}, 'History database'), el('strong', {}, system.database_status || 'Status not reported')),
    ),
    el('a', { className: 'text-link', href: '#/system' }, 'View system details →'),
  )
  root.append(el('section', { className: 'grid dashboard-grid' }, modelCard, recentCard))
  root.append(el('section', { className: 'grid grid-2 dashboard-lower' }, activityCard, systemCard))
  root.append(el('details', { className: 'scope-details' }, el('summary', {}, 'Research scope and limitations'),
    el('div', {}, el('p', {}, 'Image classifiers were evaluated on CIFAKE v3 at 32 × 32 pixels. Their scores do not establish general AI image detection or identify manipulated faces. The video workflow inspects metadata and sampled frames; it does not provide a validated video deepfake prediction.'))))
}
