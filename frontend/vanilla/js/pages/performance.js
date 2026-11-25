import { badge, card, el, emptyState, formatPercent, metricTile, pageTitle } from '../components.js'

function matrixView(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 2) return el('p', { className: 'help-text' }, 'Confusion matrix was not saved.')
  return el('div', { className: 'matrix', role: 'table', 'aria-label': 'Confusion matrix, rows are true labels and columns are predicted labels' },
    el('div', { className: 'matrix-head' }, 'True / predicted'), el('div', { className: 'matrix-head' }, 'REAL'), el('div', { className: 'matrix-head' }, 'FAKE'),
    el('div', { className: 'matrix-head' }, 'REAL'), el('div', { className: 'matrix-value' }, String(matrix[0][0])), el('div', { className: 'matrix-value' }, String(matrix[0][1])),
    el('div', { className: 'matrix-head' }, 'FAKE'), el('div', { className: 'matrix-value' }, String(matrix[1][0])), el('div', { className: 'matrix-value' }, String(matrix[1][1])),
  )
}

function modelCard(model) {
  const metrics = model.metrics || {}
  const dataset = String(model.dataset || 'CIFAKE')
  const versionLabel = model.dataset_version && !new RegExp(`\\bv${String(model.dataset_version)}$`, 'i').test(dataset)
    ? `${dataset} v${model.dataset_version}`
    : dataset
  return card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, model.name || model.id), el('p', {}, `${model.architecture} · ${versionLabel} · epoch ${model.checkpoint?.epoch ?? '—'}`)), badge(model.status === 'available' ? 'Research only' : 'Unavailable', model.status === 'available' ? 'blue' : 'red')),
    el('div', { className: 'metric-grid grid-4' },
      metricTile('Accuracy', formatPercent(metrics.accuracy)),
      metricTile('Precision · FAKE', formatPercent(metrics.precision_fake)),
      metricTile('Recall · FAKE', formatPercent(metrics.recall_fake)),
      metricTile('F1 · FAKE', formatPercent(metrics.f1_fake)),
      metricTile('ROC-AUC', formatPercent(metrics.roc_auc, 4)),
      metricTile('PR-AUC', metrics.pr_auc == null ? 'Not recorded' : formatPercent(metrics.pr_auc, 4)),
      metricTile('Validation ROC-AUC', formatPercent(model.validation_metrics?.roc_auc, 4)),
      metricTile('Test samples', model.test_samples ? model.test_samples.toLocaleString() : 'Not recorded'),
    ),
    el('div', { className: 'section-heading' }, el('h2', {}, 'Confusion matrix'), el('p', {}, 'Rows=true [REAL, FAKE]; columns=predicted [REAL, FAKE]')),
    matrixView(metrics.confusion_matrix),
    el('dl', { className: 'detail-list' },
      el('div', {}, el('dt', {}, 'Run identifier'), el('dd', {}, model.run_id || 'Not recorded')),
      el('div', {}, el('dt', {}, 'Task'), el('dd', {}, model.task || 'Not recorded')),
      el('div', {}, el('dt', {}, 'Input'), el('dd', {}, (model.input_dimensions || []).join(' × ') || 'Not recorded')),
      el('div', {}, el('dt', {}, 'Test manifest'), el('dd', {}, model.test_manifest_sha256 || 'Not recorded')),
    ),
    el('details', { className: 'scope-details' }, el('summary', {}, 'Benchmark limitations'),
      el('div', {}, el('p', {}, model.limitations || 'CIFAKE v3 metrics describe this benchmark only. They do not establish general AI-generated image detection or facial deepfake performance.'))),
  )
}

export async function renderPerformance(root, { api }) {
  const report = await api.get('/performance')
  const models = report.items || []
  root.append(pageTitle('Saved evaluation records', 'Model performance', 'Review the recorded CIFAKE v3 benchmark metrics. This view does not re-run evaluation or use the held-out set for model selection.'),
    el('details', { className: 'scope-details' }, el('summary', {}, 'Dataset and task scope'),
      el('div', {}, el('p', {}, report.scope_warning || 'These results distinguish CIFAKE classes only. They do not detect facial manipulation or video deepfakes. CIFAKE source images are 32 × 32 pixels.'))),
    el('p', { className: 'help-text', style: { margin: '13px 2px' } }, `Evaluation: ${report.dataset || 'CIFAKE v3'} · threshold 0.5 · FAKE is positive.`),
    models.length ? el('section', { className: 'grid grid-2' }, models.map(modelCard)) : emptyState('No evaluation records', 'The backend did not return saved benchmark metrics.'))
}
