import { badge, card, el, emptyState, formatPercent, metricTile, pageTitle } from '../components.js'

function matrixView(matrix, labels = ['REAL', 'FAKE']) {
  if (!Array.isArray(matrix) || matrix.length !== 2) return el('p', { className: 'help-text' }, 'Confusion matrix was not saved.')
  return el('div', { className: 'matrix', role: 'table', 'aria-label': 'Confusion matrix, rows are true labels and columns are predicted labels' },
    el('div', { className: 'matrix-head' }, 'True / predicted'), el('div', { className: 'matrix-head' }, labels[0]), el('div', { className: 'matrix-head' }, labels[1]),
    el('div', { className: 'matrix-head' }, labels[0]), el('div', { className: 'matrix-value' }, String(matrix[0][0])), el('div', { className: 'matrix-value' }, String(matrix[0][1])),
    el('div', { className: 'matrix-head' }, labels[1]), el('div', { className: 'matrix-value' }, String(matrix[1][0])), el('div', { className: 'matrix-value' }, String(matrix[1][1])),
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
    matrixView(metrics.confusion_matrix, ['REAL', 'FAKE']),
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

function strataSection(strata) {
  const keys = Object.keys(strata || {})
  if (!keys.length) return null
  return el('div', {},
    el('div', { className: 'section-heading' }, el('h2', {}, 'Manipulation method strata'), el('p', {}, 'Per-method detection rate and binary evaluation vs. all REAL test clips')),
    el('div', { className: 'kv-grid' }, keys.map((method) => {
      const entry = strata[method] || {}
      const auc = entry.vs_all_real?.roc_auc
      return el('div', { className: 'kv' },
        el('span', {}, `${method} (${entry.manipulated_count ?? 0} clips)`),
        el('strong', {}, `Detection ${formatPercent(entry.detection_rate)} · AUC ${auc == null ? '—' : Number(auc).toFixed(4)}`),
      )
    })),
  )
}

function videoModelCard(model) {
  const metrics = model.metrics || {}
  return card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, model.name || model.id), el('p', {}, `${model.architecture} · ${model.dataset} · epoch ${model.checkpoint?.epoch ?? '—'}`)), badge('Research only', 'blue')),
    el('div', { className: 'metric-grid grid-4' },
      metricTile('Accuracy', formatPercent(metrics.accuracy)),
      metricTile('Precision', formatPercent(metrics.precision)),
      metricTile('Recall', formatPercent(metrics.recall)),
      metricTile('F1', formatPercent(metrics.f1)),
      metricTile('ROC-AUC', metrics.roc_auc == null ? '—' : formatPercent(metrics.roc_auc, 4)),
      metricTile('PR-AUC', metrics.pr_auc == null ? '—' : formatPercent(metrics.pr_auc, 4)),
      metricTile('Validation ROC-AUC', model.validation_metrics?.roc_auc == null ? '—' : formatPercent(model.validation_metrics.roc_auc, 4)),
      metricTile('Evaluated videos', model.evaluated_videos ? model.evaluated_videos.toLocaleString() : 'Not recorded'),
    ),
    el('div', { className: 'section-heading' }, el('h2', {}, 'Confusion matrix'), el('p', {}, 'Rows=true [REAL, MANIPULATED]; columns=predicted [REAL, MANIPULATED]')),
    matrixView(metrics.confusion_matrix, ['REAL', 'MANIPULATED']),
    strataSection(model.manipulation_strata),
    el('dl', { className: 'detail-list' },
      el('div', {}, el('dt', {}, 'Run identifier'), el('dd', {}, model.run_id || 'Not recorded')),
      el('div', {}, el('dt', {}, 'Task'), el('dd', {}, model.task || 'Not recorded')),
      el('div', {}, el('dt', {}, 'Input'), el('dd', {}, `${(model.input_dimensions || []).join(' × ')} · up to ${model.max_frames ?? 12} frames`)),
      el('div', {}, el('dt', {}, 'Test manifest'), el('dd', {}, model.test_manifest_sha256 || 'Not recorded')),
    ),
    el('details', { className: 'scope-details' }, el('summary', {}, 'Benchmark limitations'),
      el('div', {}, el('p', {}, model.limitations || 'Evaluated on source-grouped test videos only.'))),
  )
}

export async function renderPerformance(root, { api }) {
  const report = await api.get('/performance')
  const models = report.items || []
  const videoModels = report.video_items || []
  root.append(
    pageTitle('Saved evaluation records', 'Model performance', 'Review the recorded benchmark metrics. This view does not re-run evaluation or use the held-out set for model selection.'),
    el('details', { className: 'scope-details' }, el('summary', {}, 'Dataset and task scope'),
      el('div', {}, el('p', {}, report.scope_warning || 'These results distinguish CIFAKE classes only. They do not detect facial manipulation or video deepfakes. CIFAKE source images are 32 × 32 pixels.'))),
    el('div', { className: 'section-heading' }, el('h2', {}, 'Image classification benchmarks'), el('p', {}, `${report.dataset || 'CIFAKE v3'} · threshold 0.5 · FAKE is positive`)),
    models.length ? el('section', { className: 'grid grid-2' }, models.map(modelCard)) : emptyState('No image evaluation records', 'The backend did not return saved benchmark metrics.'),
    el('div', { className: 'section-heading' }, el('h2', {}, 'Video facial manipulation benchmarks'), el('p', {}, report.video_dataset || 'FaceForensics++ c23 source-grouped held-out test split')),
    videoModels.length
      ? el('section', { className: 'grid grid-2' }, videoModels.map(videoModelCard))
      : card(emptyState('No validated video evaluation records yet', 'Complete FaceForensics++ c23 acquisition, grouped splitting, training, and held-out test evaluation to register a video benchmark.')),
  )
}
