import { badge, button, card, el, emptyState, formatBytes, formatPercent, icon, notice, pageTitle } from '../components.js'

const MAX_BYTES = 200 * 1024 * 1024
const EXTENSIONS = new Set(['.mp4', '.avi', '.mov', '.mkv', '.webm'])

function metadataTiles(metadata) {
  const items = [
    ['Duration', metadata.duration_seconds == null ? 'Not reported' : `${Number(metadata.duration_seconds).toFixed(2)} s`],
    ['Frame rate', metadata.fps == null ? 'Not reported' : `${Number(metadata.fps).toFixed(2)} fps`],
    ['Frame size', metadata.width && metadata.height ? `${metadata.width} × ${metadata.height}` : 'Not reported'],
    ['Decoded frames', metadata.frame_count ?? 'Not reported'],
    ['Codec', metadata.codec || 'Not reported'],
    ['Sampled frames', metadata.sampled_frame_count ?? 0],
  ]
  return el('div', { className: 'kv-grid' }, items.map(([label, value]) =>
    el('div', { className: 'kv' }, el('span', {}, label), el('strong', {}, String(value))),
  ))
}

function probabilityRow(label, value, style = '') {
  const safe = Number.isFinite(Number(value)) ? Math.max(0, Math.min(1, Number(value))) : 0
  return el('div', { className: `probability-row ${style}` },
    el('span', {}, label),
    el('div', { className: 'probability-track', role: 'img', 'aria-label': `${label} ${formatPercent(safe)}` }, el('span', { style: { width: `${safe * 100}%` } })),
    el('strong', {}, formatPercent(value)),
  )
}

function evidenceGrid(evidence) {
  if (!Array.isArray(evidence) || !evidence.length) return null
  return el('div', {},
    el('div', { className: 'section-heading' }, el('h2', {}, 'Frame-level face evidence'), el('p', {}, 'Per-sampled-frame manipulation indicator scores')),
    el('div', { className: 'frame-grid' }, evidence.map((item) => {
      const preview = item.heatmap_jpeg_base64
        ? el('img', { src: `data:image/jpeg;base64,${item.heatmap_jpeg_base64}`, alt: `Face crop frame ${item.frame_index}` })
        : null
      const score = Number(item.probability_manipulated || 0)
      return el('div', { className: 'frame-card' },
        preview,
        el('small', {}, `Frame ${item.frame_index} · ${Number(item.timestamp_seconds || 0).toFixed(2)} s`),
        el('div', { className: 'frame-score-row' },
          el('strong', {}, formatPercent(score)),
          badge(item.suspicious ? 'Elevated' : 'Low', item.suspicious ? 'amber' : 'green'),
        ),
      )
    })),
  )
}

function sampledGrid(frames) {
  if (!Array.isArray(frames) || !frames.length) return el('p', { className: 'help-text' }, 'No preview frames were returned.')
  return el('div', {},
    el('div', { className: 'section-heading' }, el('h2', {}, 'Sampled frame previews')),
    el('div', { className: 'frame-grid' }, frames.map((frame) => {
      const img = el('img', { alt: `Sampled video frame ${frame.frame_index}` })
      img.src = `data:image/jpeg;base64,${frame.preview_jpeg_base64}`
      return el('div', { className: 'frame-card' }, img, el('small', {}, `Frame ${frame.frame_index} · ${Number(frame.timestamp_seconds || 0).toFixed(2)} s`))
    })),
  )
}

function videoResultCard(record, api) {
  const res = record.result || {}
  const predictionAvailable = Boolean(record.prediction_available)
  const inconclusive = record.status === 'inconclusive'
  const headerBadge = predictionAvailable
    ? badge('Model indication', res.prediction === 'MANIPULATED' ? 'amber' : 'blue')
    : badge(inconclusive ? 'Inconclusive' : 'Metadata only', 'amber')

  const assessmentBlock = predictionAvailable
    ? [
        el('div', { className: 'result-label' }, icon(res.prediction === 'MANIPULATED' ? 'alert' : 'shield', 24),
          el('div', {}, el('strong', {}, res.prediction || 'UNKNOWN'), el('div', { className: 'help-text' }, record.limitation || 'Model-based indicator only.'))),
        probabilityRow('REAL score', res.probability_real),
        probabilityRow('MANIPULATED score', res.probability_manipulated, 'fake'),
        el('dl', { className: 'detail-list' },
          el('div', {}, el('dt', {}, 'Model checkpoint'), el('dd', {}, res.model_identifier || 'Not returned')),
          el('div', {}, el('dt', {}, 'Runtime device'), el('dd', {}, res.device || 'Not returned')),
          el('div', {}, el('dt', {}, 'Inference time'), el('dd', {}, Number.isFinite(Number(res.inference_duration_ms)) ? `${Number(res.inference_duration_ms).toFixed(1)} ms` : 'Not recorded')),
          el('div', {}, el('dt', {}, 'Analysis ID'), el('dd', {}, record.id || 'Not returned')),
        ),
      ]
    : [
        notice(
          inconclusive ? 'Video prediction inconclusive' : 'No video deepfake prediction is available',
          record.limitation || 'The project has no validated trained video checkpoint or real-video evaluation result.',
          'warning',
        ),
      ]

  return card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, predictionAvailable ? 'Video analysis complete' : 'Video inspection complete'), el('p', {}, `Record ${record.id}`)), headerBadge),
    ...assessmentBlock,
    metadataTiles(record.metadata || {}),
    evidenceGrid(res.frame_evidence),
    sampledGrid(record.sampled_frames),
    el('div', { className: 'report-actions' },
      el('a', { className: 'button button-secondary button-small', href: `${api.baseUrl}/reports/${encodeURIComponent(record.id)}.json`, target: '_blank', rel: 'noreferrer' }, icon('download', 14), el('span', {}, 'JSON report')),
      el('a', { className: 'button button-secondary button-small', href: `${api.baseUrl}/reports/${encodeURIComponent(record.id)}.html`, target: '_blank', rel: 'noreferrer' }, icon('file', 14), el('span', {}, 'Printable report')),
      el('a', { className: 'button button-secondary button-small', href: '#/history' }, 'Open analysis history'),
    ),
  )
}

export async function renderVideoAnalysis(root, { api }) {
  const [system, modelsPayload] = await Promise.all([api.get('/system'), api.get('/models')])
  const videoModels = (modelsPayload.video_items || []).filter((item) => item.status === 'available')
  const hasModel = videoModels.length > 0

  const modelSelect = el('select', { className: 'select-input', name: 'model_id' })
  videoModels.forEach((model) => modelSelect.append(el('option', { value: model.id }, `${model.name} · ${model.dataset || 'FF++ c23'}`)))
  const modelContext = el('div', { className: 'model-context' })
  function updateModelContext() {
    const selected = videoModels.find((item) => item.id === modelSelect.value)
    modelContext.replaceChildren(selected
      ? el('div', {}, el('strong', {}, selected.name || selected.id), el('span', {}, `${selected.architecture} · epoch ${selected.checkpoint?.epoch ?? '—'} · ${selected.preprocessing}`))
      : el('span', {}, 'No validated video checkpoint is currently available.'))
  }
  if (hasModel) {
    updateModelContext()
    modelSelect.addEventListener('change', updateModelContext)
  }

  const fileInput = el('input', { className: 'hidden-input', type: 'file', accept: 'video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.avi,.mov,.mkv,.webm', id: 'video-file' })
  const upload = el('label', { className: 'upload-zone', htmlFor: 'video-file', tabindex: '0' }, icon('upload', 26), el('strong', {}, 'Drop a video or choose a file'), el('p', {}, 'MP4, AVI, MOV, MKV, or WebM · up to 200 MB · up to 10 minutes · up to 16 megapixels per frame'), el('span', { className: 'button button-secondary button-small' }, 'Choose video'), fileInput)
  const preview = el('div', { className: 'hidden' })
  const error = el('p', { className: 'settings-error', role: 'alert' })
  const progressText = hasModel
    ? 'Inspecting metadata, sampling frames, and running local video inference…'
    : 'Inspecting metadata and sampling up to six frames locally…'
  const progress = el('div', { className: 'hidden' }, el('div', { className: 'progress-line' }, el('span')), el('p', { className: 'help-text' }, progressText))
  const result = el('div', { className: 'grid' },
    card(emptyState(
      hasModel ? 'Video assessment will appear here' : 'Inspection report will appear here',
      hasModel ? 'Choose a supported video and run the local validated video detector.' : 'Choose a supported video to request metadata and bounded frame sampling.',
    )))
  const actionLabel = hasModel ? 'Analyze video' : 'Inspect video'
  const run = button(actionLabel, { variant: 'primary', iconName: 'video', disabled: true })
  let selectedFile = null
  let previewUrl = null
  let busy = false
  function choose(file) {
    error.textContent = ''; result.replaceChildren()
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    previewUrl = null; selectedFile = null; preview.replaceChildren(); preview.classList.add('hidden'); run.disabled = true
    if (!file) return
    const extension = file.name.includes('.') ? `.${file.name.split('.').pop().toLowerCase()}` : ''
    if (!EXTENSIONS.has(extension)) { error.textContent = 'Choose an MP4, AVI, MOV, MKV, or WebM file.'; return }
    if (file.size > MAX_BYTES) { error.textContent = 'The video exceeds the 200 MB limit.'; return }
    selectedFile = file; previewUrl = URL.createObjectURL(file)
    const video = el('video', { className: 'preview-video', controls: true, preload: 'metadata' }); video.src = previewUrl
    const decoded = el('p', { className: 'help-text' }, 'Reading video metadata…')
    video.addEventListener('loadedmetadata', () => {
      decoded.textContent = `${Number.isFinite(video.duration) ? `${video.duration.toFixed(2)} seconds` : 'Duration unavailable'} · ${video.videoWidth || '?'} × ${video.videoHeight || '?'} pixels`
    })
    video.addEventListener('error', () => { decoded.textContent = 'Browser preview could not decode this file; the backend will check its format during inspection.' })
    const remove = button('Remove video', { variant: 'secondary', iconName: 'trash', className: 'button button-secondary button-small', onClick: () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      previewUrl = null; selectedFile = null; fileInput.value = ''
      preview.replaceChildren(); preview.classList.add('hidden'); result.replaceChildren(); run.disabled = true
      error.textContent = 'Video selection cleared.'
    } })
    preview.append(card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, file.name), el('p', {}, `${formatBytes(file.size)} · local preview`)), remove), video, decoded))
    preview.classList.remove('hidden'); run.disabled = false
  }
  fileInput.addEventListener('change', () => choose(fileInput.files?.[0]))
  upload.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click() }
  })
  upload.addEventListener('dragover', (event) => { event.preventDefault(); upload.classList.add('dragging') })
  upload.addEventListener('dragleave', () => upload.classList.remove('dragging'))
  upload.addEventListener('drop', (event) => {
    event.preventDefault(); upload.classList.remove('dragging')
    const file = event.dataTransfer?.files?.[0]
    if (!file) return
    const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files
    choose(file)
  })
  run.addEventListener('click', async () => {
    if (!selectedFile || busy) return
    busy = true; run.disabled = true; run.querySelector('span').textContent = 'Processing…'; progress.classList.remove('hidden'); error.textContent = ''
    try {
      const body = new FormData()
      body.append('file', selectedFile)
      if (hasModel && modelSelect.value) body.append('model_id', modelSelect.value)
      const record = await api.postForm('/analysis/videos', body)
      result.replaceChildren(videoResultCard(record, api))
    } catch (failure) {
      error.textContent = failure.message
    } finally {
      busy = false; run.disabled = !selectedFile; run.querySelector('span').textContent = actionLabel; progress.classList.add('hidden')
    }
  })

  const modelField = hasModel
    ? el('div', { className: 'field' }, el('span', { className: 'field-label' }, 'Choose a video model'), modelSelect, modelContext)
    : null

  root.append(
    pageTitle(
      hasModel ? 'Video forensics' : 'Video inspection',
      'Video analysis',
      hasModel
        ? 'Inspect metadata, sample bounded face tracks, and run the registered local facial manipulation model.'
        : 'Inspect file metadata and a bounded set of frames. This workflow does not classify video as real or manipulated until a validated video model is registered.',
    ),
    el('details', { className: 'scope-details' }, el('summary', {}, 'What this workflow does'),
      el('div', {}, el('p', {}, hasModel
        ? 'The backend validates the temporary upload inside the local storage budget, samples uniformly spaced frames, tracks the dominant face with the local fallback detector, and reports model-based facial manipulation indicators before removing the temporary file.'
        : `${system.video_prediction_status || 'No validated video checkpoint is registered.'} The backend samples at most six frames with 320-pixel previews, then removes its temporary upload. No real-video validated classifier is available.`))),
    el('section', { className: 'grid grid-2 video-workspace' },
      card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Select a video'), el('p', {}, 'Temporary uploads are deleted immediately after processing.'))), modelField, upload, error, progress, run, preview),
      result),
  )
  return () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }
}
