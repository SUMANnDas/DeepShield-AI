import { badge, button, card, el, emptyState, formatBytes, icon, notice, pageTitle } from '../components.js'

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

export async function renderVideoAnalysis(root, { api }) {
  const system = await api.get('/system')
  const fileInput = el('input', { className: 'hidden-input', type: 'file', accept: 'video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.avi,.mov,.mkv,.webm', id: 'video-file' })
  const upload = el('label', { className: 'upload-zone', htmlFor: 'video-file', tabindex: '0' }, icon('upload', 26), el('strong', {}, 'Drop a video or choose a file'), el('p', {}, 'MP4, AVI, MOV, MKV, or WebM · up to 200 MB · up to 10 minutes · up to 16 megapixels per frame'), el('span', { className: 'button button-secondary button-small' }, 'Choose video'), fileInput)
  const preview = el('div', { className: 'hidden' })
  const error = el('p', { className: 'settings-error', role: 'alert' })
  const progress = el('div', { className: 'hidden' }, el('div', { className: 'progress-line' }, el('span')), el('p', { className: 'help-text' }, 'Inspecting metadata and sampling up to six frames locally…'))
  const result = el('div', { className: 'grid' },
    card(emptyState('Inspection report will appear here', 'Choose a supported video to request metadata and bounded frame sampling.')))
  const run = button('Inspect video', { variant: 'primary', iconName: 'video', disabled: true })
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
      const body = new FormData(); body.append('file', selectedFile)
      const record = await api.postForm('/analysis/videos', body)
      const frameCards = (record.sampled_frames || []).map((frame) => {
        const img = el('img', { alt: `Sampled video frame ${frame.frame_index}` })
        img.src = `data:image/jpeg;base64,${frame.preview_jpeg_base64}`
        return el('div', { className: 'frame-card' }, img, el('small', {}, `Frame ${frame.frame_index} · ${Number(frame.timestamp_seconds || 0).toFixed(2)} s`))
      })
      result.replaceChildren(card(
        el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Video inspection complete'), el('p', {}, `Record ${record.id}`)), badge(record.prediction_available ? 'Prediction available' : 'Metadata only', record.prediction_available ? 'green' : 'amber')),
        metadataTiles(record.metadata || {}),
        notice('No video deepfake prediction is available', record.limitation || 'The project has no validated trained video checkpoint or real-video evaluation result.', 'warning'),
        frameCards.length ? el('div', {}, el('div', { className: 'section-heading' }, el('h2', {}, 'Sampled frame previews')), el('div', { className: 'frame-grid' }, frameCards)) : el('p', { className: 'help-text' }, 'No preview frames were returned.'),
        el('div', { className: 'report-actions' }, el('a', { className: 'button button-secondary button-small', href: `#/history` }, 'Open analysis history')),
      ))
    } catch (failure) {
      error.textContent = failure.message
    } finally {
      busy = false; run.disabled = !selectedFile; run.querySelector('span').textContent = 'Inspect video'; progress.classList.add('hidden')
    }
  })
  root.append(pageTitle('Video inspection', 'Video analysis', 'Inspect file metadata and a bounded set of frames. This workflow does not classify video as real or manipulated.'),
    el('details', { className: 'scope-details' }, el('summary', {}, 'What this inspection does'),
      el('div', {}, el('p', {}, `${system.video_prediction_status || 'No validated video checkpoint is registered.'} The backend samples at most six frames with 320-pixel previews, then removes its temporary upload. No real-video validated classifier is available.`))),
    el('section', { className: 'grid grid-2 video-workspace' }, card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Select a video'), el('p', {}, 'Metadata and sampled frames are returned by the local API.'))), upload, error, progress, run, preview), result))
  return () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }
}
