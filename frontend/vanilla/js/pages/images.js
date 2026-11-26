import { badge, button, card, el, emptyState, formatBytes, formatPercent, icon, pageTitle } from '../components.js'

const MAX_BYTES = 10 * 1024 * 1024
const ACCEPTED = new Set(['.jpg', '.jpeg', '.png', '.webp'])

function fileExtension(name) { return name.includes('.') ? `.${name.split('.').pop().toLowerCase()}` : '' }

function probabilityRow(label, value, style = '') {
  const safe = Number.isFinite(Number(value)) ? Math.max(0, Math.min(1, Number(value))) : 0
  return el('div', { className: `probability-row ${style}` },
    el('span', {}, label),
    el('div', { className: 'probability-track', role: 'img', 'aria-label': `${label} ${formatPercent(safe)}` }, el('span', { style: { width: `${safe * 100}%` } })),
    el('strong', {}, formatPercent(value)),
  )
}

function attributionPanel(explainability) {
  if (!explainability || !explainability.overlay_png_base64) return null
  const inputTile = explainability.model_input_png_base64
    ? el('div', { className: 'attribution-tile' },
        el('img', {
          src: `data:image/png;base64,${explainability.model_input_png_base64}`,
          alt: 'Prepared 32 by 32 model input',
        }),
        el('small', {}, 'Prepared 32 × 32 input'),
      )
    : null
  const overlayTile = el('div', { className: 'attribution-tile' },
    el('img', {
      src: `data:image/png;base64,${explainability.overlay_png_base64}`,
      alt: `${explainability.method_label || 'Attribution'} overlay`,
    }),
    el('small', {}, `${explainability.method_label || 'Attribution'} · target ${explainability.target_class || '—'}`),
  )
  return el('div', { className: 'attribution-panel' },
    el('div', { className: 'section-heading' },
      el('h2', {}, 'Visual attribution'),
      el('span', { className: 'badge badge-blue' }, explainability.method === 'vit_patch_gradient_input' ? 'ViT patch gradient' : 'Grad-CAM'),
    ),
    el('div', { className: 'attribution-grid' }, inputTile, overlayTile),
    el('p', { className: 'help-text' }, explainability.limitation || 'Visual attribution is coarse at 32 × 32 and is not causal forensic proof.'),
  )
}

function resultCard(record, api) {
  const result = record.result || {}
  const assessment = result.assessment || { status: 'inconclusive', decision: 'INCONCLUSIVE', raw_model_label: result.predicted_label, reason: 'This result predates the resolution-domain assessment.' }
  const uncertain = assessment.status === 'inconclusive'
  const cardNode = card(
    el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Image assessment'), el('p', {}, 'Response from the selected local CIFAKE model')), badge(uncertain ? 'Inconclusive' : 'Model indication', uncertain ? 'amber' : 'blue')),
    el('div', { className: 'result-label' }, icon(uncertain ? 'alert' : 'shield', 24), el('div', {}, el('strong', {}, assessment.decision || 'INCONCLUSIVE'), el('div', { className: 'help-text' }, `Raw model label: ${assessment.raw_model_label || result.predicted_label || 'Not returned'}`))),
    el('p', { className: 'page-description' }, assessment.reason || 'Model outputs are indicators, not proof.'),
    probabilityRow('REAL score', result.real_probability),
    probabilityRow('FAKE score', result.fake_probability, 'fake'),
    el('p', { className: 'help-text' }, 'Scores are uncalibrated model outputs, not forensic confidence. This checkpoint was trained on CIFAKE v3.'),
    attributionPanel(result.explainability),
    el('dl', { className: 'detail-list' },
      el('div', {}, el('dt', {}, 'Model checkpoint'), el('dd', {}, result.model_identifier || 'Not returned')),
      el('div', {}, el('dt', {}, 'Runtime device'), el('dd', {}, result.device || 'Not returned')),
      el('div', {}, el('dt', {}, 'Inference time'), el('dd', {}, Number.isFinite(Number(result.inference_duration_ms)) ? `${Number(result.inference_duration_ms).toFixed(1)} ms` : 'Not recorded')),
      el('div', {}, el('dt', {}, 'Input size'), el('dd', {}, `${record.metadata?.width ?? '?'} × ${record.metadata?.height ?? '?'}`)),
      el('div', {}, el('dt', {}, 'Analysis ID'), el('dd', {}, record.id || 'Not returned')),
    ),
    el('div', { className: 'report-actions' },
      el('a', { className: 'button button-secondary button-small', href: `${api.baseUrl}/reports/${encodeURIComponent(record.id)}.json`, target: '_blank', rel: 'noreferrer' }, icon('download', 14), el('span', {}, 'JSON report')),
      el('a', { className: 'button button-secondary button-small', href: `${api.baseUrl}/reports/${encodeURIComponent(record.id)}.html`, target: '_blank', rel: 'noreferrer' }, icon('file', 14), el('span', {}, 'Printable report')),
    ),
  )
  return cardNode
}

export async function renderImageAnalysis(root, { api }) {
  const modelsResponse = await api.get('/models')
  const models = (modelsResponse.items || []).filter((item) => item.status === 'available')
  const title = pageTitle('Image forensics', 'Image analysis', 'Upload a local image and request an actual API prediction. The trained models accept a 32×32 representation and are validated on CIFAKE v3 only.')
  const fileInput = el('input', { className: 'hidden-input', type: 'file', accept: 'image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp', id: 'image-file' })
  const uploadLabel = el('label', { className: 'upload-zone', htmlFor: 'image-file', tabindex: '0' },
    icon('upload', 26), el('strong', {}, 'Drop an image here or choose a file'),
    el('p', {}, 'JPEG, PNG or WebP · up to 10 MB · 40 megapixel limit'),
    el('span', { className: 'button button-secondary button-small' }, 'Choose image'), fileInput,
  )
  const selectionError = el('p', { className: 'settings-error', role: 'alert' })
  const previewArea = el('div', { className: 'hidden' })
  const modelSelect = el('select', { className: 'select-input', name: 'model_id', required: true })
  const storedModel = (() => { try { return localStorage.getItem('deepshield.defaultModel') || '' } catch { return '' } })()
  models.forEach((model) => modelSelect.append(el('option', { value: model.id, selected: model.id === storedModel }, `${model.name} · ${model.architecture}`)))
  if (!models.length) modelSelect.append(el('option', { value: '', disabled: true, selected: true }, 'No model is available'))
  const modelContext = el('div', { className: 'model-context' })
  function updateModelContext() {
    const model = models.find((item) => item.id === modelSelect.value)
    modelContext.replaceChildren(model
      ? el('div', {}, el('strong', {}, model.name || model.id), el('span', {}, `${model.architecture || 'Architecture not reported'} · ${model.task || 'CIFAKE image classification'} · CIFAKE v3`))
      : el('span', {}, 'No available image checkpoint was reported by the backend.'))
  }
  updateModelContext()
  modelSelect.addEventListener('change', updateModelContext)
  const submit = button('Analyze image', { variant: 'primary', iconName: 'arrow', type: 'submit', disabled: true })
  const progress = el('div', { className: 'hidden' }, el('div', { className: 'progress-line' }, el('span')), el('p', { className: 'help-text' }, 'Loading the checkpoint and analyzing the image locally…'))
  const resultArea = el('div', { className: 'grid', 'aria-live': 'polite' },
    card(emptyState('Assessment will appear here', 'Choose an image and run the local model to view its response.')))
  const form = el('form', { className: 'form-grid', onSubmit: async (event) => {
    event.preventDefault()
    const file = fileInput.files?.[0]
    if (!file) return
    selectionError.textContent = ''
    progress.classList.remove('hidden')
    submit.disabled = true
    submit.querySelector('span').textContent = 'Analyzing…'
    try {
      const body = new FormData()
      body.append('model_id', modelSelect.value)
      body.append('file', file)
      const response = await api.postForm('/analysis/images', body)
      resultArea.replaceChildren(resultCard(response, api))
    } catch (error) {
      resultArea.replaceChildren(el('div', { className: 'error-panel', role: 'alert' }, icon('alert', 20), el('div', {}, el('h2', {}, 'Image analysis failed'), el('p', {}, error.message))))
    } finally {
      progress.classList.add('hidden')
      submit.disabled = !fileInput.files?.length || !models.length
      submit.querySelector('span').textContent = 'Analyze image'
    }
  } },
    el('div', { className: 'field' }, el('span', { className: 'field-label' }, 'Choose an image model'), modelSelect, modelContext,
      el('small', {}, 'Image class scores only. No face-manipulation or video classifier is available.')),
    uploadLabel,
    selectionError,
    previewArea,
    progress,
    submit,
  )

  let previewUrl = null
  function selectFile(file) {
    selectionError.textContent = ''
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    previewUrl = null
    previewArea.replaceChildren()
    previewArea.classList.add('hidden')
    submit.disabled = true
    if (!file) return
    const extension = fileExtension(file.name)
    if (!ACCEPTED.has(extension)) { selectionError.textContent = 'Choose a JPEG, PNG, or WebP image.'; return }
    if (file.size > MAX_BYTES) { selectionError.textContent = 'The image exceeds the 10 MB limit.'; return }
    if (file.type && !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { selectionError.textContent = 'The selected file content type is not supported.'; return }
    previewUrl = URL.createObjectURL(file)
    const image = el('img', { className: 'preview-image', alt: 'Selected image preview' })
    image.src = previewUrl
    const dimensions = el('span', {}, 'Reading dimensions…')
    image.addEventListener('load', () => { dimensions.textContent = `${image.naturalWidth} × ${image.naturalHeight}` })
    image.addEventListener('error', () => { dimensions.textContent = 'Preview could not be decoded' })
    const remove = button('Remove image', { variant: 'secondary', iconName: 'trash', className: 'button button-secondary button-small', onClick: () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      previewUrl = null
      fileInput.value = ''
      previewArea.replaceChildren()
      previewArea.classList.add('hidden')
      selectionError.textContent = 'Image selection cleared.'
      submit.disabled = true
    } })
    const heading = el('div', { className: 'card-header' },
      el('div', {}, el('h2', {}, file.name), el('p', {}, 'Preview remains in this browser until you submit the form')),
      remove,
    )
    const dimensionValue = el('strong', {}, dimensions)
    const metadata = el('div', { className: 'file-meta' },
      el('span', {}, 'Size: ', el('strong', {}, formatBytes(file.size))),
      el('span', {}, 'Dimensions: ', dimensionValue),
      el('span', {}, 'Format: ', el('strong', {}, file.type || extension.replace('.', '').toUpperCase())),
    )
    previewArea.append(card(heading, image, metadata))
    previewArea.classList.remove('hidden')
    submit.disabled = !models.length
  }

  fileInput.addEventListener('change', () => selectFile(fileInput.files?.[0]))
  uploadLabel.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click() }
  })
  uploadLabel.addEventListener('dragover', (event) => { event.preventDefault(); uploadLabel.classList.add('dragging') })
  uploadLabel.addEventListener('dragleave', () => uploadLabel.classList.remove('dragging'))
  uploadLabel.addEventListener('drop', (event) => {
    event.preventDefault(); uploadLabel.classList.remove('dragging')
    const file = event.dataTransfer?.files?.[0]
    if (!file) return
    const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files
    selectFile(file)
  })

  root.append(title, el('details', { className: 'scope-details' }, el('summary', {}, 'Input resolution and model scope'),
    el('div', {}, el('p', {}, 'CIFAKE images are 32 × 32 pixels. Larger uploads are reduced for an exploratory score and marked inconclusive because the model input domain has only been validated on native CIFAKE samples. This is AI-generated image classification and does not detect facial manipulation or video deepfakes.'))))
  root.append(el('section', { className: 'grid grid-2', style: { marginTop: '15px' } }, card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Select media'), el('p', {}, 'Uploads are processed in memory and are not retained.'))), form), resultArea))
  return () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }
}
