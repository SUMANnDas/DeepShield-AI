import { button, card, el, icon, pageTitle } from '../components.js'
import { DEFAULT_API_BASE, saveApiBase, storedApiBase, createApiClient } from '../api.js'

export async function renderSettings(root, { navigate }) {
  const models = await createApiClient().get('/models')
  const baseInput = el('input', { className: 'text-input', type: 'url', value: storedApiBase(), autocomplete: 'url', spellcheck: false, 'aria-describedby': 'api-base-help' })
  const modelSelect = el('select', { className: 'select-input', id: 'default-model' })
  modelSelect.append(el('option', { value: '' }, 'Ask each time / use first available'))
  for (const model of models.items || []) if (model.status === 'available') modelSelect.append(el('option', { value: model.id }, model.name))
  try { modelSelect.value = localStorage.getItem('deepshield.defaultModel') || '' } catch { /* storage can be disabled */ }
  const error = el('p', { className: 'settings-error', role: 'alert' })
  const saved = el('p', { className: 'help-text', role: 'status' })
  const save = button('Save local settings', { variant: 'primary', iconName: 'check', onClick: () => {
    try {
      saveApiBase(baseInput.value)
      try { localStorage.setItem('deepshield.defaultModel', modelSelect.value) } catch { throw new Error('Browser storage is unavailable. Settings were not saved.') }
      error.textContent = ''; saved.textContent = 'Settings saved in this browser. Restart or reload the local app connection to apply the API URL.'
      window.dispatchEvent(new StorageEvent('storage', { key: 'deepshield.apiBase' }))
    } catch (failure) { error.textContent = failure.message; saved.textContent = '' }
  } })
  const reset = button('Reset defaults', { variant: 'secondary', onClick: () => {
    baseInput.value = DEFAULT_API_BASE; modelSelect.value = ''
    try { localStorage.removeItem('deepshield.apiBase'); localStorage.removeItem('deepshield.defaultModel') } catch { /* no secrets are stored */ }
    error.textContent = ''; saved.textContent = 'Local settings reset.'; window.dispatchEvent(new StorageEvent('storage', { key: 'deepshield.apiBase' }))
  } })
  root.append(pageTitle('Local preferences', 'Settings', 'Configure the loopback API address and your default image model. No credentials are stored in the frontend.'),
    el('section', { className: 'grid grid-2' },
      card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Backend connection'), el('p', {}, 'Uploads are restricted to localhost or 127.0.0.1.'))),
        el('div', { className: 'form-grid' }, el('label', { className: 'field' }, el('span', { className: 'field-label' }, 'Local API base URL'), baseInput, el('small', { id: 'api-base-help' }, 'Default: http://127.0.0.1:8000/api/v1. Remote API hosts are blocked to keep media local.')), error, el('div', { className: 'toolbar' }, save, reset), saved)),
      card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Image model preference'), el('p', {}, 'A default selection is a convenience and does not change model evaluation.'))),
        el('div', { className: 'form-grid' }, el('label', { className: 'field' }, el('span', { className: 'field-label' }, 'Default model'), modelSelect), el('p', { className: 'help-text' }, 'Available models are CIFAKE v3 classifiers. They are not validated for general deepfake detection.'))),
      card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Privacy'), el('p', {}, 'Settings remain in local browser storage.'))), el('p', { className: 'page-description' }, 'Image uploads are held in memory by the API. Video files use a temporary local file that is removed after processing. Analysis history stores metadata, file hashes, and returned outputs; it does not store uploaded media.'), el('div', { className: 'report-actions' }, el('a', { className: 'button button-secondary', href: '#/about' }, icon('info', 15), el('span', {}, 'Review scope and limitations')))),
    ))
  save.querySelector('span').textContent = 'Save local settings'
  reset.querySelector('span').textContent = 'Reset defaults'
  void navigate
}
