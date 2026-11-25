import { card, el, formatBytes, icon, pageTitle } from '../components.js'

function kv(label, value) { return el('div', { className: 'kv' }, el('span', {}, label), el('strong', {}, value == null || value === '' ? 'Not reported' : String(value))) }

export async function renderSystem(root, { api }) {
  const data = await api.get('/system')
  const gpu = data.cuda_available ? data.gpu_name || 'CUDA available' : 'CUDA unavailable'
  const disk = data.disk || {}
  root.append(pageTitle('Runtime status', 'System health', 'Live software, accelerator, memory, disk, and model-availability information from the local backend.'),
    el('section', { className: 'grid grid-2' },
      card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Backend runtime'), el('p', {}, 'Reported by the running API process')), el('span', { className: 'card-icon' }, icon('system', 17))),
        el('div', { className: 'kv-grid' }, kv('Backend', 'Connected'), kv('Python', data.python_version), kv('PyTorch', data.pytorch_version), kv('Torchvision', data.torchvision_version), kv('Operating system', data.platform), kv('CPU', data.cpu_name), kv('Logical CPUs', data.logical_cpu_count), kv('Database', data.database_status))),
      card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'GPU and memory'), el('p', {}, 'Live device values where the runtime supports them')), el('span', { className: 'card-icon' }, icon('shield', 17))),
        el('div', { className: 'kv-grid' }, kv('CUDA', data.cuda_available ? 'Available' : 'Unavailable'), kv('CUDA runtime', data.cuda_version), kv('GPU', gpu), kv('VRAM total', formatBytes(data.gpu_memory_total_bytes)), kv('VRAM free', formatBytes(data.gpu_memory_free_bytes)), kv('VRAM allocated', formatBytes(data.gpu_memory_allocated_bytes)), kv('VRAM reserved', formatBytes(data.gpu_memory_reserved_bytes)), kv('RAM available', formatBytes(data.ram_available_bytes)), kv('RAM total', formatBytes(data.ram_total_bytes)))),
      card(el('div', { className: 'card-header' }, el('div', {}, el('h2', {}, 'Storage and model services'), el('p', {}, 'Disk usage is measured for the project volume')), el('span', { className: 'card-icon' }, icon('file', 17))),
        el('div', { className: 'kv-grid' }, kv('Project volume free', formatBytes(disk.project_root_free_bytes)), kv('Project volume total', formatBytes(disk.total_bytes)), kv('Image models', `${data.image_models_available ?? 0} available`), kv('Video prediction', data.video_prediction_status || 'Unavailable'))),
    ))
}
