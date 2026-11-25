import { card, el, notice, pageTitle } from '../components.js'

export async function renderAbout(root) {
  root.append(pageTitle('Project information', 'About DeepShield AI', 'A local research prototype for media analysis with clear separation between supported tasks.'),
    notice('Current scope', 'The image models distinguish the REAL and FAKE labels in CIFAKE v3 only. They are not facial deepfake detectors and do not make forensic conclusions.', 'warning'),
    el('section', { className: 'grid grid-2' },
      card(el('h2', {}, 'AI-generated image classification'), el('p', { className: 'page-description' }, 'ResNet50, EfficientNet-B0, and a small Vision Transformer use CIFAKE v3 checkpoints. CIFAKE images are 32×32 pixels. High-resolution uploads are reduced for an exploratory model score and marked inconclusive because that input domain has not been validated.'), el('p', { className: 'page-description' }, 'Saved CIFAKE test metrics are reported as benchmark results only. The current app does not re-evaluate the held-out test set.')),
      card(el('h2', {}, 'Facial manipulation detection'), el('p', { className: 'page-description' }, 'Face swaps, facial reenactment, and manipulated portraits are different tasks. No validated portrait or face-manipulation classifier is registered in this application.')),
      card(el('h2', {}, 'Video deepfake detection'), el('p', { className: 'page-description' }, 'The application can inspect video metadata and sample up to six frames. There is no trained, real-video-validated checkpoint, so no REAL or MANIPULATED video prediction is available.')),
      card(el('h2', {}, 'Local processing and privacy'), el('p', { className: 'page-description' }, 'The UI sends uploads only to a local loopback API address. Image bytes are processed in memory; videos are staged in a temporary file and removed after processing. History records metadata, hashes, and model outputs.')),
    ),
    el('p', { className: 'help-text', style: { marginTop: '16px' } }, 'Treat every model output as an indicator requiring independent review. Do not use it as proof of authenticity or manipulation.'))
}
