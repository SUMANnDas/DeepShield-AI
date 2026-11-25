import { el, icon, pageTitle } from '../components.js'

export async function renderAbout(root) {
  root.append(pageTitle('Project information', 'About DeepShield AI', 'A local research workspace with separate image classification, video inspection, and system reporting workflows.'),
    el('section', { className: 'about-grid' },
      el('article', { className: 'about-card' }, el('span', { className: 'card-icon' }, icon('image', 18)), el('h2', {}, 'AI-generated image classification'),
        el('p', {}, 'ResNet50, EfficientNet-B0, and a small Vision Transformer classify the REAL and FAKE labels from CIFAKE v3. CIFAKE images are 32 × 32 pixels. Larger uploads are reduced for exploratory scoring and marked inconclusive because that resolution domain was not validated.'),
        el('p', {}, 'Saved CIFAKE test metrics are benchmark records. The application does not re-evaluate the held-out test set.')),
      el('article', { className: 'about-card' }, el('span', { className: 'card-icon' }, icon('shield', 18)), el('h2', {}, 'Facial manipulation detection'),
        el('p', {}, 'Face swaps, facial reenactment, and manipulated portraits are different tasks from AI-generated image classification. No validated portrait or face-manipulation classifier is registered.')),
      el('article', { className: 'about-card' }, el('span', { className: 'card-icon' }, icon('video', 18)), el('h2', {}, 'Video deepfake detection'),
        el('p', {}, 'The app can inspect video metadata and sample up to six frames. There is no real-video-validated checkpoint, so it does not return a REAL or MANIPULATED video prediction.')),
      el('article', { className: 'about-card' }, el('span', { className: 'card-icon' }, icon('database', 18)), el('h2', {}, 'Local processing and privacy'),
        el('p', {}, 'Uploads are sent only to a local loopback API. Images are processed in memory; videos use a temporary local file removed after processing. History stores metadata, hashes, and model outputs.')),
    ),
    el('div', { className: 'docs-callout' }, el('p', {}, 'Treat model outputs as research indicators that need independent review. They are not proof of authenticity or manipulation.'),
      el('a', { className: 'button button-secondary button-small', href: '#/performance' }, 'Review benchmark records')),
    el('p', { className: 'scope-footer' }, 'DeepShield image scores describe CIFAKE v3 only. AI-generated image classification is not facial deepfake detection.'))
}
