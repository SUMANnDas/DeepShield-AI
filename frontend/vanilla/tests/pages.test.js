import { test } from 'vitest'
import assert from 'node:assert/strict'
import { renderImageAnalysis } from '../js/pages/images.js'
import { renderVideoAnalysis } from '../js/pages/videos.js'
import { renderPerformance } from '../js/pages/performance.js'

test('image analysis renders live Grad-CAM visual attribution panel when returned', async () => {
  const root = document.createElement('div')
  const api = {
    baseUrl: 'http://127.0.0.1:8000/api/v1',
    get: async (path) => {
      assert.equal(path, '/models')
      return {
        items: [{ id: 'resnet50', name: 'ResNet50', architecture: 'resnet50', status: 'available', task: 'AI-generated image classification' }],
      }
    },
    postForm: async () => ({
      id: 'analysis-1',
      result: {
        predicted_label: 'FAKE',
        real_probability: 0.1,
        fake_probability: 0.9,
        model_identifier: 'resnet50:run:epoch-16',
        device: 'cuda:0',
        inference_duration_ms: 4.2,
        assessment: { status: 'model_indication', decision: 'FAKE', raw_model_label: 'FAKE', reason: 'Native 32x32' },
        explainability: {
          method: 'gradcam',
          method_label: 'Grad-CAM (final convolutional block)',
          target_class: 'FAKE',
          model_input_png_base64: 'AAAA',
          overlay_png_base64: 'BBBB',
          limitation: 'Coarse 32x32 visual attribution.',
        },
      },
      metadata: { width: 32, height: 32 },
    }),
  }

  await renderImageAnalysis(root, { api })
  const input = root.querySelector('#image-file')
  const file = new File([new Uint8Array([1, 2, 3])], 'sample.png', { type: 'image/png' })
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  input.dispatchEvent(new Event('change'))
  root.querySelector('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
  await new Promise((resolve) => setTimeout(resolve, 20))

  assert.match(root.textContent, /Visual attribution/)
  assert.match(root.textContent, /Grad-CAM/)
  assert.equal(root.querySelectorAll('.attribution-tile img').length, 2)
})

test('video analysis stays metadata-only when no validated video checkpoint is registered', async () => {
  const root = document.createElement('div')
  const api = {
    baseUrl: 'http://127.0.0.1:8000/api/v1',
    get: async (path) => {
      if (path === '/system') return { video_prediction_status: 'unavailable_no_validated_video_checkpoint' }
      if (path === '/models') return { items: [], video_items: [] }
      throw new Error(`Unexpected GET ${path}`)
    },
  }

  await renderVideoAnalysis(root, { api })
  assert.equal(root.querySelector('select[name="model_id"]'), null)
  assert.match(root.textContent, /Inspect video/)
})

test('video analysis enables model selection and evidence view when a validated checkpoint exists', async () => {
  const root = document.createElement('div')
  const api = {
    baseUrl: 'http://127.0.0.1:8000/api/v1',
    get: async (path) => {
      if (path === '/system') return { video_prediction_status: 'available' }
      if (path === '/models') {
        return {
          items: [],
          video_items: [{
            id: 'eff-bigru-run-1',
            name: 'EfficientNet-B0 + BiGRU (Attention)',
            architecture: 'efficientnet-b0-bigru-attention',
            dataset: 'FaceForensics++ c23',
            status: 'available',
            checkpoint: { epoch: 8, size_bytes: 1024 },
            preprocessing: 'Uniform up to 12 frames',
          }],
        }
      }
      throw new Error(`Unexpected GET ${path}`)
    },
    postForm: async () => ({
      id: 'vid-1',
      status: 'complete',
      prediction_available: true,
      limitation: 'Model-based indicator only.',
      metadata: { duration_seconds: 2.0, fps: 25, width: 640, height: 480, frame_count: 50, codec: 'h264', sampled_frame_count: 1 },
      sampled_frames: [{ frame_index: 0, timestamp_seconds: 0.0, preview_jpeg_base64: 'CCC' }],
      result: {
        prediction: 'MANIPULATED',
        probability_real: 0.15,
        probability_manipulated: 0.85,
        model_identifier: 'efficientnet-b0-bigru-attention:eff-bigru-run-1:epoch-8',
        device: 'cuda:0',
        inference_duration_ms: 42.5,
        frame_evidence: [{ frame_index: 0, timestamp_seconds: 0.0, probability_manipulated: 0.85, suspicious: true, heatmap_jpeg_base64: 'DDD' }],
      },
    }),
  }

  await renderVideoAnalysis(root, { api })
  const select = root.querySelector('select[name="model_id"]')
  assert.ok(select)
  assert.equal(select.value, 'eff-bigru-run-1')
  const input = root.querySelector('#video-file')
  const file = new File([new Uint8Array([1, 2, 3])], 'clip.mp4', { type: 'video/mp4' })
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  input.dispatchEvent(new Event('change'))
  root.querySelector('button.button-primary').click()
  await new Promise((resolve) => setTimeout(resolve, 20))

  assert.match(root.textContent, /MANIPULATED/)
  assert.match(root.textContent, /Frame-level face evidence/)
})

test('performance page displays both image and video benchmarks with manipulation strata', async () => {
  const root = document.createElement('div')
  const api = {
    get: async () => ({
      dataset: 'CIFAKE v3',
      items: [],
      video_dataset: 'FaceForensics++ c23',
      video_items: [{
        id: 'vid-run',
        name: 'EfficientNet-B0 + BiGRU (Attention)',
        architecture: 'efficientnet-b0-bigru-attention',
        dataset: 'FaceForensics++ c23',
        status: 'available',
        checkpoint: { epoch: 8 },
        validation_metrics: { roc_auc: 0.94 },
        evaluated_videos: 140,
        input_dimensions: [224, 224],
        max_frames: 12,
        metrics: { accuracy: 0.92, precision: 0.91, recall: 0.93, f1: 0.92, roc_auc: 0.96, pr_auc: 0.95, confusion_matrix: [[65, 5], [5, 65]] },
        manipulation_strata: { Deepfakes: { manipulated_count: 35, detection_rate: 0.94, vs_all_real: { roc_auc: 0.97 } } },
      }],
    }),
  }

  await renderPerformance(root, { api })
  assert.match(root.textContent, /Video facial manipulation benchmarks/)
  assert.match(root.textContent, /Manipulation method strata/)
  assert.match(root.textContent, /Deepfakes/)
})
