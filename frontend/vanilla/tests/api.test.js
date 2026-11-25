import { test } from 'vitest'
import assert from 'node:assert/strict'
import { createApiClient, normalizeApiBase, saveApiBase, storedApiBase, DEFAULT_API_BASE } from '../js/api.js'

test('normalizes a loopback API URL and appends the configured API prefix', () => {
  assert.equal(normalizeApiBase('http://127.0.0.1:8123/'), 'http://127.0.0.1:8123/api/v1')
  assert.equal(normalizeApiBase('http://localhost:8000/api/v1/'), 'http://localhost:8000/api/v1')
})

test('blocks remote API hosts and URLs with embedded credentials', () => {
  assert.throws(() => normalizeApiBase('https://example.com/api/v1'), /loopback/)
  assert.throws(() => normalizeApiBase('http://user:secret@127.0.0.1:8000/api/v1'), /credentials/)
})

test('stores only the normalized local base URL', () => {
  const values = new Map()
  const storage = { getItem: (key) => values.get(key) || null, setItem: (key, value) => values.set(key, value) }
  assert.equal(saveApiBase('http://127.0.0.1:9000', storage), 'http://127.0.0.1:9000/api/v1')
  assert.equal(storedApiBase(storage), 'http://127.0.0.1:9000/api/v1')
})

test('sends multipart requests to the local backend and parses error details', async () => {
  const calls = []
  const fetchImpl = async (url, init) => {
    calls.push({ url, init })
    return { ok: true, headers: { get: () => 'application/json' }, json: async () => ({ accepted: true }) }
  }
  const api = createApiClient({ baseUrl: DEFAULT_API_BASE, fetchImpl })
  const form = new FormData()
  form.append('sample', 'value')
  assert.deepEqual(await api.postForm('/analysis/images', form), { accepted: true })
  assert.equal(calls[0].url, `${DEFAULT_API_BASE}/analysis/images`)
  assert.equal(calls[0].init.body, form)
  assert.equal(calls[0].init.headers.Accept, 'application/json')

  const broken = createApiClient({ baseUrl: DEFAULT_API_BASE, fetchImpl: async () => ({ ok: false, status: 415, headers: { get: () => 'application/json' }, json: async () => ({ detail: 'Unsupported image' }) }) })
  await assert.rejects(broken.postForm('/analysis/images', form), /Unsupported image/)
})
