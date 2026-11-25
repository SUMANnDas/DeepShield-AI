const API_PREFIX = '/api/v1'

export function resolveDefaultApiBase(location = globalThis.location) {
  if (location) {
    try {
      const current = new URL(location.href || String(location))
      const hostname = current.hostname.replace(/^\[|\]$/g, '').toLowerCase()
      if (['127.0.0.1', 'localhost', '::1'].includes(hostname)) {
        // The Vite development server is frontend-only. Other local ports serve
        // the FastAPI application and should use their own origin.
        const origin = current.port === '5173' ? 'http://127.0.0.1:8000' : current.origin
        return `${origin}${API_PREFIX}`
      }
    } catch {
      // Fall back to the documented local backend URL outside a browser.
    }
  }
  return `http://127.0.0.1:8000${API_PREFIX}`
}

export const DEFAULT_API_BASE = resolveDefaultApiBase()
export const API_BASE_STORAGE_KEY = 'deepshield.apiBase'
export const REQUEST_TIMEOUT_MS = 120_000

export function normalizeApiBase(value) {
  let url
  try {
    url = new URL(String(value).trim())
  } catch {
    throw new Error('Enter a valid local API URL, such as http://127.0.0.1:8000/api/v1.')
  }
  const host = url.hostname.replace(/^\[|\]$/g, '').toLowerCase()
  if (!['http:', 'https:'].includes(url.protocol) || !['127.0.0.1', 'localhost', '::1'].includes(host)) {
    throw new Error('For privacy, DeepShield only sends uploads to a local loopback API address.')
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error('The API URL cannot contain credentials, query parameters, or a fragment.')
  }
  const path = url.pathname.replace(/\/+$/, '')
  url.pathname = path.endsWith('/api/v1') ? path : `${path}/api/v1`
  return url.href.replace(/\/+$/, '')
}

export function storedApiBase(storage = globalThis.localStorage) {
  try {
    return normalizeApiBase(storage?.getItem(API_BASE_STORAGE_KEY) || DEFAULT_API_BASE)
  } catch {
    return DEFAULT_API_BASE
  }
}

export function saveApiBase(value, storage = globalThis.localStorage) {
  const normalized = normalizeApiBase(value)
  storage?.setItem(API_BASE_STORAGE_KEY, normalized)
  return normalized
}

export function createApiClient({ baseUrl = storedApiBase(), fetchImpl = globalThis.fetch, timeoutMs = REQUEST_TIMEOUT_MS } = {}) {
  const base = normalizeApiBase(baseUrl)

  async function request(path, init = {}) {
    const endpoint = `${base}/${String(path).replace(/^\/+/, '')}`
    let response
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), timeoutMs)
    try {
      response = await fetchImpl(endpoint, {
        ...init,
        signal: controller.signal,
        headers: { Accept: 'application/json', ...init.headers },
      })
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw new Error('The local backend request timed out. Check the server and retry.')
      }
      throw new Error('Cannot reach the local DeepShield backend. Start the API and try again.')
    } finally {
      clearTimeout(timeout)
    }
    const contentType = response.headers?.get?.('content-type') || ''
    let payload
    try {
      if (contentType.includes('application/json')) {
        payload = await response.json()
      } else {
        payload = await response.text()
      }
    } catch {
      throw new Error('The local backend returned an unreadable response.')
    }
    if (!response.ok) {
      const detail = payload && typeof payload === 'object' ? payload.detail : payload
      throw new Error(typeof detail === 'string' && detail ? detail : `Backend request failed (${response.status}).`)
    }
    return payload
  }

  return {
    baseUrl: base,
    get: (path) => request(path),
    postForm: (path, body) => request(path, { method: 'POST', body }),
  }
}
