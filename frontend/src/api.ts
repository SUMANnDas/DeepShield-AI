import type { Analysis, Model, SystemStatus } from './types'

const base = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}
export const api = {
  health: () => request<{status:string;service:string;version:string}>('/health/live'),
  dashboard: () => request<{counts: Record<string, number>; recent: Analysis[]; models: Model[]; system: SystemStatus}>('/dashboard'),
  models: () => request<{items: Model[]}>('/models'),
  performance: () => request<{dataset: string; scope_warning: string; items: Model[]}>('/performance'),
  system: () => request<SystemStatus>('/system'),
  history: (query = '') => request<{items: Analysis[]; total: number}>(`/history?query=${encodeURIComponent(query)}&limit=500`),
  analyzeImage: (file: File, model: string) => { const data = new FormData(); data.append('file', file); data.append('model_id', model); return request<Analysis>('/analysis/images', {method:'POST', body:data}) },
  analyzeVideo: (file: File) => { const data = new FormData(); data.append('file',file); return request<any>('/analysis/videos',{method:'POST',body:data}) },
  reportUrl: (id: string, format: 'json'|'html') => `${base}/reports/${encodeURIComponent(id)}.${format}`,
}
