// Simple API utility using native fetch
const API_BASE = import.meta.env.VITE_API_URL || ''

async function request(url, options = {}) {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw { response: { data: error }, message: error.detail || response.statusText }
  }

  return { data: await response.json() }
}

export const api = {
  get: (url) => request(url, { method: 'GET' }),
  post: (url, data) => request(url, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  put: (url, data) => request(url, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  delete: (url) => request(url, { method: 'DELETE' }),
}

