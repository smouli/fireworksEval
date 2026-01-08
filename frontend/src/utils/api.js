// Simple API utility using native fetch
const API_BASE = import.meta.env.VITE_API_URL || ''

async function request(url, options = {}) {
  const fullUrl = `${API_BASE}${url}`
  
  try {
    const response = await fetch(fullUrl, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }))
      console.error(`API Error (${response.status}):`, fullUrl, error)
      throw { response: { data: error }, message: error.detail || response.statusText }
    }

    return { data: await response.json() }
  } catch (error) {
    console.error('API Request failed:', fullUrl, error)
    throw error
  }
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

