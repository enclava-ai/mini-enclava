/**
 * URL utilities for handling HTTP/HTTPS protocol detection
 */

/**
 * Get the base URL with proper protocol detection
 * This ensures API calls use the same protocol as the page was loaded with
 *
 * IMPORTANT: NEXT_PUBLIC_BASE_URL must be set in production environments.
 * Server-side rendering without this variable will result in incorrect URLs.
 */
export const getBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    // Client-side: detect current protocol
    const protocol = window.location.protocol === 'https:' ? 'https' : 'http'
    const host = process.env.NEXT_PUBLIC_BASE_URL || window.location.host // Use host (includes port) instead of hostname
    return `${protocol}://${host}`
  }

  // Server-side: require NEXT_PUBLIC_BASE_URL in production
  const baseUrl = process.env.NEXT_PUBLIC_BASE_URL
  if (!baseUrl) {
    if (process.env.NODE_ENV === 'production') {
      console.warn('NEXT_PUBLIC_BASE_URL not set in production - URLs may be incorrect')
    }
    // Development fallback only
    return 'http://localhost'
  }

  const protocol = process.env.NODE_ENV === 'production' ? 'https' : 'http'
  return `${protocol}://${baseUrl}`
}

/**
 * Get the API URL with proper protocol detection
 * This is the main function that should be used for all API calls
 *
 * IMPORTANT: NEXT_PUBLIC_BASE_URL must be set in production environments.
 */
export const getApiUrl = (): string => {
  if (typeof window !== 'undefined') {
    // Client-side: use the same protocol as the current page
    const protocol = window.location.protocol.slice(0, -1) // Remove ':' from 'https:'
    const host = window.location.host // Use host (includes port) instead of hostname
    return `${protocol}://${host}`
  }

  // Server-side: require NEXT_PUBLIC_BASE_URL in production
  const baseUrl = process.env.NEXT_PUBLIC_BASE_URL
  if (!baseUrl) {
    if (process.env.NODE_ENV === 'production') {
      console.warn('NEXT_PUBLIC_BASE_URL not set in production - API URLs may be incorrect')
    }
    // Development fallback only
    return 'http://localhost'
  }

  return `http://${baseUrl}`
}

/**
 * Get the internal API URL for authenticated endpoints
 * This ensures internal API calls use the same protocol as the page
 */
export const getInternalApiUrl = (): string => {
  const baseUrl = getApiUrl()
  return `${baseUrl}/api-internal`
}

/**
 * Get the public API URL for external client endpoints
 * This ensures public API calls use the same protocol as the page
 */
export const getPublicApiUrl = (): string => {
  const baseUrl = getApiUrl()
  return `${baseUrl}/api`
}

/**
 * Helper function to make API calls with proper protocol
 */
export const apiFetch = async (
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> => {
  const baseUrl = getApiUrl()
  const url = `${baseUrl}${endpoint}`

  return fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
}

/**
 * Helper function for internal API calls
 */
export const internalApiFetch = async (
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> => {
  const url = `${getInternalApiUrl()}${endpoint}`

  return fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
}

/**
 * Helper function for public API calls
 */
export const publicApiFetch = async (
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> => {
  const url = `${getPublicApiUrl()}${endpoint}`

  return fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
}