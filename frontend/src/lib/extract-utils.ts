/**
 * Utility functions for Extract module components
 */

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * API error structure returned by the backend
 */
export interface ApiError {
  message?: string
  details?: {
    detail?: string
    [key: string]: unknown
  }
}

/**
 * Field detected during wizard document analysis
 */
export interface WizardField {
  name: string
  type: string
  description?: string
}

/**
 * Analysis results from wizard document processing
 */
export interface WizardAnalysis {
  document_type?: string
  fields?: WizardField[]
}

/**
 * Template suggestion from wizard
 */
export interface WizardTemplate {
  id?: string
  description?: string
  system_prompt?: string
  user_prompt?: string
}

/**
 * Complete wizard result structure
 */
export interface WizardResult {
  analysis?: WizardAnalysis
  template?: WizardTemplate
}

/**
 * Model information from API
 */
export interface ModelResponse {
  id: string
  name?: string
  provider?: string
  owned_by?: string
  capabilities?: string[]
}

/**
 * Generic JSON value type for extracted data
 */
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }

// ============================================================================
// Error Handling Utilities
// ============================================================================

/**
 * Extract a user-friendly error message from various error types
 */
export function getErrorMessage(error: unknown, fallbackMessage: string = "An error occurred"): string {
  if (error instanceof Error) {
    return error.message
  }

  // Handle API error shape
  if (error && typeof error === "object") {
    const apiError = error as ApiError
    if (apiError.details?.detail) {
      return apiError.details.detail
    }
    if (apiError.message) {
      return apiError.message
    }
  }

  return fallbackMessage
}

/**
 * Toast configuration for success and error messages
 */
export interface ToastConfig {
  title: string
  description: string
  variant?: "default" | "destructive"
}

/**
 * Toast function type matching use-toast hook
 */
export type ToastFn = (config: ToastConfig) => void

/**
 * Options for withErrorHandling wrapper
 */
export interface ErrorHandlingOptions {
  /** Toast function for displaying messages */
  toast: ToastFn
  /** Error message title */
  errorTitle?: string
  /** Fallback error description if none can be extracted */
  fallbackErrorMessage?: string
  /** Optional success toast configuration */
  successToast?: {
    title: string
    description: string
  }
}

/**
 * Wraps an async function with standardized error handling and optional success toast
 *
 * @example
 * ```ts
 * const handleSave = withErrorHandling(
 *   async () => {
 *     await api.save(data)
 *   },
 *   {
 *     toast,
 *     errorTitle: "Save Failed",
 *     fallbackErrorMessage: "Could not save the item",
 *     successToast: { title: "Saved", description: "Item saved successfully" }
 *   }
 * )
 * ```
 */
export function withErrorHandling<T>(
  fn: () => Promise<T>,
  options: ErrorHandlingOptions
): () => Promise<T | undefined> {
  return async () => {
    try {
      const result = await fn()
      if (options.successToast) {
        options.toast({
          title: options.successToast.title,
          description: options.successToast.description,
        })
      }
      return result
    } catch (error) {
      options.toast({
        title: options.errorTitle || "Error",
        description: getErrorMessage(error, options.fallbackErrorMessage),
        variant: "destructive",
      })
      return undefined
    }
  }
}

// ============================================================================
// Formatting Utilities
// ============================================================================

/**
 * Format a cost value from cents to dollar string
 */
export function formatCost(cents: number): string {
  return `$${(cents / 100).toFixed(4)}`
}

/**
 * Format any value for display, handling objects, nulls, and primitives
 */
export function formatValue(value: JsonValue): string {
  if (value === null || value === undefined) return "N/A"
  if (typeof value === "object") {
    return JSON.stringify(value, null, 2)
  }
  return String(value)
}

// ============================================================================
// File Validation
// ============================================================================

/** Maximum file size: 10MB */
export const MAX_FILE_SIZE = 10 * 1024 * 1024

/**
 * Validate file size and return error message if invalid
 */
export function validateFileSize(file: File): string | null {
  if (file.size > MAX_FILE_SIZE) {
    return "Maximum file size is 10MB"
  }
  return null
}
