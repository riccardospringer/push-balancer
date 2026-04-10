import { ApiError } from '@/api/generated/api-client-base'

export function getApiErrorMessage(
  error: unknown,
  fallback: string,
): string {
  if (error instanceof ApiError) {
    return (
      error.problem?.detail ??
      error.problem?.title ??
      error.message ??
      fallback
    )
  }

  if (error instanceof Error && error.message) {
    return error.message
  }

  return fallback
}
