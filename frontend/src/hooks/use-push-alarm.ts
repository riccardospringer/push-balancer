import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/api-client'

export function usePushAlarm() {
  return useQuery({
    queryKey: ['pushAlarm'],
    queryFn: ({ signal }) => api.pushAlarm(signal),
    refetchInterval: 30_000,
    staleTime: 25_000,
  })
}

export function useDismissPushAlarm() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.dismissPushAlarm(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['pushAlarm'] }),
  })
}
