import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { TagesplanMode } from '@/types/api'

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 30_000,
  })
}

export function useFeed() {
  return useQuery({
    queryKey: ['feed'],
    queryFn: ({ signal }) => api.feed(signal),
    refetchInterval: 60_000,
  })
}

export function usePushStats() {
  return useQuery({
    queryKey: ['pushStats'],
    queryFn: ({ signal }) => api.pushStats(signal),
    refetchInterval: 60_000,
  })
}

export function useMlStatus() {
  return useQuery({
    queryKey: ['mlStatus'],
    queryFn: ({ signal }) => api.mlStatus(signal),
    staleTime: 120_000,
  })
}

export function useMlMonitoring() {
  return useQuery({
    queryKey: ['mlMonitoring'],
    queryFn: ({ signal }) => api.mlMonitoring(signal),
    refetchInterval: 60_000,
  })
}

export function useMlExperiments() {
  return useQuery({
    queryKey: ['mlExperiments'],
    queryFn: ({ signal }) => api.mlExperiments(signal),
    staleTime: 300_000,
  })
}

export function useGbrtStatus() {
  return useQuery({
    queryKey: ['gbrtStatus'],
    queryFn: ({ signal }) => api.gbrtStatus(signal),
    staleTime: 120_000,
  })
}

export function useTagesplan(date: string, mode: TagesplanMode) {
  return useQuery({
    queryKey: ['tagesplan', date, mode],
    queryFn: ({ signal }) => api.tagesplan(date, mode, signal),
    // Wenn Backend noch rechnet (loading: true), alle 3s pollen; sonst alle 2min
    refetchInterval: (query) => {
      const d = query.state.data
      return d?.loading ? 3_000 : 120_000
    },
    // Cache 55s als frisch gelten lassen — verhindert Spinner-Flash bei Tab-Wechsel
    staleTime: 55_000,
  })
}

export function useTagesplanSuggestions(date: string, mode: TagesplanMode) {
  return useQuery({
    queryKey: ['tagesplanSuggestions', date, mode],
    queryFn: ({ signal }) => api.tagesplanSuggestions(date, mode, signal),
    staleTime: 120_000,
  })
}

export function useCompetitorRedaktion() {
  return useQuery({
    queryKey: ['competitorRedaktion'],
    queryFn: ({ signal }) => api.competitorRedaktion(signal),
    refetchInterval: 120_000,
  })
}

export function useCompetitorSport() {
  return useQuery({
    queryKey: ['competitorSport'],
    queryFn: ({ signal }) => api.competitorSport(signal),
    refetchInterval: 120_000,
  })
}

export function useForschung() {
  return useQuery({
    queryKey: ['forschung'],
    queryFn: ({ signal }) => api.forschung(signal),
    staleTime: 300_000,
  })
}

export function useResearchRules() {
  return useQuery({
    queryKey: ['researchRules'],
    queryFn: ({ signal }) => api.researchRules(signal),
    staleTime: 300_000,
  })
}

export function useAdobeTraffic() {
  return useQuery({
    queryKey: ['adobeTraffic'],
    queryFn: ({ signal }) => api.adobeTraffic(signal),
    refetchInterval: 300_000,
  })
}

export function useLearnings() {
  return useQuery({
    queryKey: ['learnings'],
    queryFn: ({ signal }) => api.learnings(signal),
    staleTime: 300_000,
  })
}

export function useSyncPush() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.syncPush(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['pushStats'] }),
  })
}

export function useTagesplanRetro(mode: TagesplanMode) {
  return useQuery({
    queryKey: ['tagesplanRetro', mode],
    queryFn: ({ signal }) => api.tagesplanRetro(mode, signal),
    staleTime: 300_000,
  })
}

export function useMlRetrain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.mlRetrain(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['mlStatus'] })
      qc.invalidateQueries({ queryKey: ['mlMonitoring'] })
    },
  })
}

export function useGbrtRetrain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.gbrtRetrain(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['gbrtStatus'] }),
  })
}

export function useGbrtPromote() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.gbrtPromote(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['gbrtStatus'] })
      qc.invalidateQueries({ queryKey: ['mlStatus'] })
    },
  })
}

export function useGenerateTitle() {
  return useMutation({
    mutationFn: (body: Parameters<typeof api.generateTitle>[0]) =>
      api.generateTitle(body),
  })
}
