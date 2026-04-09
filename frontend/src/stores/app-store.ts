import { create } from 'zustand'
import type { TagesplanMode } from '@/types/api'

interface AppState {
  // Kandidaten filters
  kandidatenSearch: string
  kandidatenCategory: string
  kandidatenChannel: string
  setKandidatenSearch: (v: string) => void
  setKandidatenCategory: (v: string) => void
  setKandidatenChannel: (v: string) => void

  // Live Pushes filters
  liveDays: number
  liveChannel: string
  setLiveDays: (v: number) => void
  setLiveChannel: (v: string) => void

  // Konkurrenz tab
  konkurrenzMode: 'redaktion' | 'sport'
  setKonkurrenzMode: (v: 'redaktion' | 'sport') => void

  // Tagesplan
  tagesplanMode: TagesplanMode
  tagesplanDate: string
  setTagesplanMode: (v: TagesplanMode) => void
  setTagesplanDate: (v: string) => void
}

export const useAppStore = create<AppState>((set) => ({
  kandidatenSearch: '',
  kandidatenCategory: 'alle',
  kandidatenChannel: 'alle',
  setKandidatenSearch: (v) => set({ kandidatenSearch: v }),
  setKandidatenCategory: (v) => set({ kandidatenCategory: v }),
  setKandidatenChannel: (v) => set({ kandidatenChannel: v }),

  liveDays: 7,
  liveChannel: 'alle',
  setLiveDays: (v) => set({ liveDays: v }),
  setLiveChannel: (v) => set({ liveChannel: v }),

  konkurrenzMode: 'redaktion',
  setKonkurrenzMode: (v) => set({ konkurrenzMode: v }),

  tagesplanMode: 'redaktion',
  tagesplanDate: new Date().toISOString().slice(0, 10),
  setTagesplanMode: (v) => set({ tagesplanMode: v }),
  setTagesplanDate: (v) => set({ tagesplanDate: v }),
}))
