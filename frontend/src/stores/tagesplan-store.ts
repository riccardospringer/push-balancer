import { create } from 'zustand'
import type { TagesplanMode } from '@/types/api'

interface TagesplanState {
  tagesplanMode: TagesplanMode
  tagesplanDate: string
  setTagesplanMode: (value: TagesplanMode) => void
  setTagesplanDate: (value: string) => void
}

export const useTagesplanStore = create<TagesplanState>((set) => ({
  tagesplanMode: 'redaktion',
  tagesplanDate: new Date().toISOString().slice(0, 10),
  setTagesplanMode: (value) => set({ tagesplanMode: value }),
  setTagesplanDate: (value) => set({ tagesplanDate: value }),
}))
