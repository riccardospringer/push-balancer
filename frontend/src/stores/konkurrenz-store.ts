import { create } from 'zustand'

interface KonkurrenzState {
  konkurrenzMode: 'redaktion' | 'sport'
  setKonkurrenzMode: (value: 'redaktion' | 'sport') => void
}

export const useKonkurrenzStore = create<KonkurrenzState>((set) => ({
  konkurrenzMode: 'redaktion',
  setKonkurrenzMode: (value) => set({ konkurrenzMode: value }),
}))
