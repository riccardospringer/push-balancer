import { create } from 'zustand'

interface KandidatenFilterState {
  kandidatenSearch: string
  kandidatenCategory: string
  setKandidatenSearch: (value: string) => void
  setKandidatenCategory: (value: string) => void
}

export const useKandidatenFilterStore = create<KandidatenFilterState>(
  (set) => ({
    kandidatenSearch: '',
    kandidatenCategory: 'alle',
    setKandidatenSearch: (value) => set({ kandidatenSearch: value }),
    setKandidatenCategory: (value) => set({ kandidatenCategory: value }),
  }),
)
