import { create } from 'zustand'

interface LivePushFilterState {
  liveChannel: string
  setLiveChannel: (value: string) => void
}

export const useLivePushFilterStore = create<LivePushFilterState>((set) => ({
  liveChannel: 'alle',
  setLiveChannel: (value) => set({ liveChannel: value }),
}))
