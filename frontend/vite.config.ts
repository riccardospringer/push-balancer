import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/dist-frontend/' : '/',
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@spring-media/editorial-one-ui': path.resolve(
        __dirname,
        './src/editorial-one-ui-shim/index.tsx',
      ),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8050',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../dist-frontend',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          query: ['@tanstack/react-query'],
          charts: ['recharts'],
          state: ['zustand'],
        },
      },
    },
  },
}))
