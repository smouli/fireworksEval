import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import axiosFix from './vite-axios-fix.js'

export default defineConfig({
  plugins: [react(), axiosFix()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
  optimizeDeps: {
    include: ['axios'],
  },
})

