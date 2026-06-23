import { defineConfig } from 'vite'

export default defineConfig({
  base: '/web-rasterization/', 
  server: {
    watch: {
      usePolling: true,
      interval: 100
    },
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true
      }
    }
  }
})
