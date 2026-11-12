import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 前端开发服务器把 /api 转发到后端，避免跨域问题。
// 后端地址可通过环境变量 VITE_API_TARGET 覆盖。
const target = process.env.VITE_API_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    host: '127.0.0.1',
    proxy: {
      '/api': { target, changeOrigin: true },
      '/health': { target, changeOrigin: true }
    }
  }
})