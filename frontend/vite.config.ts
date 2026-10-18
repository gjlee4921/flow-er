import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Requests to /api go to the FastAPI backend during development.
// FLOW_ER_API and FLOW_ER_PORT override the defaults when those ports are taken locally.
const api = process.env.FLOW_ER_API ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.FLOW_ER_PORT ?? 5173),
    strictPort: true,
    proxy: { "/api": { target: api, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
