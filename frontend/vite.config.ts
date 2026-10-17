import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Requests to /api go to the FastAPI backend during development.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
