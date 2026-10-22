import { defineConfig } from "vite";

export default defineConfig({
  server: { fs: { allow: [".."] } },
  worker: { format: "es" },
  build: { rollupOptions: { input: { main: "index.html", hideseek: "hideseek.html" } } },
});
