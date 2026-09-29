/// <reference types="vitest/config" />
import { fileURLToPath, URL } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type ProxyOptions } from "vite";
import { loopbackOnly, servicePort } from "./service-proxy.ts";

// The service binds 127.0.0.1 on RYUK_PORT (8000 by default) and only
// accepts localhost Host headers, so the proxy must rewrite Host
// (changeOrigin). The client always calls it with relative /api URLs; there
// is no CORS. `ws` carries the live monitor's socket, /api/monitor, through
// the same proxy.
const serviceProxy = (): Record<string, ProxyOptions> => ({
  "/api": {
    target: `http://127.0.0.1:${servicePort(process.env)}`,
    changeOrigin: true,
    ws: true,
  },
});

export default defineConfig({
  plugins: [
    // Must run before the React plugin: it generates src/routeTree.gen.ts and
    // splits route components before JSX is transformed.
    tanstackRouter({
      target: "react",
      autoCodeSplitting: true,
      quoteStyle: "double",
      semicolons: true,
    }),
    react(),
    tailwindcss(),
    // The proxy reaches an unauthenticated API: loopback only, always.
    loopbackOnly(),
  ],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    rolldownOptions: {
      output: {
        // Long-lived vendor chunks, so app changes don't bust their cache.
        codeSplitting: {
          groups: [
            {
              name: "react",
              test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/,
            },
            { name: "tanstack", test: /node_modules[\\/]@tanstack[\\/]/ },
            { name: "effect", test: /node_modules[\\/](effect|@effect)[\\/]/ },
          ],
        },
      },
    },
  },
  server: { proxy: serviceProxy() },
  preview: { proxy: serviceProxy() },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}", "service-proxy.test.ts"],
    setupFiles: ["./src/test/setup.ts"],
  },
});
