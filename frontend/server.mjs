/**
 * J.A.R.V.I.S. custom Next.js server.
 *
 * Serves the Next.js app on :3000 and transparently proxies the backend's
 * WebSocket (/ws/*) and REST (/api/*) endpoints from the SAME ORIGIN, so the
 * browser never needs cross-origin connections (and the app works unchanged
 * behind reverse proxies such as the e2b preview or Tauri).
 */
import { createServer } from "node:http";
import next from "next";
import httpProxy from "http-proxy";

const PORT = parseInt(process.env.PORT || "3000", 10);
const HOSTNAME = process.env.HOSTNAME || "0.0.0.0";
const BACKEND = process.env.BACKEND_URL || "http://127.0.0.1:8000";
const dev = process.env.NODE_ENV !== "production";

const app = next({ dev, hostname: HOSTNAME, port: PORT });
const handle = app.getRequestHandler();
const upgrade = app.getUpgradeHandler();

const proxy = httpProxy.createProxyServer({ target: BACKEND, ws: true });
proxy.on("error", (err, _req, res) => {
  console.warn(`[proxy] backend unreachable: ${err.code || err.message}`);
  if (res && typeof res.writeHead === "function") {
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "backend unavailable", backend: BACKEND }));
  }
});

const isBackendPath = (url = "") => url.startsWith("/ws/") || url.startsWith("/api/");

app
  .prepare()
  .then(() => {
    const server = createServer((req, res) => {
      if (isBackendPath(req.url)) {
        proxy.web(req, res);
      } else {
        handle(req, res);
      }
    });

    // WebSocket upgrades: backend sockets + Next.js HMR.
    server.on("upgrade", (req, socket, head) => {
      if (isBackendPath(req.url)) {
        proxy.ws(req, socket, head);
      } else {
        upgrade(req, socket, head);
      }
    });

    server.listen(PORT, HOSTNAME, () => {
      console.log(`> J.A.R.V.I.S. interface ready on http://${HOSTNAME}:${PORT} (dev=${dev})`);
      console.log(`> proxying /ws/* and /api/* -> ${BACKEND}`);
    });
  })
  .catch((err) => {
    console.error("failed to start server:", err);
    process.exit(1);
  });
