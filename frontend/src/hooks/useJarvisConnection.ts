"use client";

/**
 * Owns both WebSocket uplinks (same-origin, proxied by server.mjs):
 *   /ws/logs  - telemetry in, commands out
 *   /ws/audio - TTS PCM stream in
 *
 * Reconnects with capped backoff; feeds PCM chunks into the AudioEngine.
 */

import { useEffect } from "react";
import { audioEngine, decodePcm16LE } from "@/audio/engine";
import { audioLevels } from "@/audio/levels";
import { applyActivation, setGraph } from "@/state/neural";
import { useJarvis } from "@/state/jarvis";
import type {
  AssistantStatus,
  AudioServerFrame,
  CommandFrame,
  LogsServerFrame,
  SettingsCommandFrame,
} from "@/lib/protocol";

/** Module-level handle so imperative senders (input, STT) reach the socket. */
const sockets: { logs: WebSocket | null } = { logs: null };

export function sendCommand(text: string, origin: "text" | "voice" = "text"): boolean {
  const socket = sockets.logs;
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  const frame: CommandFrame = { type: "command", text, origin };
  socket.send(JSON.stringify(frame));
  return true;
}

/** Push a settings delta over the live socket, falling back to REST. */
export function sendSettings(delta: Omit<SettingsCommandFrame, "type">): void {
  const socket = sockets.logs;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "settings", ...delta }));
    return;
  }
  void fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(delta),
  }).catch(() => undefined);
}

/** Invoke a skill by hand — the same path the cortex takes. */
export function sendSkill(name: string, args: Record<string, unknown> = {}): void {
  const socket = sockets.logs;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "skill", name, arguments: args }));
    return;
  }
  void fetch(`/api/skills/${encodeURIComponent(name)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arguments: args }),
  }).catch(() => undefined);
}

/** Barge-in: cut the assistant off mid-sentence, locally and server-side. */
export function sendStop(): void {
  audioEngine.flush();
  const socket = sockets.logs;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "stop" }));
  } else {
    void fetch("/api/stop", { method: "POST" }).catch(() => undefined);
  }
}

function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}`;
}

export function useJarvisConnection(): void {
  /* ---------------- initial settings snapshot ---------------- */

  useEffect(() => {
    fetch("/api/settings")
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((data) => useJarvis.getState().setSettings(data.settings))
      .catch(() => undefined);
  }, []);

  /* ---------------- /ws/logs ---------------- */

  useEffect(() => {
    let stopped = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pingTimer: ReturnType<typeof setInterval> | undefined;
    let socket: WebSocket | null = null;

    const connect = () => {
      const store = useJarvis.getState();
      socket = new WebSocket(wsUrl("/ws/logs"));
      sockets.logs = socket;

      socket.onopen = () => {
        if (stopped) return;
        retry = 0;
        store.setConnected("logs", true);
        store.setStatus("idle");
        pingTimer = setInterval(() => {
          if (socket?.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "ping", ts: Date.now() }));
          }
        }, 10_000);
      };

      socket.onmessage = (event) => {
        if (stopped) return;
        const s = useJarvis.getState();
        let frame: LogsServerFrame & { ts?: number };
        try {
          frame = JSON.parse(event.data as string);
        } catch {
          return;
        }
        switch (frame.type) {
          case "hello":
            s.setEngines({
              tts: frame.engines.tts,
              ttsLabel: frame.engines.tts_label ?? frame.engines.tts,
              ttsMode: frame.engines.tts_mode,
              agents: frame.engines.agents,
              mode: frame.engines.mode ?? "",
              model: frame.engines.model,
            });
            if (frame.operator) s.setOperator(frame.operator);
            if (frame.settings) {
              s.setSettings(frame.settings);
              audioEngine.setVolume(frame.settings.volume ?? 0.9);
            }
            if (frame.vitals) s.setVitals(frame.vitals);
            if (frame.skills) s.setSkills(frame.skills);
            if (frame.memory) s.setMemoryStats(frame.memory);
            break;
          case "log":
            s.pushLog(frame.level, frame.source, frame.msg);
            break;
          case "status":
            if (["thinking", "speaking", "idle", "listening"].includes(frame.status)) {
              s.setStatus(frame.status as AssistantStatus, frame.detail);
              // Drive the scene's "thinking" visual without React re-renders.
              audioLevels.thinking = frame.status === "thinking" ? 1 : 0;
            }
            break;
          case "settings.update":
            s.setSettings(frame.settings);
            s.setEngines({ model: frame.settings.model });
            // Volume lives in the registry so every client agrees on it, but
            // it is only ever *applied* here, in the browser's audio graph.
            audioEngine.setVolume(frame.settings.volume ?? 0.9);
            break;
          case "neural.graph":
            setGraph(frame.nodes, frame.edges);
            s.setNeuralSize(frame.nodes.length, frame.edges.length);
            break;
          case "neural":
            // Straight into the mutable singleton: at 20 Hz this must not
            // touch React at all, or the whole WebGL tree re-renders.
            applyActivation(frame.levels, frame.flows, frame.regions, frame.fired);
            break;
          case "metrics":
            s.setMetrics(frame);
            break;
          case "tool":
            s.pushTool(frame);
            break;
          case "reminder":
          case "reminder.set":
            s.pushLog(
              "success",
              "reminder",
              frame.type === "reminder.set"
                ? `scheduled: ${frame.text}`
                : `due: ${frame.text}`,
            );
            break;
          case "vitals":
            s.setVitals(frame);
            break;
          case "answer.start":
            s.startCaption();
            break;
          case "answer.delta":
            s.appendCaption(frame.text);
            break;
          case "answer":
            s.setCaption(frame.text);
            break;
          case "transcript":
            s.pushTurn(frame.role, frame.text);
            break;
          case "ui":
            if (frame.action === "open_settings") s.setSettingsOpen(true);
            else if (frame.action === "close_settings") s.setSettingsOpen(false);
            else if (frame.action === "clear_logs") s.clearLogs();
            else if (frame.action === "clear_transcript") s.clearTranscript();
            break;
          case "pong":
            if (typeof frame.ts === "number") s.setLatency(Math.max(0, Date.now() - frame.ts));
            break;
        }
      };

      socket.onclose = () => {
        if (stopped) return;
        const st = useJarvis.getState();
        st.setConnected("logs", false);
        st.setLatency(null);
        clearInterval(pingTimer);
        timer = setTimeout(connect, Math.min(8000, 500 * 2 ** retry++));
      };

      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      clearInterval(pingTimer);
      sockets.logs = null;
      socket?.close();
    };
  }, []);

  /* ---------------- /ws/audio ---------------- */

  useEffect(() => {
    let stopped = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let socket: WebSocket | null = null;

    const connect = () => {
      socket = new WebSocket(wsUrl("/ws/audio"));
      socket.onopen = () => {
        if (stopped) return;
        retry = 0;
        useJarvis.getState().setConnected("audio", true);
      };
      socket.onmessage = (event) => {
        if (stopped) return;
        const s = useJarvis.getState();
        let frame: AudioServerFrame;
        try {
          frame = JSON.parse(event.data as string) as AudioServerFrame;
        } catch {
          return;
        }
        switch (frame.type) {
          case "tts.start":
            audioEngine.reset();
            audioEngine.kick();
            audioEngine.unlock();
            s.pushLog("voice", "tts", `◀ ${frame.utterance_id} [${frame.voice}] "${frame.text.slice(0, 60)}"`);
            break;
          case "tts.chunk":
            audioEngine.push(decodePcm16LE(frame.data), frame.sample_rate);
            break;
          case "tts.end":
            if (frame.cancelled) audioEngine.flush();
            s.pushLog(
              "info",
              "tts",
              `${frame.utterance_id} ${frame.cancelled ? "cancelled" : "complete"}` +
                (frame.duration_s ? ` - ${frame.duration_s}s` : ""),
            );
            break;
          case "tts.flush":
            audioEngine.flush();
            break;
          case "tts.error":
            s.pushLog("error", "tts", frame.detail.slice(0, 140));
            break;
        }
      };
      socket.onclose = () => {
        if (stopped) return;
        useJarvis.getState().setConnected("audio", false);
        timer = setTimeout(connect, Math.min(8000, 500 * 2 ** retry++));
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, []);

  /* -------- autoplay unlock on first gesture -------- */

  useEffect(() => {
    audioEngine.watchState((unlocked) => useJarvis.getState().setAudioUnlocked(unlocked));
    const onGesture = () => audioEngine.unlock();
    window.addEventListener("pointerdown", onGesture);
    window.addEventListener("keydown", onGesture);
    return () => {
      window.removeEventListener("pointerdown", onGesture);
      window.removeEventListener("keydown", onGesture);
    };
  }, []);

  useEffect(() => {
    useJarvis.getState().pushLog("info", "client", "interface mounted - webgl canvas + web audio online");
  }, []);
}
