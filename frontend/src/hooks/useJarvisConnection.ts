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
import { useJarvis } from "@/state/jarvis";
import type { AssistantStatus, AudioServerFrame, CommandFrame, LogsServerFrame } from "@/lib/protocol";

/** Module-level handle so imperative senders (input, STT) reach the socket. */
const sockets: { logs: WebSocket | null } = { logs: null };

export function sendCommand(text: string, origin: "text" | "voice" = "text"): boolean {
  const socket = sockets.logs;
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  const frame: CommandFrame = { type: "command", text, origin };
  socket.send(JSON.stringify(frame));
  return true;
}

function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}`;
}

export function useJarvisConnection(): void {
  const pushLog = useJarvis((s) => s.pushLog);
  const setStatus = useJarvis((s) => s.setStatus);
  const setEngines = useJarvis((s) => s.setEngines);
  const setConnected = useJarvis((s) => s.setConnected);
  const setLatency = useJarvis((s) => s.setLatency);

  /* ---------------- /ws/logs ---------------- */

  useEffect(() => {
    let stopped = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pingTimer: ReturnType<typeof setInterval> | undefined;
    let socket: WebSocket | null = null;

    const connect = () => {
      socket = new WebSocket(wsUrl("/ws/logs"));
      sockets.logs = socket;

      socket.onopen = () => {
        if (stopped) return;
        retry = 0;
        setConnected("logs", true);
        setStatus("idle");
        pingTimer = setInterval(() => {
          if (socket?.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "ping", ts: Date.now() }));
          }
        }, 10_000);
      };

      socket.onmessage = (event) => {
        if (stopped) return;
        let frame: LogsServerFrame & { ts?: number };
        try {
          frame = JSON.parse(event.data as string);
        } catch {
          return;
        }
        switch (frame.type) {
          case "hello":
            setEngines({
              tts: frame.engines.tts,
              ttsMode: frame.engines.tts_mode,
              agents: frame.engines.agents,
              model: frame.engines.model,
            });
            break;
          case "log":
            pushLog(frame.level, frame.source, frame.msg);
            break;
          case "status":
            if (["thinking", "speaking", "idle"].includes(frame.status)) {
              setStatus(frame.status as AssistantStatus, frame.detail);
            }
            break;
          case "pong":
            if (typeof frame.ts === "number") setLatency(Math.max(0, Date.now() - frame.ts));
            break;
        }
      };

      socket.onclose = () => {
        if (stopped) return;
        setConnected("logs", false);
        setLatency(null);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        setConnected("audio", true);
      };
      socket.onmessage = (event) => {
        if (stopped) return;
        let frame: AudioServerFrame;
        try {
          frame = JSON.parse(event.data as string) as AudioServerFrame;
        } catch {
          return;
        }
        switch (frame.type) {
          case "tts.start":
            audioEngine.reset();
            pushLog("voice", "tts", `◀ stream ${frame.utterance_id} [${frame.engine}] "${frame.text.slice(0, 60)}"`);
            audioEngine.unlock();
            break;
          case "tts.chunk":
            audioEngine.push(decodePcm16LE(frame.data), frame.sample_rate);
            break;
          case "tts.end":
            if (frame.cancelled) audioEngine.reset();
            pushLog(
              "info",
              "tts",
              `stream ${frame.utterance_id} ${frame.cancelled ? "cancelled" : "complete"}${frame.duration_s ? ` - ${frame.duration_s}s` : ""}`,
            );
            break;
          case "tts.error":
            pushLog("error", "tts", frame.detail.slice(0, 120));
            break;
        }
      };
      socket.onclose = () => {
        if (stopped) return;
        setConnected("audio", false);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* -------- autoplay unlock on first gesture -------- */

  useEffect(() => {
    const onGesture = () => audioEngine.unlock();
    window.addEventListener("pointerdown", onGesture);
    window.addEventListener("keydown", onGesture);
    return () => {
      window.removeEventListener("pointerdown", onGesture);
      window.removeEventListener("keydown", onGesture);
    };
  }, []);

  useEffect(() => {
    pushLog("info", "client", "interface mounted - webgl canvas + web audio online");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
