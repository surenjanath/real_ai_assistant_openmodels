/**
 * Thin REST client for the same-origin backend (proxied by `server.mjs`).
 *
 * The WebSocket carries everything live; these are the request/response
 * corners that do not belong on a broadcast bus — browsing stored notes,
 * deleting a fact, exporting a transcript. Every call resolves to a value
 * rather than throwing, because a panel that cannot reach the backend should
 * render "unavailable", not crash the interface.
 */

export interface Fact {
  key: string;
  value: string;
  ts: number;
  hits: number;
}

export interface Note {
  id: number;
  text: string;
  tags: string | null;
  ts: number;
}

export interface Reminder {
  id: number;
  text: string;
  due_ts: number;
  created_ts?: number;
}

export interface Recollection {
  role: string;
  text: string;
  ts: number;
  score: number;
}

async function request<T>(path: string, init?: RequestInit, fallback?: T): Promise<T | null> {
  try {
    const res = await fetch(path, {
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
      ...init,
    });
    if (!res.ok) return fallback ?? null;
    return (await res.json()) as T;
  } catch {
    return fallback ?? null;
  }
}

export const api = {
  facts: () => request<{ ok: boolean; facts: Fact[] }>("/api/facts"),
  forgetFact: (key: string) =>
    request<{ ok: boolean }>(`/api/facts/${encodeURIComponent(key)}`, { method: "DELETE" }),

  notes: () => request<{ ok: boolean; notes: Note[] }>("/api/notes"),
  addNote: (text: string) =>
    request<{ ok: boolean; id: number }>("/api/notes", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  deleteNote: (id: number) =>
    request<{ ok: boolean }>(`/api/notes/${id}`, { method: "DELETE" }),

  reminders: () => request<{ ok: boolean; reminders: Reminder[] }>("/api/reminders"),
  addReminder: (text: string, when: string) =>
    request<{ ok: boolean }>("/api/reminders", {
      method: "POST",
      body: JSON.stringify({ text, when }),
    }),
  cancelReminder: (id: number) =>
    request<{ ok: boolean }>(`/api/reminders/${id}`, { method: "DELETE" }),

  searchMemory: (q: string) =>
    request<{ ok: boolean; matches: Recollection[] }>(
      `/api/memory/search?q=${encodeURIComponent(q)}`,
    ),
  memoryStats: () => request<{ ok: boolean; stats: Record<string, unknown> }>("/api/memory"),
  wipeMemory: () => request<{ ok: boolean; removed: number }>("/api/memory", { method: "DELETE" }),

  /** Opens the plain-text transcript in a new tab; the backend renders it. */
  exportUrl: (fmt: "text" | "json" = "text") => `/api/export?fmt=${fmt}&limit=500`,
};
