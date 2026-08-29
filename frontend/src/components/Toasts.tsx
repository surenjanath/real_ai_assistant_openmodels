"use client";

/**
 * Transient notices, bottom-right.
 *
 * The telemetry terminal already carries everything, but it scrolls, and it is
 * frequently covered by an overlay or simply not where the operator is
 * looking. Anything that actually needs a decision — a failed model call, a
 * finished download, a fired reminder — surfaces here as well and then gets
 * out of the way on its own.
 */

import { useEffect } from "react";
import { useJarvis, type Toast } from "@/state/jarvis";

/** Failures stay up long enough to read a stack-ish message; good news does not. */
const LIFETIME: Record<Toast["kind"], number> = {
  error: 12_000,
  warn: 9_000,
  success: 5_000,
  info: 6_000,
};

const GLYPH: Record<Toast["kind"], string> = {
  error: "✕",
  warn: "!",
  success: "✓",
  info: "·",
};

function ToastCard({ toast }: { toast: Toast }) {
  const dismiss = useJarvis((s) => s.dismissToast);

  useEffect(() => {
    const timer = setTimeout(() => dismiss(toast.id), LIFETIME[toast.kind]);
    return () => clearTimeout(timer);
  }, [toast.id, toast.kind, dismiss]);

  return (
    <div className={`toast toast-${toast.kind}`} role="status">
      <span className="toast-glyph" aria-hidden="true">
        {GLYPH[toast.kind]}
      </span>
      <span className="toast-body">
        <b className="toast-title">{toast.title}</b>
        {toast.detail && <span className="toast-detail">{toast.detail}</span>}
      </span>
      <button
        className="toast-close"
        type="button"
        onClick={() => dismiss(toast.id)}
        aria-label="Dismiss"
      >
        ×
      </button>
      <i className="toast-life" style={{ animationDuration: `${LIFETIME[toast.kind]}ms` }} />
    </div>
  );
}

export default function Toasts() {
  const toasts = useJarvis((s) => s.toasts);
  const pull = useJarvis((s) => s.pull);

  if (toasts.length === 0 && !pull) return null;

  return (
    <div className="toasts" aria-live="polite">
      {/* A download is not transient — it stays pinned until it finishes. */}
      {pull && (
        <div className="toast toast-progress">
          <span className="toast-glyph" aria-hidden="true">
            ↓
          </span>
          <span className="toast-body">
            <b className="toast-title">pulling {pull.model}</b>
            <span className="toast-detail">
              {pull.status}
              {typeof pull.percent === "number" ? ` · ${pull.percent}%` : ""}
            </span>
            <span className="toast-bar">
              <i style={{ width: `${Math.max(2, pull.percent ?? 2)}%` }} />
            </span>
          </span>
        </div>
      )}
      {toasts.map((toast) => (
        <ToastCard key={toast.id} toast={toast} />
      ))}
    </div>
  );
}
