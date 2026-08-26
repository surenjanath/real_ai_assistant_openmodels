"use client";

/**
 * Mathematical anchor (PRD §2): sphere volume formula, centered at the
 * bottom of the viewport, muted gray, clean sans-serif.
 */

export default function MathAnchor() {
  return (
    <div className="math-anchor" aria-label="Volume of a sphere">
      <span className="math-rule" />
      <span className="math-formula">
        V&thinsp;=&thinsp;⁴⁄₃<span className="pi">π</span>&hairsp;r³
      </span>
      <span className="math-rule" />
    </div>
  );
}
