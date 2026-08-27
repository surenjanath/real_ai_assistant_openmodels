"use client";

/**
 * Mathematical anchor (PRD §2): sphere volume formula, centered at the
 * bottom of the viewport, muted gray, clean sans-serif.
 *
 * Spacing uses explicit unicode escapes rather than HTML entities - JSX only
 * decodes a subset of named entities, and `&hairsp;` renders literally.
 */

const THIN = " "; // thin space
const HAIR = " "; // hair space

export default function MathAnchor() {
  return (
    <div className="math-anchor" aria-label="Volume of a sphere">
      <span className="math-rule" />
      <span className="math-formula">
        {`V${THIN}=${THIN}`}
        <span className="math-frac">
          <span>4</span>
          <span>3</span>
        </span>
        <span className="pi">π</span>
        {`${HAIR}r³`}
      </span>
      <span className="math-rule" />
    </div>
  );
}
