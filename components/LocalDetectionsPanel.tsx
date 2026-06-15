"use client";

import type { LocalDetection } from "../lib/types";

export function LocalDetectionsPanel({ detections }: { detections: LocalDetection[] }) {
  return (
    <section className="panel edge-hints-panel">
      <div className="panel-heading">
        <h2>Edge Hints</h2>
        <span>{detections.length} high-confidence</span>
      </div>
      {detections.length === 0 ? (
        <p className="empty">High-confidence browser hints will appear here. Gemini handles scene understanding.</p>
      ) : (
        <div className="chip-list">
          {detections.map((detection, index) => (
            <span key={`${detection.label}-${index}`} className="chip">
              hint: {detection.label} {Math.round(detection.score * 100)}%
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
