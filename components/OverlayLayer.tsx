"use client";

import type { AnalysisResponse } from "../lib/types";

type OverlayLayerProps = {
  analysis: AnalysisResponse | null;
};

function severityClass(severity: string): string {
  if (severity === "high") {
    return "overlay-high";
  }
  if (severity === "positive") {
    return "overlay-positive";
  }
  return "overlay-medium";
}

export function OverlayLayer({ analysis }: OverlayLayerProps) {
  const overlays = analysis?.overlays ?? [];
  return (
    <div className="overlay-layer" aria-hidden="true">
      {overlays.map((overlay, index) => {
        if (overlay.type === "arrow" && overlay.toX !== undefined && overlay.toY !== undefined) {
          return (
            <svg key={`${overlay.label}-${index}`} className="arrow-overlay">
              <line
                x1={`${overlay.x * 100}%`}
                y1={`${overlay.y * 100}%`}
                x2={`${overlay.toX * 100}%`}
                y2={`${overlay.toY * 100}%`}
                className={severityClass(overlay.severity)}
              />
              <text x={`${overlay.toX * 100}%`} y={`${overlay.toY * 100}%`}>
                {overlay.label}
              </text>
            </svg>
          );
        }

        return (
          <div
            key={`${overlay.label}-${index}`}
            className={`model-overlay ${severityClass(overlay.severity)}`}
            style={{
              left: `${overlay.x * 100}%`,
              top: `${overlay.y * 100}%`,
              width: `${(overlay.w ?? 0.18) * 100}%`,
              height: `${(overlay.h ?? 0.09) * 100}%`,
            }}
          >
            <span>{overlay.label}</span>
          </div>
        );
      })}
    </div>
  );
}
