"use client";

import type { EdgeMetrics } from "../lib/types";

const metricLabels: Array<[keyof EdgeMetrics, string]> = [
  ["brightness", "Brightness"],
  ["contrast", "Contrast"],
  ["sharpness", "Sharpness"],
  ["edgeDensity", "Edge density"],
  ["visualComplexity", "Complexity"],
  ["motionScore", "Motion"],
  ["stability", "Stability"],
];

export function EdgeMetricsPanel({ metrics }: { metrics: EdgeMetrics | null }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Edge Metrics</h2>
        <span>{metrics?.usable ? "usable keyframe" : "local sensing"}</span>
      </div>
      {!metrics ? (
        <p className="empty">Start scanning to see browser-side video analytics.</p>
      ) : (
        <>
          <div className="metric-list">
            {metricLabels.map(([key, label]) => (
              <div key={key} className="metric-row">
                <span>{label}</span>
                <div className="meter">
                  <div style={{ width: `${Number(metrics[key])}%` }} />
                </div>
                <strong>{Math.round(Number(metrics[key]))}</strong>
              </div>
            ))}
          </div>
          {metrics.rejectionReason ? <p className="warning-text">{metrics.rejectionReason}</p> : null}
        </>
      )}
    </section>
  );
}
