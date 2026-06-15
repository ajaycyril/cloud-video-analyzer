"use client";

import type { AnalysisResponse } from "../lib/types";

export function ScorePanel({ analysis }: { analysis: AnalysisResponse | null }) {
  const score = analysis?.score ?? 0;
  return (
    <section className="panel score-panel">
      <div className="score-ring" style={{ background: `conic-gradient(#77f7b0 ${score * 3.6}deg, rgba(255,255,255,0.09) 0deg)` }}>
        <div>
          <strong>{analysis ? Math.round(score) : "--"}</strong>
          <span>/100</span>
        </div>
      </div>
      <div>
        <div className="panel-heading inline">
          <h2>Organization Score</h2>
          <span>{analysis?.scoreLabel ?? "waiting"}</span>
        </div>
        {analysis ? (
          <div className="subscore-grid">
            {Object.entries(analysis.subscores).map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{Math.round(value)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="empty">Score appears after the first valid Gemini analysis.</p>
        )}
      </div>
    </section>
  );
}
