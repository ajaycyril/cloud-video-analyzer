"use client";

import type { AnalysisResponse } from "../lib/types";

export function VerificationPanel({ analysis, baseline }: { analysis: AnalysisResponse | null; baseline: AnalysisResponse | null }) {
  const verification = analysis?.verification;
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Verification</h2>
        <span>closed loop</span>
      </div>
      {!analysis ? (
        <p className="empty">Set a baseline, change the scene, then rescore.</p>
      ) : (
        <div className="verification-grid">
          <div>
            <span>Baseline</span>
            <strong>{baseline ? Math.round(baseline.score) : "--"}</strong>
          </div>
          <div>
            <span>Current</span>
            <strong>{Math.round(analysis.score)}</strong>
          </div>
          <div>
            <span>Delta</span>
            <strong>{verification?.delta === null || verification?.delta === undefined ? "--" : `${verification.delta > 0 ? "+" : ""}${Math.round(verification.delta)}`}</strong>
          </div>
        </div>
      )}
      {verification ? (
        <div className="resolved-grid">
          <div>
            <h3>Resolved</h3>
            {verification.resolved.map((item) => <span key={item}>{item}</span>)}
          </div>
          <div>
            <h3>Remaining</h3>
            {verification.remaining.map((item) => <span key={item}>{item}</span>)}
          </div>
        </div>
      ) : null}
    </section>
  );
}
