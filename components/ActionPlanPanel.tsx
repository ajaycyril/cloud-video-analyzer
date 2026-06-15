"use client";

import type { AnalysisResponse } from "../lib/types";

export function ActionPlanPanel({ analysis }: { analysis: AnalysisResponse | null }) {
  return (
    <section className="panel action-panel">
      <div className="panel-heading">
        <h2>Action Plan</h2>
        <span>human actuator</span>
      </div>
      {!analysis ? (
        <p className="empty">Recommended physical actions will appear here.</p>
      ) : (
        <div className="action-list">
          {analysis.actions.map((action) => (
            <div key={`${action.priority}-${action.instruction}`} className={`action-item ${action.status}`}>
              <strong>{action.priority}. {action.instruction}</strong>
              <p>{action.reason}</p>
              <span>+{Math.round(action.expectedGain)} expected</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
