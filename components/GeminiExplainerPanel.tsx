"use client";

export function GeminiExplainerPanel() {
  return (
    <section className="panel gemini-explainer">
      <div>
        <p className="eyebrow">Gemini API in this demo</p>
        <h2>From camera frame to physical action plan</h2>
        <p>
          Gemini API is the multimodal reasoning layer. The app sends a selected stable camera keyframe plus browser sensor signals, then Gemini returns structured JSON the UI can use directly: world state, score, commentary, overlays, actions, and verification.
        </p>
      </div>

      <div className="explainer-grid">
        <div>
          <strong>What it does</strong>
          <span>Reads the visual scene, reasons about objects and affordances, and recommends physical improvements.</span>
        </div>
        <div>
          <strong>Why it matters</strong>
          <span>This is a small physical AI loop: observe the real world, model state, plan an action, then rescore after the world changes.</span>
        </div>
        <div>
          <strong>What stays local</strong>
          <span>The browser handles camera access, frame quality checks, motion stability, edge metrics, and high-confidence hints.</span>
        </div>
        <div>
          <strong>Demo guardrails</strong>
          <span>API keys stay server-side, frames are capped, request bursts are limited, and the demo has a per-session usage counter.</span>
        </div>
      </div>
    </section>
  );
}
