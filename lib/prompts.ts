import type { AnalyzeFrameRequest } from "./types";

export const PHYSICAL_AI_SYSTEM_PROMPT = `You are a robotics-style physical AI observer for everyday spaces.
The browser is running edge video analytics before calling you.
Treat edge metrics as reliable sensor signals.
Treat local object detections as weak hints from a generic on-device detector, never as ground truth.
Use the image itself as the authority for object identity, spatial context, and physical meaning.
Build a scene graph / world state.
Identify objects, affordances, spatial relationships, risks, constraints, and usable zones.
Track changes from the previous world state.
Generate concise real-time commentary.
Score organization quality from 0 to 100.
Recommend ordered physical actions that improve organization, layout, accessibility, safety, and visual clarity.
Predict expected score gain.
During verification, compare against baseline and decide what improved.

Mode behavior:
- Space Scan: general physical organization for any visible area: shelves, counters, closets, drawers, cabinets, entryways, room corners, desks, storage zones, or backgrounds.
- Desk Productivity: desk/work readiness, focus zone, reachable tools, cable mess, drink/electronics risk, distractions.
- Webcam Coach: visible laptop webcam scene only: lighting, framing, background clarity, visible clutter, professionalism.

Rules:
- Return JSON only.
- Match the schema exactly.
- Commentary under 28 words.
- Avoid overclaiming.
- If uncertain, mark unknown.
- Do not repeat local detection labels unless the image visually confirms them.
- Prefer generic labels such as item, container, paper, device, cable, or surface when exact identity is uncertain.
- Never claim precise measurements.
- Prefer practical physical actions.
- Suggest grouping, removing, relocating, spacing, labeling, containment, cleaning, layout, access, and safety improvements when relevant.
- Speak like a live visual agent, not a static image captioner.
- Use normalized overlay coordinates from 0 to 1.`;

export function buildAnalysisPrompt(request: AnalyzeFrameRequest): string {
  const previousWorldState = request.previousWorldState
    ? {
        summary: request.previousWorldState.summary,
        objects: request.previousWorldState.objects.slice(0, 6).map((object) => ({
          label: object.label,
          location: object.location,
          movable: object.movable,
        })),
      }
    : null;

  const baseline = request.baseline
    ? {
        score: request.baseline.score,
        scoreLabel: request.baseline.scoreLabel,
        worldSummary: request.baseline.worldState.summary,
        actions: request.baseline.actions.slice(0, 4).map((action) => ({
          instruction: action.instruction,
          status: action.status,
        })),
      }
    : null;

  return JSON.stringify({
    task: "Analyze this camera keyframe as a physical AI organization observer for any visible space.",
    mode: request.mode,
    scanPhase: request.scanPhase,
    edgeMetrics: request.edgeMetrics,
    localDetectionHints: request.localDetections.slice(0, 5).map((detection) => ({
      label: detection.label,
      confidence: Math.round(detection.score * 100),
      box: {
        x: detection.x,
        y: detection.y,
        w: detection.w,
        h: detection.h,
      },
    })),
    previousWorldState,
    baseline,
    outputRules: [
      "Return one compact JSON object matching the configured schema.",
      "Use at most 4 actions, 6 overlays, 6 objects, 6 relationships, and 4 events.",
      "Keep every sentence short.",
      "Prioritize broadly useful organization and layout improvements, not only desk productivity.",
      "If localDetectionHints disagree with the image, ignore the hints.",
    ],
  });
}

export const systemPrompt = PHYSICAL_AI_SYSTEM_PROMPT;

export function buildUserPrompt(request: AnalyzeFrameRequest): string {
  return buildAnalysisPrompt(request);
}
