import type { VideoAnalysisRequest } from "./videoSchema";

export const VIDEO_ANALYTICS_SYSTEM_PROMPT = [
  "You are a senior cloud video analytics system for physical AI demos.",
  "Analyze sampled video frames as a temporal sequence, not as unrelated photos.",
  "Focus on scene understanding, objects, activity, safety/operational signals, and robot/actionable recommendations.",
  "Be explicit about uncertainty. Do not invent people, objects, hazards, or events that are not visible.",
  "Return only the requested JSON shape.",
].join(" ");

const modeFocus: Record<VideoAnalysisRequest["mode"], string> = {
  industrial_general: "industrial scene, object, event, risk, alert, and action analysis",
  person_zone: "person detection, restricted-zone entry, line-crossing, and alert triage",
  ppe: "PPE visibility, worker safety compliance, helmet/vest/glove cues, and confidence limits",
  safety: "unsafe proximity, slips/trips/falls, blocked exits, hazards, and incident triage",
  operations: "process visibility, bottlenecks, queueing, machine area activity, and task recommendations",
};

export function buildVideoAnalysisPrompt(request: VideoAnalysisRequest): string {
  const localObjects = request.frames
    .flatMap((frame, frameIndex) =>
      frame.localDetections.map(
        (detection) =>
          `frame ${frameIndex + 1}: ${detection.label} ${(detection.score * 100).toFixed(0)}% at x=${detection.x.toFixed(2)}, y=${detection.y.toFixed(2)}`,
      ),
    )
    .slice(0, 24);

  const edgeSummary = request.frames
    .map(
      (frame, frameIndex) =>
        `frame ${frameIndex + 1} @ ${(frame.timestampMs / 1000).toFixed(1)}s: brightness ${frame.edgeMetrics.brightness.toFixed(0)}, sharpness ${frame.edgeMetrics.sharpness.toFixed(0)}, motion ${frame.edgeMetrics.motionScore.toFixed(0)}, complexity ${frame.edgeMetrics.visualComplexity.toFixed(0)}`,
    )
    .join("\n");

  const zoneSummary = request.zones.length
    ? request.zones
        .map(
          (zone) =>
            `${zone.id} "${zone.label}" x=${zone.x.toFixed(2)}, y=${zone.y.toFixed(2)}, w=${zone.w.toFixed(2)}, h=${zone.h.toFixed(2)}`,
        )
        .join("\n")
    : "none";

  return [
    `Analyze this ${request.source} video sample for ${modeFocus[request.mode]}.`,
    `Customer analytics objective: ${request.objective}`,
    `Drawn zones in normalized image coordinates: ${zoneSummary}. If a zone is relevant, map observations to the zoneId.`,
    `The browser sent ${request.frames.length} keyframes instead of raw high-FPS video to reduce latency and cloud cost.`,
    `Sampling: target ${request.sampling.requestedFps} fps, max ${request.sampling.maxFrames} frames, jpeg quality ${request.sampling.jpegQuality}, payload ${request.sampling.payloadBytes} bytes.`,
    "Edge metrics per frame:",
    edgeSummary,
    localObjects.length ? "Local browser detections:\n" + localObjects.join("\n") : "Local browser detections: none above threshold.",
    "Return practical results that could replace a narrow custom video analytics model for a live demo, while calling out residual uncertainty.",
    "Populate alerts for the customer objective even when no alert is triggered. Use triggered=false with evidence when conditions are not met.",
    "Keep commentary punchy enough for live narration.",
  ].join("\n\n");
}
