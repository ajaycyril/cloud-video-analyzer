import type { VideoAnalysisRequest } from "./videoSchema";
import { summarizeHybridEdgeContext } from "./hybridEdgeContext";

export const VIDEO_ANALYTICS_SYSTEM_PROMPT = [
  "You are a senior multimodal video analyst for physical AI demos.",
  "Analyze sampled video frames as a temporal sequence, not as unrelated photos.",
  "For general requests, first describe what is visibly happening in plain language, then explain what matters and what to do next.",
  "For specific safety, zone, PPE, operations, or robotics requests, focus on that requested analytic.",
  "Be explicit about uncertainty. Do not invent people, objects, hazards, or events that are not visible.",
  "Return only the requested JSON shape.",
].join(" ");

const modeFocus: Record<VideoAnalysisRequest["mode"], string> = {
  industrial_general: "general video understanding, visible activity, important objects, practical interpretation, and next-step recommendations",
  person_zone: "person detection, restricted-zone entry, line-crossing, and alert triage",
  ppe: "PPE visibility, worker safety compliance, helmet/vest/glove cues, and confidence limits",
  safety: "unsafe proximity, slips/trips/falls, blocked exits, hazards, and incident triage",
  operations: "process visibility, bottlenecks, queueing, machine area activity, and task recommendations",
};

export function buildVideoAnalysisPrompt(request: VideoAnalysisRequest): string {
  const hybridContext = summarizeHybridEdgeContext(request.frames);

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

  const wantsSpecificAlert = request.mode !== "industrial_general" || /\b(alert|zone|ppe|hazard|risk|safety|detect|violation|intrusion|person|worker)\b/i.test(request.objective);
  const alertInstruction = wantsSpecificAlert
    ? "Populate alerts for the requested analytic when relevant. Use triggered=false with evidence when an expected condition is not met."
    : "Do not force a safety or restricted-zone alert. Keep alerts empty unless the user's question or the visible scene clearly warrants an alert.";

  return [
    `Analyze this ${request.source} video sample for ${modeFocus[request.mode]}.`,
    `Customer analytics objective: ${request.objective}`,
    wantsSpecificAlert
      ? `Drawn zones in normalized image coordinates: ${zoneSummary}. If a zone is relevant, map observations to the zoneId.`
      : `Drawn zones in normalized image coordinates: ${zoneSummary}. Ignore zones unless the user explicitly asks for zone or boundary behavior.`,
    `The browser sent ${request.frames.length} keyframes instead of raw high-FPS video to reduce latency and cloud cost.`,
    `Sampling: target ${request.sampling.requestedFps} fps, max ${request.sampling.maxFrames} frames, jpeg quality ${request.sampling.jpegQuality}, payload ${request.sampling.payloadBytes} bytes.`,
    request.sampling.edgeGate
      ? `Two-step edge gate: ${request.sampling.edgeGate.strategy}. Browser captured ${request.sampling.edgeGate.inputFrames} frames, selected ${request.sampling.edgeGate.selectedFrames}, skipped ${request.sampling.edgeGate.skippedFrames}, object frames ${request.sampling.edgeGate.objectFrames}, motion frames ${request.sampling.edgeGate.motionFrames}.`
      : "Two-step edge gate: not reported by this client.",
    `Hybrid edge-to-cloud context: browser edge models detected ${hybridContext.detections} object class box${hybridContext.detections === 1 ? "" : "es"} across ${hybridContext.objectFrames}/${hybridContext.frames} object-triggered frame${hybridContext.objectFrames === 1 ? "" : "s"}, plus ${hybridContext.motionFrames} motion-triggered frame${hybridContext.motionFrames === 1 ? "" : "s"} and ${hybridContext.usableFrames} usable frame${hybridContext.usableFrames === 1 ? "" : "s"}. Treat edge classes as cheap trigger hints, then use the attached images to reason about behavior, traffic buildup, safety risk, or operational action.`,
    hybridContext.objects.length
      ? `Edge object summary: ${hybridContext.objects
          .slice(0, 10)
          .map((object) => `${object.label} x${object.count} best ${(object.bestScore * 100).toFixed(0)}%`)
          .join(", ")}.`
      : "Edge object summary: no browser detections above threshold.",
    "Edge metrics per frame:",
    edgeSummary,
    hybridContext.promptLines.length ? "Local edge object classes sent as metadata:\n" + hybridContext.promptLines.join("\n") : "Local edge object classes sent as metadata: none above threshold.",
    "Return a practical answer suitable for a live demo. The headline and commentary should read like a direct answer to the user's plain-language question.",
    alertInstruction,
    "Keep commentary punchy enough for live narration.",
  ].join("\n\n");
}
