import type { SampledFrame } from "./videoSchema";

const DEFAULT_MAX_CLOUD_FRAMES = 3;
const MOTION_TRIGGER_SCORE = 8;
const MIN_DETECTION_SCORE = 0.5;

type EdgeGateOptions = {
  allowFallback?: boolean;
};

export type EdgeGateFrameDecision = {
  frameIndex: number;
  sendToCloud: boolean;
  score: number;
  reasons: string[];
};

export type EdgeGateSummary = {
  inputFrames: number;
  selectedFrames: number;
  skippedFrames: number;
  objectFrames: number;
  motionFrames: number;
  staticFrames: number;
  strategy: string;
  decisions: EdgeGateFrameDecision[];
};

function frameScore(frame: SampledFrame): number {
  const bestDetection = frame.localDetections.reduce((best, detection) => Math.max(best, detection.score), 0);
  const detectionWeight = bestDetection * 130;
  const motionWeight = frame.edgeMetrics.motionScore * 2.1;
  const qualityWeight = frame.edgeMetrics.usable ? 18 : -20;
  const complexityWeight = frame.edgeMetrics.visualComplexity * 0.15;
  return detectionWeight + motionWeight + qualityWeight + complexityWeight;
}

function frameReasons(frame: SampledFrame): string[] {
  const reasons: string[] = [];
  const confidentDetections = frame.localDetections.filter((detection) => detection.score >= MIN_DETECTION_SCORE);
  if (confidentDetections.length) {
    reasons.push(`${confidentDetections.length} local object${confidentDetections.length === 1 ? "" : "s"}`);
  }
  if (frame.edgeMetrics.motionScore >= MOTION_TRIGGER_SCORE) {
    reasons.push(`motion ${Math.round(frame.edgeMetrics.motionScore)}`);
  }
  if (!frame.edgeMetrics.usable && frame.edgeMetrics.rejectionReason) {
    reasons.push(frame.edgeMetrics.rejectionReason);
  }
  return reasons;
}

export function buildEdgeGateSummary(frames: SampledFrame[], maxCloudFrames = DEFAULT_MAX_CLOUD_FRAMES, options: EdgeGateOptions = {}): EdgeGateSummary {
  const decisions = frames.map((frame, frameIndex) => {
    const reasons = frameReasons(frame);
    const sendToCloud = frame.edgeMetrics.usable && reasons.some((reason) => reason.startsWith("motion") || reason.includes("local object"));
    return {
      frameIndex,
      sendToCloud,
      score: frameScore(frame),
      reasons,
    };
  });

  const eligible = decisions.filter((decision) => decision.sendToCloud);
  const fallbackUsed = eligible.length === 0 && options.allowFallback && frames.length > 0;
  const candidates = fallbackUsed ? decisions : eligible;
  const selectedIndexes = new Set(
    candidates
      .slice()
      .sort((left, right) => right.score - left.score)
      .slice(0, Math.max(1, maxCloudFrames))
      .map((decision) => decision.frameIndex),
  );
  const finalDecisions = decisions.map((decision) => ({
    ...decision,
    sendToCloud: selectedIndexes.has(decision.frameIndex),
    reasons: selectedIndexes.has(decision.frameIndex) && fallbackUsed ? [...decision.reasons, "forced cloud fallback"] : decision.reasons,
  }));

  const objectFrames = frames.filter((frame) => frame.localDetections.some((detection) => detection.score >= MIN_DETECTION_SCORE)).length;
  const motionFrames = frames.filter((frame) => frame.edgeMetrics.motionScore >= MOTION_TRIGGER_SCORE).length;
  const selectedFrames = finalDecisions.filter((decision) => decision.sendToCloud).length;

  return {
    inputFrames: frames.length,
    selectedFrames,
    skippedFrames: Math.max(0, frames.length - selectedFrames),
    objectFrames,
    motionFrames,
    staticFrames: frames.length - new Set([
      ...frames
        .map((frame, index) => (frame.localDetections.some((detection) => detection.score >= MIN_DETECTION_SCORE) ? index : -1))
        .filter((index) => index >= 0),
      ...frames.map((frame, index) => (frame.edgeMetrics.motionScore >= MOTION_TRIGGER_SCORE ? index : -1)).filter((index) => index >= 0),
    ]).size,
    strategy: fallbackUsed
      ? `browser edge gate: no trigger, forced cloud fallback selected best frames`
      : `browser edge gate: send frames only when local objects >= ${Math.round(MIN_DETECTION_SCORE * 100)}% or motion >= ${MOTION_TRIGGER_SCORE}`,
    decisions: finalDecisions,
  };
}

export function selectEdgeTriggeredFrames(frames: SampledFrame[], maxCloudFrames = DEFAULT_MAX_CLOUD_FRAMES, options: EdgeGateOptions = {}): {
  frames: SampledFrame[];
  summary: EdgeGateSummary;
} {
  const summary = buildEdgeGateSummary(frames, maxCloudFrames, options);
  const selectedIndexes = new Set(summary.decisions.filter((decision) => decision.sendToCloud).map((decision) => decision.frameIndex));
  return {
    frames: frames.filter((_, index) => selectedIndexes.has(index)),
    summary,
  };
}
