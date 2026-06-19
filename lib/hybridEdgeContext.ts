import type { SampledFrame } from "./videoSchema";

export type EdgeObjectSummary = {
  label: string;
  count: number;
  bestScore: number;
};

export type HybridEdgeContext = {
  frames: number;
  detections: number;
  objectFrames: number;
  motionFrames: number;
  usableFrames: number;
  objects: EdgeObjectSummary[];
  promptLines: string[];
};

const PROMPT_DETECTION_LIMIT = 28;
const OBJECT_TRIGGER_SCORE = 0.5;
const MOTION_TRIGGER_SCORE = 8;

export function summarizeHybridEdgeContext(frames: SampledFrame[]): HybridEdgeContext {
  const objectMap = new Map<string, EdgeObjectSummary>();
  const promptLines: string[] = [];
  let detections = 0;
  let objectFrames = 0;
  let motionFrames = 0;
  let usableFrames = 0;

  frames.forEach((frame, frameIndex) => {
    if (frame.edgeMetrics.usable) {
      usableFrames += 1;
    }
    if (frame.edgeMetrics.motionScore >= MOTION_TRIGGER_SCORE) {
      motionFrames += 1;
    }

    const strongDetections = frame.localDetections.filter((detection) => detection.score >= OBJECT_TRIGGER_SCORE);
    if (strongDetections.length) {
      objectFrames += 1;
    }

    frame.localDetections.forEach((detection) => {
      detections += 1;
      const label = detection.label.toLowerCase();
      const existing = objectMap.get(label);
      if (existing) {
        existing.count += 1;
        existing.bestScore = Math.max(existing.bestScore, detection.score);
      } else {
        objectMap.set(label, {
          label: detection.label,
          count: 1,
          bestScore: detection.score,
        });
      }

      if (promptLines.length < PROMPT_DETECTION_LIMIT) {
        promptLines.push(
          [
            `frame ${frameIndex + 1}`,
            `${detection.label} ${(detection.score * 100).toFixed(0)}%`,
            `box x=${detection.x.toFixed(2)} y=${detection.y.toFixed(2)} w=${detection.w.toFixed(2)} h=${detection.h.toFixed(2)}`,
            `motion=${frame.edgeMetrics.motionScore.toFixed(0)}`,
          ].join(": "),
        );
      }
    });
  });

  return {
    frames: frames.length,
    detections,
    objectFrames,
    motionFrames,
    usableFrames,
    objects: [...objectMap.values()].sort((left, right) => right.bestScore - left.bestScore || right.count - left.count),
    promptLines,
  };
}
