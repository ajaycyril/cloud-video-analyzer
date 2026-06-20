import type { SampledFrame, VideoAnalysisResponse } from "./videoSchema";
import { isRealObjectDetection } from "./detectionLabels";

export type EdgeDetectionSummary = {
  label: string;
  count: number;
  score: number;
};

export type EdgeRuntimeSummary = {
  detections: EdgeDetectionSummary[];
  brightness: number;
  sharpness: number;
  stability: number;
  quality: number;
  usableFrames: number;
  rejectedFrames: number;
};

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function summarizeEdgeDetections(frames: SampledFrame[]): EdgeDetectionSummary[] {
  const byLabel = new Map<string, EdgeDetectionSummary>();
  for (const detection of frames.flatMap((frame) => frame.localDetections).filter(isRealObjectDetection)) {
    const key = detection.label.toLowerCase();
    const current = byLabel.get(key);
    if (current) {
      current.count += 1;
      current.score = Math.max(current.score, detection.score);
    } else {
      byLabel.set(key, { label: detection.label, count: 1, score: detection.score });
    }
  }
  return Array.from(byLabel.values())
    .sort((left, right) => right.score - left.score)
    .slice(0, 5);
}

export function summarizeEdgeRuntime(frames: SampledFrame[]): EdgeRuntimeSummary {
  const brightness = Math.round(average(frames.map((frame) => frame.edgeMetrics.brightness)));
  const sharpness = Math.round(average(frames.map((frame) => frame.edgeMetrics.sharpness)));
  const stability = Math.round(average(frames.map((frame) => frame.edgeMetrics.stability)));
  return {
    detections: summarizeEdgeDetections(frames),
    brightness,
    sharpness,
    stability,
    quality: Math.round(average(frames.map((frame) => (frame.edgeMetrics.brightness + frame.edgeMetrics.sharpness + frame.edgeMetrics.stability) / 3))),
    usableFrames: frames.filter((frame) => frame.edgeMetrics.usable).length,
    rejectedFrames: frames.filter((frame) => !frame.edgeMetrics.usable).length,
  };
}

export function fallbackObjectChips(analysis: VideoAnalysisResponse | null): EdgeDetectionSummary[] {
  if (!analysis) {
    return [];
  }
  return analysis.objects.slice(0, 4).map((object) => ({
    label: object.label,
    count: object.count,
    score: analysis.confidence / 100,
  }));
}
