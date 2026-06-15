import { FilesetResolver, ObjectDetector, type ObjectDetectorResult } from "@mediapipe/tasks-vision";
import type { LocalDetection } from "./types";

const MODEL_PATH = "/models/efficientdet_lite0.tflite";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MIN_DETECTION_SCORE = 0.68;
const MIN_BOX_AREA = 0.012;
const MAX_DETECTIONS = 5;

let detectorPromise: Promise<ObjectDetector> | null = null;

function normalize(value: number, max: number): number {
  if (!Number.isFinite(value) || max <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, value / max));
}

export async function createObjectDetector(): Promise<ObjectDetector> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const response = await fetch(MODEL_PATH, { method: "HEAD" });
      if (!response.ok) {
        throw new Error("Edge object detection model missing at /models/efficientdet_lite0.tflite.");
      }

      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      return ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_PATH,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        scoreThreshold: MIN_DETECTION_SCORE,
        maxResults: MAX_DETECTIONS,
      });
    })();
  }
  return detectorPromise;
}

export function mapDetections(result: ObjectDetectorResult, video: HTMLVideoElement): LocalDetection[] {
  const videoWidth = video.videoWidth || 1;
  const videoHeight = video.videoHeight || 1;

  return result.detections
    .map((detection) => {
      const category = detection.categories[0];
      const box = detection.boundingBox;
      return {
        label: category?.categoryName ?? "object",
        score: Math.max(0, Math.min(1, category?.score ?? 0)),
        x: normalize(box?.originX ?? 0, videoWidth),
        y: normalize(box?.originY ?? 0, videoHeight),
        w: normalize(box?.width ?? 0, videoWidth),
        h: normalize(box?.height ?? 0, videoHeight),
      };
    })
    .filter((detection) => detection.score >= MIN_DETECTION_SCORE && detection.w * detection.h >= MIN_BOX_AREA)
    .sort((a, b) => b.score - a.score)
    .slice(0, MAX_DETECTIONS);
}

export async function detectObjectsForVideo(video: HTMLVideoElement, timestampMs: number): Promise<LocalDetection[]> {
  const detector = await createObjectDetector();
  const result = detector.detectForVideo(video, timestampMs);
  return mapDetections(result, video);
}
