import type { LocalDetection } from "./types";

const MODEL_ID = "Xenova/detr-resnet-50";
const MIN_SCORE = 0.58;
const MAX_DETECTIONS = 4;

type DetectorRuntime = {
  available: boolean;
  ready: boolean;
  loading: boolean;
  device: "webgpu" | "wasm" | "unavailable";
  label: string;
};

type DetectionPipeline = (input: string, options?: { threshold?: number; percentage?: boolean }) => Promise<unknown[]>;
type TransformersModule = {
  pipeline: (task: "object-detection", model: string, options: { device: "webgpu" }) => Promise<DetectionPipeline>;
};

let detectorPromise: Promise<DetectionPipeline> | null = null;
let runtime: DetectorRuntime = {
  available: false,
  ready: false,
  loading: false,
  device: "unavailable",
  label: "Transformers.js detector not loaded",
};

function hasWebGpu(): boolean {
  return typeof navigator !== "undefined" && Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
}

function isMobileBrowser(): boolean {
  if (typeof navigator === "undefined") {
    return true;
  }
  return /android|iphone|ipad|ipod/i.test(navigator.userAgent);
}

export function shouldUseTransformersDetector(): boolean {
  if (typeof navigator === "undefined" || !hasWebGpu() || isMobileBrowser()) {
    return false;
  }
  const nav = navigator as Navigator & { deviceMemory?: number };
  const enoughMemory = typeof nav.deviceMemory !== "number" || nav.deviceMemory >= 4;
  const enoughCores = typeof navigator.hardwareConcurrency !== "number" || navigator.hardwareConcurrency >= 4;
  return enoughMemory && enoughCores;
}

function normalizedBox(value: unknown): Pick<LocalDetection, "x" | "y" | "w" | "h"> | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const box = value as Partial<Record<"xmin" | "ymin" | "xmax" | "ymax" | "x" | "y" | "width" | "height", number>>;
  const left = typeof box.xmin === "number" ? box.xmin : box.x;
  const top = typeof box.ymin === "number" ? box.ymin : box.y;
  const right = typeof box.xmax === "number" ? box.xmax : typeof left === "number" && typeof box.width === "number" ? left + box.width : undefined;
  const bottom = typeof box.ymax === "number" ? box.ymax : typeof top === "number" && typeof box.height === "number" ? top + box.height : undefined;
  if (typeof left !== "number" || typeof top !== "number" || typeof right !== "number" || typeof bottom !== "number") {
    return null;
  }
  const x = Math.max(0, Math.min(1, left));
  const y = Math.max(0, Math.min(1, top));
  const w = Math.max(0, Math.min(1 - x, right - left));
  const h = Math.max(0, Math.min(1 - y, bottom - top));
  if (w * h < 0.006) {
    return null;
  }
  return { x, y, w, h };
}

function mapTransformersOutput(output: unknown[]): LocalDetection[] {
  return output
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const detection = item as { label?: unknown; score?: unknown; box?: unknown };
      const score = typeof detection.score === "number" ? detection.score : 0;
      const box = normalizedBox(detection.box);
      if (!box || score < MIN_SCORE || typeof detection.label !== "string") {
        return null;
      }
      return {
        label: detection.label,
        score,
        ...box,
      };
    })
    .filter((item): item is LocalDetection => Boolean(item))
    .sort((left, right) => right.score - left.score)
    .slice(0, MAX_DETECTIONS);
}

function timeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T | null> {
  return new Promise((resolve) => {
    const id = window.setTimeout(() => resolve(null), timeoutMs);
    promise
      .then((value) => {
        window.clearTimeout(id);
        resolve(value);
      })
      .catch(() => {
        window.clearTimeout(id);
        resolve(null);
      });
  });
}

async function createDetector(): Promise<DetectionPipeline> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      if (!shouldUseTransformersDetector()) {
        throw new Error("Transformers.js WebGPU detector disabled for this device.");
      }
      const device = "webgpu";
      runtime = {
        available: true,
        ready: false,
        loading: true,
        device,
        label: "Transformers.js WebGPU detector warming",
      };
      const transformers = (await import("@huggingface/transformers")) as unknown as TransformersModule;
      const detector = await transformers.pipeline("object-detection", MODEL_ID, { device });
      runtime = {
        available: true,
        ready: true,
        loading: false,
        device,
        label: "Transformers.js WebGPU detector ready",
      };
      return detector;
    })().catch((error) => {
      detectorPromise = null;
      runtime = {
        available: false,
        ready: false,
        loading: false,
        device: "unavailable",
        label: error instanceof Error ? `Transformers.js unavailable: ${error.message}` : "Transformers.js unavailable",
      };
      throw error;
    });
  }
  return detectorPromise;
}

export function getTransformersDetectorRuntime(): DetectorRuntime {
  return runtime;
}

export async function detectObjectsWithTransformers(imageDataUrl: string, timeoutMs = 1400): Promise<LocalDetection[]> {
  if (typeof window === "undefined" || !shouldUseTransformersDetector()) {
    return [];
  }
  const detector = await timeout(createDetector(), timeoutMs);
  if (!detector) {
    return [];
  }
  const output = await timeout(detector(imageDataUrl, { threshold: MIN_SCORE, percentage: true }), timeoutMs);
  return output ? mapTransformersOutput(output) : [];
}
