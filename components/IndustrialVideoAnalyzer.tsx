"use client";

import { AlertTriangle, Camera, Crosshair, FileVideo, Pause, Play, Radar, RotateCcw } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import Image from "next/image";
import { useCallback, useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import { buildVideoConstraints, listVideoInputDevices, stopStream, type CameraFacing } from "@/lib/camera";
import { selectEdgeTriggeredFrames, type EdgeGateSummary } from "@/lib/edgeFrameGate";
import { canvasToJpegDataUrl, captureVideoFrame, computeEdgeMetrics, type EdgeMetricResult } from "@/lib/edgeMetrics";
import { summarizeHybridEdgeContext } from "@/lib/hybridEdgeContext";
import { detectObjectsForVideo, getObjectDetectorRuntime, tryCreateObjectDetector } from "@/lib/mediaPipeDetector";
import { fallbackObjectChips, summarizeEdgeDetections } from "@/lib/resultPresentation";
import { detectObjectsWithTransformers, getTransformersDetectorRuntime } from "@/lib/transformersDetector";
import type { LocalDetection } from "@/lib/types";
import type { ProviderId, SampledFrame, VideoAnalysisResponse, VideoMode, VideoSource, Zone } from "@/lib/videoSchema";

type ApiError = {
  error?: string;
  detail?: string;
};

type RoboflowDetectResponse = {
  configured?: boolean;
  detections?: LocalDetection[];
  model?: string | null;
};

type ZoneCorner = "nw" | "ne" | "sw" | "se";
type VideoOrientation = "landscape-video" | "portrait-video";
type VideoViewport = {
  x: number;
  y: number;
  w: number;
  h: number;
};
type ZoneInteraction =
  | { kind: "draw"; start: { x: number; y: number } }
  | { kind: "resize"; corner: ZoneCorner; zone: Zone };

type HoldRecording = {
  active: boolean;
  stopRequested: boolean;
  frames: SampledFrame[];
  pointerId: number | null;
  startedAt: number;
};

type MotionSnapshot = {
  width: number;
  height: number;
  cells: Uint8Array;
};

const MAX_FRAMES = 4;
const MAX_CLOUD_FRAMES = 3;
const CAMERA_CAPTURE_WINDOW_MS = 1600;
const PREVIEW_DETECTION_INTERVAL_MS = 180;
const TRANSFORMERS_LIVE_INTERVAL_MS = 1400;
const SMART_RECORDING_MIN_MS = 1100;
const SMART_RECORDING_MAX_MS = 2200;
const SMART_RECORDING_INTERVAL_MS = 320;
const EDGE_MOTION_TRIGGER_SCORE = 8;
const EDGE_OBJECT_TRIGGER_SCORE = 0.5;
const MOTION_GRID_COLUMNS = 18;
const MOTION_GRID_ROWS = 12;
const MOTION_CELL_TRIGGER = 26;
const JPEG_QUALITY = 0.54;
const DEMO_ANALYSIS_LIMIT = Number(process.env.NEXT_PUBLIC_DEMO_ANALYSIS_LIMIT ?? 20);

const OBJECTIVE_PRESETS: Array<{ label: string; mode: VideoMode; objective: string }> = [
  {
    label: "Person in zone",
    mode: "person_zone",
    objective: "Detect whether any person enters the marked restricted zone. Trigger an alert only when there is visible evidence.",
  },
  {
    label: "PPE check",
    mode: "ppe",
    objective: "Check whether visible workers appear to be wearing required PPE such as hard hats and safety vests. Mention uncertainty when PPE is unclear.",
  },
  {
    label: "Safety hazards",
    mode: "safety",
    objective: "Identify visible industrial safety hazards, blocked walkways, spill/trip risks, unsafe proximity, and immediate corrective actions.",
  },
  {
    label: "Operations",
    mode: "operations",
    objective: "Summarize operational activity, detect queues or bottlenecks, and recommend what a supervisor or robot should do next.",
  },
  {
    label: "Queue/crowd",
    mode: "operations",
    objective: "Estimate crowding or queue buildup, identify where movement is blocked, and recommend whether a human supervisor or robot should intervene.",
  },
  {
    label: "Asset watch",
    mode: "industrial_general",
    objective: "Identify important visible assets, note whether anything appears misplaced or obstructed, and recommend layout or routing improvements.",
  },
  {
    label: "Custom audit",
    mode: "industrial_general",
    objective: "Analyze the scene using the user's custom instruction. Return alerts only when visible evidence supports them, and include uncertainty where needed.",
  },
];

const DEFAULT_OBJECTIVE =
  "Look at this video and tell me what is happening, what matters, and what I should do next. Be specific, practical, and only use visible evidence.";

const SAMPLE_CLIPS: Array<{ label: string; url: string; note: string; mode: VideoMode; objective: string }> = [
  {
    label: "Paste public MP4/WebM URL",
    url: "",
    note: "Use your own site footage when CORS allows frame extraction.",
    mode: "industrial_general",
    objective: "Analyze the uploaded or linked video for visible activity, hazards, alerts, evidence, and recommended next actions.",
  },
  {
    label: "Factory people",
    url: "/samples/factory-people.webm",
    note: "People flow, zone monitoring, and operations supervision.",
    mode: "person_zone",
    objective: "Detect visible people movement near the marked restricted zone, trigger an alert only if a person appears inside the zone, and summarize operational flow.",
  },
  {
    label: "Road safety",
    url: "/samples/road-traffic.webm",
    note: "Traffic, pedestrians, roadside risk, and supervisor actions.",
    mode: "safety",
    objective: "Analyze road activity, visible vehicles, pedestrian risk, lane or roadside hazards, and recommended supervisor actions.",
  },
  {
    label: "Outdoor activity",
    url: "/samples/outdoor-activity.mp4",
    note: "General-purpose scene understanding outside a factory.",
    mode: "industrial_general",
    objective: "Analyze the visible activity, identify people or objects that matter, flag layout or safety concerns only if visible, and recommend practical next actions.",
  },
  {
    label: "General motion",
    url: "/samples/general-motion.webm",
    note: "Small clip for quick generic video analytics checks.",
    mode: "industrial_general",
    objective: "Describe the visible motion, identify objects or people if visible, and recommend useful analytics outputs for this clip.",
  },
  {
    label: "General activity",
    url: "/samples/general-activity.mp4",
    note: "Short MP4 for cross-browser upload-style analysis.",
    mode: "operations",
    objective: "Analyze visible activity, detect notable objects or movement, and recommend operational actions based only on the video evidence.",
  },
  {
    label: "People/action",
    url: "/samples/people-action.mp4",
    note: "People and motion for general-purpose temporal understanding.",
    mode: "industrial_general",
    objective: "Analyze people, movement, scene changes, risks, and recommended actions across the sampled frames.",
  },
];

const PROVIDER_COPY: Record<ProviderId, { label: string; detail: string }> = {
  gemini: { label: "Gemini Vision", detail: "Live + structured" },
  openai: { label: "OpenAI Vision", detail: "Responses frames" },
  nvidia: { label: "NVIDIA Cosmos", detail: "Physical AI" },
};

function isApiError(value: unknown): value is ApiError {
  return Boolean(value && typeof value === "object" && ("error" in value || "detail" in value));
}

function readableError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected analyzer error.";
}

function formatCost(value: number): string {
  return `$${value.toFixed(4)}`;
}

function payloadBytes(frames: SampledFrame[]): number {
  return frames.reduce((total, frame) => total + frame.imageDataUrl.length, 0);
}

function isNormalizedNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isLocalDetection(value: unknown): value is LocalDetection {
  if (!value || typeof value !== "object") {
    return false;
  }
  const detection = value as Partial<LocalDetection>;
  return (
    typeof detection.label === "string" &&
    detection.label.length > 0 &&
    isNormalizedNumber(detection.score) &&
    isNormalizedNumber(detection.x) &&
    isNormalizedNumber(detection.y) &&
    isNormalizedNumber(detection.w) &&
    isNormalizedNumber(detection.h)
  );
}

function isProposalDetection(detection: LocalDetection): boolean {
  return detection.label.includes("/") || detection.label.includes("region") || detection.label.includes("attention") || detection.label.includes("candidate");
}

function mergeDetections(current: LocalDetection[], additions: LocalDetection[]): LocalDetection[] {
  return [...current, ...additions]
    .sort((left, right) => right.score - left.score)
    .slice(0, 20);
}

function selectDisplayDetections(detections: LocalDetection[], limit = 2): LocalDetection[] {
  const realDetections = detections.filter((detection) => !isProposalDetection(detection));
  const proposals = detections.filter(isProposalDetection);
  if (realDetections.length) {
    return realDetections.slice(0, limit);
  }
  const motionProposals = proposals.filter((detection) => detection.label.includes("motion"));
  if (motionProposals.length) {
    return motionProposals.slice(0, 1);
  }
  const screenProposals = proposals.filter((detection) => detection.label.includes("screen") || detection.label.includes("dark panel"));
  if (screenProposals.length) {
    return screenProposals.slice(0, 1);
  }
  return proposals.slice(0, 1);
}

function safelySetPointerCapture(element: Element, pointerId: number): void {
  if ("setPointerCapture" in element) {
    try {
      element.setPointerCapture(pointerId);
    } catch {
      // Pointer capture can fail if the browser already ended the touch sequence.
    }
  }
}

function safelyReleasePointerCapture(element: Element, pointerId: number): void {
  if ("hasPointerCapture" in element && "releasePointerCapture" in element) {
    try {
      if (element.hasPointerCapture(pointerId)) {
        element.releasePointerCapture(pointerId);
      }
    } catch {
      // Releasing stale capture should never break the capture/analyze flow.
    }
  }
}

function overlayStyleForDetection(detection: LocalDetection, viewport: VideoViewport): CSSProperties {
  return {
    left: `${viewport.x + detection.x * viewport.w}%`,
    top: `${viewport.y + detection.y * viewport.h}%`,
    width: `${detection.w * viewport.w}%`,
    height: `${detection.h * viewport.h}%`,
  };
}

type ProposalCell = {
  x: number;
  y: number;
  weight: number;
};

function componentDetections(cells: ProposalCell[], label: string, baseScore: number, limit: number): LocalDetection[] {
  const byKey = new Map(cells.map((cell) => [`${cell.x}:${cell.y}`, cell]));
  const visited = new Set<string>();
  const components: ProposalCell[][] = [];

  for (const cell of cells) {
    const startKey = `${cell.x}:${cell.y}`;
    if (visited.has(startKey)) {
      continue;
    }
    const stack = [cell];
    const component: ProposalCell[] = [];
    visited.add(startKey);
    while (stack.length) {
      const current = stack.pop();
      if (!current) {
        continue;
      }
      component.push(current);
      for (let ny = current.y - 1; ny <= current.y + 1; ny += 1) {
        for (let nx = current.x - 1; nx <= current.x + 1; nx += 1) {
          if (nx === current.x && ny === current.y) {
            continue;
          }
          const key = `${nx}:${ny}`;
          const next = byKey.get(key);
          if (next && !visited.has(key)) {
            visited.add(key);
            stack.push(next);
          }
        }
      }
    }
    components.push(component);
  }

  return components
    .map((component) => {
      const minX = Math.min(...component.map((cell) => cell.x));
      const minY = Math.min(...component.map((cell) => cell.y));
      const maxX = Math.max(...component.map((cell) => cell.x));
      const maxY = Math.max(...component.map((cell) => cell.y));
      const avgWeight = component.reduce((sum, cell) => sum + cell.weight, 0) / component.length;
      const w = (maxX - minX + 1) / MOTION_GRID_COLUMNS;
      const h = (maxY - minY + 1) / MOTION_GRID_ROWS;
      return {
        label,
        score: Math.max(baseScore, Math.min(0.94, avgWeight)),
        x: Math.max(0, minX / MOTION_GRID_COLUMNS),
        y: Math.max(0, minY / MOTION_GRID_ROWS),
        w: Math.min(1, w),
        h: Math.min(1, h),
      };
    })
    .filter((detection) => detection.w * detection.h >= 0.02 && detection.w <= 0.9 && detection.h <= 0.9)
    .sort((left, right) => right.score * right.w * right.h - left.score * left.w * left.h)
    .slice(0, limit);
}

function proposalOverlap(a: LocalDetection, b: LocalDetection): number {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.w, b.x + b.w);
  const bottom = Math.min(a.y + a.h, b.y + b.h);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const smallerArea = Math.min(a.w * a.h, b.w * b.h);
  return smallerArea > 0 ? intersection / smallerArea : 0;
}

function mergeProposalDetections(detections: LocalDetection[]): LocalDetection[] {
  return detections
    .sort((left, right) => {
      const leftRealBoost = isProposalDetection(left) ? 0 : 0.22;
      const rightRealBoost = isProposalDetection(right) ? 0 : 0.22;
      const leftScreenBoost = left.label.includes("screen") ? 0.12 : 0;
      const rightScreenBoost = right.label.includes("screen") ? 0.12 : 0;
      return right.score + rightRealBoost + rightScreenBoost - (left.score + leftRealBoost + leftScreenBoost);
    })
    .reduce<LocalDetection[]>((merged, detection) => {
      if (merged.some((existing) => proposalOverlap(existing, detection) > 0.74)) {
        return merged;
      }
      return [...merged, detection];
    }, [])
    .slice(0, 2);
}

function buildMotionCandidateBoxes(imageData: ImageData, previous: MotionSnapshot | null): {
  snapshot: MotionSnapshot;
  detections: LocalDetection[];
} {
  const { data, width, height } = imageData;
  const sums = new Uint32Array(MOTION_GRID_COLUMNS * MOTION_GRID_ROWS);
  const cells = new Uint8Array(MOTION_GRID_COLUMNS * MOTION_GRID_ROWS);
  const counts = new Uint16Array(cells.length);
  const minimums = new Uint8Array(cells.length).fill(255);
  const maximums = new Uint8Array(cells.length);

  for (let y = 0; y < height; y += 3) {
    for (let x = 0; x < width; x += 3) {
      const pixelIndex = y * width + x;
      const offset = pixelIndex * 4;
      const lum = Math.round(0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2]);
      const gridX = Math.min(MOTION_GRID_COLUMNS - 1, Math.floor((x / width) * MOTION_GRID_COLUMNS));
      const gridY = Math.min(MOTION_GRID_ROWS - 1, Math.floor((y / height) * MOTION_GRID_ROWS));
      const cellIndex = gridY * MOTION_GRID_COLUMNS + gridX;
      sums[cellIndex] += lum;
      counts[cellIndex] += 1;
      minimums[cellIndex] = Math.min(minimums[cellIndex], lum);
      maximums[cellIndex] = Math.max(maximums[cellIndex], lum);
    }
  }

  for (let index = 0; index < cells.length; index += 1) {
    cells[index] = counts[index] ? Math.round(sums[index] / counts[index]) : 0;
  }

  const snapshot = { width, height, cells };
  const visualCells: Array<{ x: number; y: number; range: number }> = [];
  const darkPanelCells: ProposalCell[] = [];
  for (let index = 0; index < cells.length; index += 1) {
    const range = maximums[index] - minimums[index];
    const x = index % MOTION_GRID_COLUMNS;
    const y = Math.floor(index / MOTION_GRID_COLUMNS);
    if (range >= 42) {
      visualCells.push({
        x,
        y,
        range,
      });
    }
    if (cells[index] <= 94 && range >= 10) {
      darkPanelCells.push({
        x,
        y,
        weight: Math.min(0.94, 0.48 + (94 - cells[index]) / 120 + range / 180),
      });
    }
  }

  const darkPanelDetections = componentDetections(darkPanelCells, "screen / dark panel", 0.62, 3).filter(
    (detection) => detection.w >= 0.18 && detection.h >= 0.12,
  );
  const visualDetections = componentDetections(
    visualCells.map((cell) => ({ x: cell.x, y: cell.y, weight: Math.min(0.9, cell.range / 110) })),
    "edge attention",
    0.52,
    3,
  );
  const staticProposals = mergeProposalDetections([...darkPanelDetections, ...visualDetections]);

  if (!previous || previous.width !== width || previous.height !== height) {
    return { snapshot, detections: staticProposals };
  }

  const changedCells: Array<{ x: number; y: number; diff: number }> = [];
  for (let index = 0; index < cells.length; index += 1) {
    const diff = Math.abs(cells[index] - previous.cells[index]);
    if (diff >= MOTION_CELL_TRIGGER) {
      changedCells.push({
        x: index % MOTION_GRID_COLUMNS,
        y: Math.floor(index / MOTION_GRID_COLUMNS),
        diff,
      });
    }
  }

  if (!changedCells.length) {
    return { snapshot, detections: staticProposals };
  }

  const motionDetections = componentDetections(
    changedCells.map((cell) => ({ x: cell.x, y: cell.y, weight: Math.min(0.94, cell.diff / 90) })),
    "motion region",
    0.58,
    2,
  );
  return {
    snapshot,
    detections: mergeProposalDetections([...motionDetections, ...staticProposals]),
  };
}

function addRecent(items: string[], next: string, limit = 6): string[] {
  const trimmed = next.trim();
  if (!trimmed) {
    return items;
  }
  if (items[0] === trimmed) {
    return items;
  }
  return [trimmed, ...items].slice(0, limit);
}

function waitForVideoData(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Video is still loading. Wait a moment and analyze again."));
    }, 6000);

    function cleanup() {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("error", onError);
    }

    function onReady() {
      cleanup();
      resolve();
    }

    function onError() {
      cleanup();
      reject(new Error("Video could not be decoded by this browser."));
    }

    video.addEventListener("loadeddata", onReady, { once: true });
    video.addEventListener("canplay", onReady, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

function initialAnalysesUsed(): number {
  if (typeof window === "undefined") {
    return 0;
  }
  const stored = window.sessionStorage.getItem("cloud-video-analyzer-analyses-used");
  const parsed = stored ? Number(stored) : 0;
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : 0;
}

export function IndustrialVideoAnalyzer({
  providerStatus,
  roboflowReady,
}: {
  providerStatus: Record<ProviderId, boolean>;
  roboflowReady: boolean;
}) {
  const prefersReducedMotion = useReducedMotion();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoFrameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const liveSocketRef = useRef<WebSocket | null>(null);
  const liveTimerRef = useRef<number | null>(null);
  const liveStreamingStartedRef = useRef(false);
  const liveStartedAtRef = useRef(0);
  const liveCloudBusyRef = useRef(false);
  const liveLastCloudAtRef = useRef(0);
  const previewDetectionFrameRef = useRef<number | null>(null);
  const previewDetectionBusyRef = useRef(false);
  const lastPreviewDetectionAtRef = useRef(0);
  const latestPreviewDetectionsRef = useRef<LocalDetection[]>([]);
  const lastRollingFrameAtRef = useRef(0);
  const lastTransformersDetectionAtRef = useRef(0);
  const motionSnapshotRef = useRef<MotionSnapshot | null>(null);
  const previousFrameRef = useRef<EdgeMetricResult["snapshot"] | null>(null);
  const zoneInteractionRef = useRef<ZoneInteraction | null>(null);
  const holdRecordingRef = useRef<HoldRecording>({
    active: false,
    stopRequested: false,
    frames: [],
    pointerId: null,
    startedAt: 0,
  });
  const analysesUsedRef = useRef(0);

  const [provider, setProvider] = useState<ProviderId>(providerStatus.gemini ? "gemini" : providerStatus.openai ? "openai" : "nvidia");
  const [mode, setMode] = useState<VideoMode>("industrial_general");
  const [source, setSource] = useState<VideoSource>("camera");
  const [objective, setObjective] = useState(DEFAULT_OBJECTIVE);
  const [zones, setZones] = useState<Zone[]>([
    { id: "zone-1", label: "Restricted zone", x: 0.58, y: 0.2, w: 0.3, h: 0.58 },
  ]);
  const [zoneToolOpen, setZoneToolOpen] = useState(false);
  const [status, setStatus] = useState("camera off - no recording");
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [holdRecording, setHoldRecording] = useState(false);
  const [recordingProgress, setRecordingProgress] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>("environment");
  const [sampleUrl, setSampleUrl] = useState(SAMPLE_CLIPS[1].url);
  const [analysis, setAnalysis] = useState<VideoAnalysisResponse | null>(null);
  const [lastFrames, setLastFrames] = useState<SampledFrame[]>([]);
  const [analysesUsed, setAnalysesUsed] = useState(initialAnalysesUsed);
  const [videoOrientation, setVideoOrientation] = useState<VideoOrientation>("landscape-video");
  const [liveStatus, setLiveStatus] = useState("Live edge camera idle");
  const [liveRunning, setLiveRunning] = useState(false);
  const [liveEvents, setLiveEvents] = useState<string[]>([]);
  const [liveTranscript, setLiveTranscript] = useState<string[]>([]);
  const [liveGeminiInsight, setLiveGeminiInsight] = useState<string | null>(null);
  const [liveFrameCount, setLiveFrameCount] = useState(0);
  const [liveUsage, setLiveUsage] = useState({ requests: 0, frames: 0, costUsd: 0, lastLatencyMs: 0 });
  const [detectorStatus, setDetectorStatus] = useState(roboflowReady ? "Roboflow specialist detector ready" : "Browser detector active");
  const [edgeGateSummary, setEdgeGateSummary] = useState<EdgeGateSummary | null>(null);
  const [forceCloudAnalysis, setForceCloudAnalysis] = useState(true);
  const [previewDetections, setPreviewDetections] = useState<LocalDetection[]>([]);
  const [edgePreviewMetrics, setEdgePreviewMetrics] = useState<EdgeMetricResult["metrics"] | null>(null);
  const [edgePreviewFrameCount, setEdgePreviewFrameCount] = useState(0);
  const [edgeDetectionLatencyMs, setEdgeDetectionLatencyMs] = useState(0);
  const [edgeLoopFps, setEdgeLoopFps] = useState(0);
  const [videoViewport, setVideoViewport] = useState<VideoViewport>({ x: 0, y: 0, w: 100, h: 100 });

  useEffect(() => {
    return () => {
      holdRecordingRef.current.active = false;
      holdRecordingRef.current.stopRequested = true;
      if (liveTimerRef.current) {
        window.clearInterval(liveTimerRef.current);
      }
      if (previewDetectionFrameRef.current) {
        window.cancelAnimationFrame(previewDetectionFrameRef.current);
      }
      liveSocketRef.current?.close();
      stopStream(streamRef.current);
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    analysesUsedRef.current = analysesUsed;
  }, [analysesUsed]);

  useEffect(() => {
    let cancelled = false;
    const warmDetector = () => {
      setDetectorStatus((current) => (current.includes("active") ? current : "Warming browser edge detector"));
      void tryCreateObjectDetector().then((runtime) => {
        if (!cancelled) {
          setDetectorStatus(runtime.label);
        }
      });
    };

    const timerId = window.setTimeout(warmDetector, 400);
    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, []);

  const refreshDevices = useCallback(async () => {
    const nextDevices = await listVideoInputDevices();
    setDevices(nextDevices);
    if (!selectedDeviceId && nextDevices[0]?.deviceId) {
      setSelectedDeviceId(nextDevices[0].deviceId);
    }
  }, [selectedDeviceId]);

  const clearVideoObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  const updateVideoViewport = useCallback(() => {
    const video = videoRef.current;
    const frame = videoFrameRef.current;
    if (!video?.videoWidth || !video.videoHeight || !frame) {
      setVideoViewport({ x: 0, y: 0, w: 100, h: 100 });
      return;
    }
    const frameRect = frame.getBoundingClientRect();
    if (!frameRect.width || !frameRect.height) {
      return;
    }
    const scale = Math.min(frameRect.width / video.videoWidth, frameRect.height / video.videoHeight);
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    setVideoViewport({
      x: ((frameRect.width - renderedWidth) / 2 / frameRect.width) * 100,
      y: ((frameRect.height - renderedHeight) / 2 / frameRect.height) * 100,
      w: (renderedWidth / frameRect.width) * 100,
      h: (renderedHeight / frameRect.height) * 100,
    });
  }, []);

  const updateVideoOrientation = useCallback(() => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video.videoHeight) {
      return;
    }
    setVideoOrientation(video.videoHeight > video.videoWidth ? "portrait-video" : "landscape-video");
    updateVideoViewport();
  }, [updateVideoViewport]);

  useEffect(() => {
    updateVideoViewport();
    window.addEventListener("resize", updateVideoViewport);
    return () => window.removeEventListener("resize", updateVideoViewport);
  }, [updateVideoViewport]);

  const warmVideoElement = useCallback(async () => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    video.preload = "auto";
    video.load();
    await video.play().catch(() => undefined);
    await waitForVideoData(video).catch(() => undefined);
    video.pause();
    updateVideoOrientation();
  }, [updateVideoOrientation]);

  const startCamera = useCallback(async (nextFacing: CameraFacing = cameraFacing, deviceId: string | null = selectedDeviceId) => {
    setError(null);
    setCameraFacing(nextFacing);
    setStatus(`requesting ${nextFacing === "environment" ? "back" : "front"} camera permission`);
    setSource("camera");
    try {
      clearVideoObjectUrl();
      stopStream(streamRef.current);
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(buildVideoConstraints("webcam_coach", deviceId, nextFacing));
      } catch (cameraError) {
        if (!deviceId) {
          throw cameraError;
        }
        setSelectedDeviceId(null);
        stream = await navigator.mediaDevices.getUserMedia(buildVideoConstraints("webcam_coach", null, nextFacing));
      }
      streamRef.current = stream;
      if (!videoRef.current) {
        throw new Error("Video element is not ready.");
      }
      videoRef.current.srcObject = stream;
      videoRef.current.controls = false;
      await videoRef.current.play();
      updateVideoOrientation();
      await refreshDevices();
      previousFrameRef.current = null;
      motionSnapshotRef.current = null;
      setEdgePreviewMetrics(null);
      setEdgePreviewFrameCount(0);
      setEdgeDetectionLatencyMs(0);
      setEdgeLoopFps(0);
      setRunning(true);
      setStatus(`preview only - ${nextFacing === "environment" ? "back" : "front"} camera - no cloud upload`);
    } catch (startError) {
      setRunning(false);
      setStatus("camera off - no recording");
      setError(readableError(startError));
    }
  }, [cameraFacing, clearVideoObjectUrl, refreshDevices, selectedDeviceId, updateVideoOrientation]);

  const stopCamera = useCallback(() => {
    if (liveTimerRef.current) {
      window.clearInterval(liveTimerRef.current);
      liveTimerRef.current = null;
    }
    liveSocketRef.current?.close();
    liveSocketRef.current = null;
    liveStreamingStartedRef.current = false;
    liveStartedAtRef.current = 0;
    setLiveRunning(false);
    setLiveStatus("Live edge camera stopped with camera");
    holdRecordingRef.current.active = false;
    holdRecordingRef.current.stopRequested = true;
    setHoldRecording(false);
    setRecordingProgress(0);
    setRunning(false);
    setPreviewDetections([]);
    latestPreviewDetectionsRef.current = [];
    motionSnapshotRef.current = null;
    setEdgePreviewMetrics(null);
    setEdgePreviewFrameCount(0);
    setEdgeDetectionLatencyMs(0);
    setEdgeLoopFps(0);
    setStatus("camera off - no recording");
    stopStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    if (previewDetectionFrameRef.current) {
      window.cancelAnimationFrame(previewDetectionFrameRef.current);
      previewDetectionFrameRef.current = null;
    }
    previewDetectionBusyRef.current = false;

    if (source !== "camera" || !running) {
      latestPreviewDetectionsRef.current = [];
      motionSnapshotRef.current = null;
      lastRollingFrameAtRef.current = 0;
      return;
    }

    let cancelled = false;
    lastPreviewDetectionAtRef.current = 0;
    lastRollingFrameAtRef.current = 0;
    lastTransformersDetectionAtRef.current = 0;

  const runPreviewDetection = async (now: number) => {
      if (previewDetectionBusyRef.current) {
        return;
      }
      if (holdRecordingRef.current.active) {
        return;
      }
      const video = videoRef.current;
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return;
      }
      previewDetectionBusyRef.current = true;
      const detectionStartedAt = performance.now();
      let motionDetections: LocalDetection[] = [];
      let edgeMetricsForFrame: EdgeMetricResult["metrics"] | null = null;
      let frameDataUrl: string | null = null;
      try {
        const canvas = canvasRef.current;
        if (canvas) {
          const imageData = captureVideoFrame(video, canvas, 360);
          if (imageData) {
            const result = computeEdgeMetrics(imageData, previousFrameRef.current);
            const motionResult = buildMotionCandidateBoxes(imageData, motionSnapshotRef.current);
            previousFrameRef.current = result.snapshot;
            motionSnapshotRef.current = motionResult.snapshot;
            edgeMetricsForFrame = result.metrics;
            frameDataUrl = canvasToJpegDataUrl(canvas, JPEG_QUALITY);
            motionDetections = motionResult.detections;
            if (!cancelled) {
              setEdgePreviewMetrics(result.metrics);
              setEdgePreviewFrameCount((count) => count + 1);
              if (motionDetections.length) {
                setPreviewDetections(motionDetections);
                latestPreviewDetectionsRef.current = motionDetections;
                updateVideoViewport();
              } else if (!getObjectDetectorRuntime().ready) {
                setPreviewDetections([]);
                latestPreviewDetectionsRef.current = [];
              }
            }
          }
        }
        const runtime = getObjectDetectorRuntime();
        if (!runtime.ready) {
          void tryCreateObjectDetector().then((nextRuntime) => {
            if (!cancelled) {
              setDetectorStatus(nextRuntime.label);
            }
          });
          if (!cancelled) {
            setDetectorStatus(motionDetections.length ? "Instant motion edge active; object model warming" : runtime.label);
            setEdgeDetectionLatencyMs(Math.round(performance.now() - detectionStartedAt));
            if (edgeMetricsForFrame && frameDataUrl && now - lastRollingFrameAtRef.current >= 650) {
              lastRollingFrameAtRef.current = now;
              const imageDataUrl = frameDataUrl;
              const edgeMetrics = edgeMetricsForFrame;
              setLastFrames((frames) =>
                [
                  ...frames,
                  {
                    imageDataUrl,
                    timestampMs: now,
                    edgeMetrics,
                    localDetections: motionDetections,
                  },
                ].slice(-MAX_FRAMES),
              );
            }
          }
          return;
        }
        const shouldRunTransformers = source !== "camera" && Boolean(frameDataUrl) && now - lastTransformersDetectionAtRef.current >= TRANSFORMERS_LIVE_INTERVAL_MS;
        if (shouldRunTransformers) {
          lastTransformersDetectionAtRef.current = now;
        }
        setDetectorStatus(runtime.label);
        const [mediaPipeDetections, transformerDetections] = await Promise.all([
          detectObjectsForVideo(video, now),
          shouldRunTransformers && frameDataUrl ? detectObjectsWithTransformers(frameDataUrl, 900) : Promise.resolve([]),
        ]);
        if (!cancelled) {
          const transformersRuntime = getTransformersDetectorRuntime();
          const nextDetections = mergeProposalDetections([
            ...mediaPipeDetections,
            ...transformerDetections,
            ...motionDetections,
          ]);
          setPreviewDetections(nextDetections);
          latestPreviewDetectionsRef.current = nextDetections;
          setEdgeDetectionLatencyMs(Math.round(performance.now() - detectionStartedAt));
          setDetectorStatus(
            transformerDetections.length
              ? `${runtime.label}; ${transformersRuntime.label}`
              : transformersRuntime.loading
                ? `${runtime.label}; Transformers.js warming`
                : runtime.label,
          );
          if (edgeMetricsForFrame && frameDataUrl && now - lastRollingFrameAtRef.current >= 650) {
            lastRollingFrameAtRef.current = now;
            const imageDataUrl = frameDataUrl;
            const edgeMetrics = edgeMetricsForFrame;
            setLastFrames((frames) =>
              [
                ...frames,
                {
                  imageDataUrl,
                  timestampMs: now,
                  edgeMetrics,
                  localDetections: nextDetections,
                },
              ].slice(-MAX_FRAMES),
            );
          }
          updateVideoViewport();
        }
      } catch {
        if (!cancelled) {
          if (motionDetections.length) {
            setPreviewDetections(motionDetections);
            latestPreviewDetectionsRef.current = motionDetections;
            if (edgeMetricsForFrame && frameDataUrl && now - lastRollingFrameAtRef.current >= 650) {
              lastRollingFrameAtRef.current = now;
              const imageDataUrl = frameDataUrl;
              const edgeMetrics = edgeMetricsForFrame;
              setLastFrames((frames) =>
                [
                  ...frames,
                  {
                    imageDataUrl,
                    timestampMs: now,
                    edgeMetrics,
                    localDetections: motionDetections,
                  },
                ].slice(-MAX_FRAMES),
              );
            }
            updateVideoViewport();
          }
          setDetectorStatus("Motion edge active; object model still loading");
        }
      } finally {
        previewDetectionBusyRef.current = false;
      }
    };

    const tick = (now: number) => {
      if (cancelled) {
        return;
      }
      const elapsedSinceDetection = now - lastPreviewDetectionAtRef.current;
      if (!previewDetectionBusyRef.current && elapsedSinceDetection >= PREVIEW_DETECTION_INTERVAL_MS) {
        const previousDetectionAt = lastPreviewDetectionAtRef.current;
        lastPreviewDetectionAtRef.current = now;
        void runPreviewDetection(now).finally(() => {
          if (!cancelled) {
            setEdgeLoopFps(previousDetectionAt ? Math.round(1000 / Math.max(1, now - previousDetectionAt)) : 0);
            previewDetectionFrameRef.current = window.requestAnimationFrame(tick);
          }
        });
        return;
      }
      previewDetectionFrameRef.current = window.requestAnimationFrame(tick);
    };

    previewDetectionFrameRef.current = window.requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      if (previewDetectionFrameRef.current) {
        window.cancelAnimationFrame(previewDetectionFrameRef.current);
        previewDetectionFrameRef.current = null;
      }
      previewDetectionBusyRef.current = false;
    };
  }, [running, source, updateVideoViewport]);

  const switchCamera = useCallback(async () => {
    const nextFacing: CameraFacing = cameraFacing === "environment" ? "user" : "environment";
    setSelectedDeviceId(null);
    setStatus(`switching to ${nextFacing === "environment" ? "back" : "front"} camera`);
    if (running) {
      await startCamera(nextFacing, null);
      return;
    }
    setCameraFacing(nextFacing);
    setStatus(`camera off - ${nextFacing === "environment" ? "back" : "front"} camera selected`);
  }, [cameraFacing, running, startCamera]);

  const loadFile = useCallback(
    async (file: File) => {
      setError(null);
      setStatus("loading local video");
      stopStream(streamRef.current);
      streamRef.current = null;
      clearVideoObjectUrl();
      const objectUrl = URL.createObjectURL(file);
      objectUrlRef.current = objectUrl;
      setSource("file");
      setPreviewDetections([]);
      latestPreviewDetectionsRef.current = [];
      motionSnapshotRef.current = null;
      setEdgePreviewMetrics(null);
      setEdgePreviewFrameCount(0);
      setEdgeDetectionLatencyMs(0);
      setEdgeLoopFps(0);
      if (videoRef.current) {
        videoRef.current.crossOrigin = "";
        videoRef.current.srcObject = null;
        videoRef.current.src = objectUrl;
        videoRef.current.controls = true;
        videoRef.current.muted = true;
        await warmVideoElement();
      }
      setRunning(false);
      setStatus("local video ready");
    },
    [clearVideoObjectUrl, warmVideoElement],
  );

  const loadSampleUrl = useCallback(async () => {
    if (!sampleUrl.trim()) {
      setError("Paste a public MP4 URL or upload a local clip.");
      return;
    }
    setError(null);
    stopStream(streamRef.current);
    streamRef.current = null;
    clearVideoObjectUrl();
    setSource("sample");
    setPreviewDetections([]);
    latestPreviewDetectionsRef.current = [];
    motionSnapshotRef.current = null;
    setEdgePreviewMetrics(null);
    setEdgePreviewFrameCount(0);
    setEdgeDetectionLatencyMs(0);
    setEdgeLoopFps(0);
    if (videoRef.current) {
      videoRef.current.crossOrigin = "anonymous";
      videoRef.current.srcObject = null;
      videoRef.current.src = sampleUrl.trim();
      videoRef.current.controls = true;
      videoRef.current.muted = true;
      await warmVideoElement();
    }
    setRunning(false);
    setStatus("sample url loaded; press play or analyze");
  }, [clearVideoObjectUrl, sampleUrl, warmVideoElement]);

  const loadSampleClip = useCallback(
    async (clip: (typeof SAMPLE_CLIPS)[number]) => {
      setSampleUrl(clip.url);
      setMode(clip.mode);
      setObjective(clip.objective);
      setError(null);
      stopStream(streamRef.current);
      streamRef.current = null;
      clearVideoObjectUrl();
      setSource("sample");
      setPreviewDetections([]);
      latestPreviewDetectionsRef.current = [];
      motionSnapshotRef.current = null;
      setEdgePreviewMetrics(null);
      setEdgePreviewFrameCount(0);
      setEdgeDetectionLatencyMs(0);
      setEdgeLoopFps(0);
      if (videoRef.current) {
        videoRef.current.crossOrigin = "anonymous";
        videoRef.current.srcObject = null;
        videoRef.current.src = clip.url;
        videoRef.current.controls = true;
        videoRef.current.muted = true;
        await warmVideoElement();
      }
      setRunning(false);
      setStatus(`${clip.label.toLowerCase()} sample ready`);
    },
    [clearVideoObjectUrl, warmVideoElement],
  );

  const captureOneFrame = useCallback(async (timestampMs: number, options: { lightweight?: boolean } = {}): Promise<SampledFrame | null> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const imageData = captureVideoFrame(video, canvas, options.lightweight ? 360 : 640);
    if (!imageData) {
      return null;
    }

    const result = computeEdgeMetrics(imageData, previousFrameRef.current);
    const proposalResult = buildMotionCandidateBoxes(imageData, motionSnapshotRef.current);
    previousFrameRef.current = result.snapshot;
    motionSnapshotRef.current = proposalResult.snapshot;
    setEdgePreviewMetrics(result.metrics);
    setEdgePreviewFrameCount((count) => count + 1);

    let localDetections: LocalDetection[] = [];
    if (source === "camera") {
      try {
        const mediaPipeDetections = options.lightweight ? [] : await detectObjectsForVideo(video, performance.now());
        localDetections = mergeProposalDetections([
          ...mediaPipeDetections,
          ...proposalResult.detections,
          ...latestPreviewDetectionsRef.current,
        ]);
      } catch {
        localDetections = mergeProposalDetections([
          ...proposalResult.detections,
          ...latestPreviewDetectionsRef.current,
        ]);
      }
      setPreviewDetections(localDetections);
      latestPreviewDetectionsRef.current = localDetections;
      updateVideoViewport();
    } else {
      try {
        const imageDataUrl = canvasToJpegDataUrl(canvas, JPEG_QUALITY);
        const [mediaPipeDetections, transformerDetections] = await Promise.all([
          detectObjectsForVideo(video, performance.now()),
          options.lightweight ? Promise.resolve([]) : detectObjectsWithTransformers(imageDataUrl, 2500),
        ]);
        localDetections = mergeProposalDetections([
          ...mediaPipeDetections,
          ...transformerDetections,
          ...proposalResult.detections,
        ]);
        setPreviewDetections(localDetections);
        latestPreviewDetectionsRef.current = localDetections;
        updateVideoViewport();
      } catch {
        localDetections = proposalResult.detections;
        setPreviewDetections(localDetections);
        latestPreviewDetectionsRef.current = localDetections;
        updateVideoViewport();
      }
    }

    const frame = {
      imageDataUrl: canvasToJpegDataUrl(canvas, JPEG_QUALITY),
      timestampMs,
      edgeMetrics: result.metrics,
      localDetections,
    };
    return frame;
  }, [source, updateVideoViewport]);

  const seekVideo = useCallback((seconds: number) => {
    return new Promise<void>((resolve, reject) => {
      const video = videoRef.current;
      if (!video) {
        reject(new Error("Video element is not ready."));
        return;
      }
      const timeout = window.setTimeout(() => {
        video.removeEventListener("seeked", onSeeked);
        reject(new Error("Timed out while sampling video."));
      }, 4000);
      function onSeeked() {
        window.clearTimeout(timeout);
        video?.removeEventListener("seeked", onSeeked);
        resolve();
      }
      video.addEventListener("seeked", onSeeked);
      video.currentTime = seconds;
    });
  }, []);

  const sampleFrames = useCallback(async () => {
    const video = videoRef.current;
    if (!video) {
      throw new Error("Video element is not ready.");
    }
    setStatus("waiting for video frame");
    await waitForVideoData(video);
    const detectorRuntime = await tryCreateObjectDetector();
    setDetectorStatus(detectorRuntime.label);
    previousFrameRef.current = null;
    motionSnapshotRef.current = null;
    const frames: SampledFrame[] = [];

    if (source === "camera" || !Number.isFinite(video.duration) || video.duration <= 0) {
      const frameIntervalMs = CAMERA_CAPTURE_WINDOW_MS / MAX_FRAMES;
      for (let i = 0; i < MAX_FRAMES; i += 1) {
        setStatus(`capturing local ${Math.round(CAMERA_CAPTURE_WINDOW_MS / 1000)}s burst - frame ${i + 1} of ${MAX_FRAMES}`);
        const frame = await captureOneFrame(i * frameIntervalMs);
        if (frame) {
          frames.push(frame);
        }
        await new Promise((resolve) => window.setTimeout(resolve, frameIntervalMs));
      }
    } else {
      const duration = Math.max(0.5, video.duration);
      for (let i = 0; i < MAX_FRAMES; i += 1) {
        setStatus(`extracting keyframe ${i + 1} of ${MAX_FRAMES}`);
        const seconds = Math.min(duration - 0.05, (duration * (i + 1)) / (MAX_FRAMES + 1));
        await seekVideo(Math.max(0, seconds));
        const frame = await captureOneFrame(seconds * 1000);
        if (frame) {
          frames.push(frame);
        }
      }
    }

    if (frames.length === 0) {
      throw new Error("No frames could be sampled. If this is an internet video, CORS may block browser preprocessing. Upload the clip locally.");
    }
    setStatus(`sampled ${frames.length} keyframes`);
    return frames;
  }, [captureOneFrame, seekVideo, source]);

  const enrichFramesWithRoboflow = useCallback(async (frames: SampledFrame[]): Promise<SampledFrame[]> => {
    if (!roboflowReady || !frames.length) {
      setDetectorStatus(roboflowReady ? "Roboflow specialist detector ready" : "Browser detector active");
      return frames;
    }

    setDetectorStatus("Roboflow checking selected frames");
    try {
      const enrichedFrames = await Promise.all(
        frames.map(async (frame) => {
          const response = await fetch("/api/roboflow-detect", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              imageDataUrl: frame.imageDataUrl,
              timestampMs: frame.timestampMs,
            }),
          });
          const payload: unknown = await response.json();
          if (!response.ok || !payload || typeof payload !== "object") {
            return frame;
          }
          const roboflowPayload = payload as RoboflowDetectResponse;
          if (!roboflowPayload.configured) {
            setDetectorStatus("Browser detector active");
            return frame;
          }
          const detections = Array.isArray(roboflowPayload.detections) ? roboflowPayload.detections.filter(isLocalDetection) : [];
          return {
            ...frame,
            localDetections: mergeDetections(frame.localDetections, detections),
          };
        }),
      );
      const addedDetections = enrichedFrames.reduce((total, frame, index) => {
        return total + Math.max(0, frame.localDetections.length - (frames[index]?.localDetections.length ?? 0));
      }, 0);
      setDetectorStatus(addedDetections ? `Roboflow added ${addedDetections} object box${addedDetections === 1 ? "" : "es"}` : "Roboflow checked frames");
      return enrichedFrames;
    } catch {
      setDetectorStatus("Browser detector active");
      return frames;
    }
  }, [roboflowReady]);

  const submitFramesForAnalysis = useCallback(async (frames: SampledFrame[]) => {
    if (!providerStatus[provider]) {
      setError(`${provider.toUpperCase()} API key is not configured on the server.`);
      return false;
    }
    if (analysesUsedRef.current >= DEMO_ANALYSIS_LIMIT) {
      setError("Demo analysis limit reached for this browser session.");
      return false;
    }
    if (!frames.length) {
      setError("No usable frames were captured. Hold the record button while pointing at the scene.");
      return false;
    }

    setError(null);
    try {
      const edgeSelection = selectEdgeTriggeredFrames(frames, MAX_CLOUD_FRAMES, { allowFallback: forceCloudAnalysis });
      setEdgeGateSummary(edgeSelection.summary);
      setLastFrames(frames);
      if (!edgeSelection.frames.length) {
        setStatus("edge gate held cloud request - no object or motion trigger");
        setError("No cloud request sent: the browser did not detect a usable object or motion trigger. Turn on force cloud analysis to send the best frames anyway.");
        return false;
      }
      const cloudFrames = await enrichFramesWithRoboflow(edgeSelection.frames);
      setLastFrames(cloudFrames);
      setStatus(`edge gate selected ${cloudFrames.length} of ${edgeSelection.summary.inputFrames} frames; sending to ${provider}`);
      const buildRequestBody = (selectedFrames: SampledFrame[]) => JSON.stringify({
        provider,
        mode,
        source,
        objective,
        zones,
        frames: selectedFrames,
        sampling: {
          requestedFps: 1,
          maxFrames: MAX_CLOUD_FRAMES,
          jpegQuality: JPEG_QUALITY,
          payloadBytes: payloadBytes(selectedFrames),
          forceCloud: forceCloudAnalysis,
          edgeGate: {
            strategy: edgeSelection.summary.strategy,
            inputFrames: edgeSelection.summary.inputFrames,
            selectedFrames: selectedFrames.length,
            skippedFrames: Math.max(0, edgeSelection.summary.inputFrames - selectedFrames.length),
            objectFrames: edgeSelection.summary.objectFrames,
            motionFrames: edgeSelection.summary.motionFrames,
            staticFrames: edgeSelection.summary.staticFrames,
          },
        },
      });
      let response = await fetch("/api/analyze-video", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: buildRequestBody(cloudFrames),
      });

      let payload: unknown = await response.json();
      if (!response.ok && cloudFrames.length > 1) {
        const retryFrames = cloudFrames.slice(0, 1);
        setStatus(`cloud retry with strongest edge frame`);
        response = await fetch("/api/analyze-video", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: buildRequestBody(retryFrames),
        });
        payload = await response.json();
      }
      if (!response.ok) {
        if (isApiError(payload)) {
          throw new Error(payload.detail ?? payload.error ?? "Analysis route failed.");
        }
        throw new Error("Analysis route failed.");
      }

      const nextAnalysis = payload as VideoAnalysisResponse;
      const nextUsage = analysesUsedRef.current + 1;
      analysesUsedRef.current = nextUsage;
      setAnalysesUsed(nextUsage);
      window.sessionStorage.setItem("cloud-video-analyzer-analyses-used", String(nextUsage));
      setAnalysis(nextAnalysis);
      setError(null);
      setStatus(`${provider} analysis complete in ${nextAnalysis.usage.latencyMs}ms`);
      return true;
    } catch (analysisError) {
      setError(readableError(analysisError));
      setStatus("analysis error");
      return false;
    }
  }, [enrichFramesWithRoboflow, forceCloudAnalysis, mode, objective, provider, providerStatus, source, zones]);

  const analyze = useCallback(async () => {
    setAnalyzing(true);
    setError(null);
    setAnalysis(null);
    setEdgeGateSummary(null);
    setStatus("extracting relevant frames in browser");
    try {
      const frames = await sampleFrames();
      return await submitFramesForAnalysis(frames);
    } finally {
      setAnalyzing(false);
    }
  }, [sampleFrames, submitFramesForAnalysis]);

  const sendLatestFramesToCloud = useCallback(async () => {
    if (analyzing || holdRecordingRef.current.active) {
      return false;
    }
    let framesForCloud = lastFrames;
    if (source === "camera" && running) {
      const freshFrame = await captureOneFrame(performance.now());
      if (freshFrame) {
        framesForCloud = [...lastFrames, freshFrame].slice(-MAX_FRAMES);
        setLastFrames(framesForCloud);
      }
    }
    if (!framesForCloud.length) {
      setError("Record a local edge burst first. The cloud button sends only selected edge frames, not the live stream.");
      setStatus("cloud send blocked - no local edge frames");
      return false;
    }
    setAnalyzing(true);
    setError(null);
    setStatus("sending selected edge frames to cloud");
    try {
      return await submitFramesForAnalysis(framesForCloud);
    } finally {
      setAnalyzing(false);
    }
  }, [analyzing, captureOneFrame, lastFrames, running, source, submitFramesForAnalysis]);

  const beginHoldRecording = useCallback(async (event?: PointerEvent<HTMLButtonElement>) => {
    if (source !== "camera" || analyzing || holdRecordingRef.current.active) {
      return;
    }
    if (event) {
      safelySetPointerCapture(event.currentTarget, event.pointerId);
    }
    setError(null);
    setAnalysis(null);
    setEdgeGateSummary(null);
    setEdgePreviewFrameCount(0);
    setRecordingProgress(0);
    setAnalyzing(true);
    setHoldRecording(true);
    holdRecordingRef.current = {
      active: true,
      stopRequested: false,
      frames: [],
      pointerId: event?.pointerId ?? null,
      startedAt: performance.now(),
    };

    try {
      if (!running) {
        await startCamera();
        await new Promise((resolve) => window.setTimeout(resolve, 450));
      }

      const video = videoRef.current;
      if (!video) {
        throw new Error("Video element is not ready.");
      }
      await waitForVideoData(video);
      void tryCreateObjectDetector().then((detectorRuntime) => setDetectorStatus(detectorRuntime.label));
      previousFrameRef.current = null;
      setStatus("smart recording - edge boxes live");

      while (holdRecordingRef.current.active && !holdRecordingRef.current.stopRequested) {
        const elapsedMs = Math.max(0, performance.now() - holdRecordingRef.current.startedAt);
        const progress = Math.min(1, elapsedMs / SMART_RECORDING_MAX_MS);
        setRecordingProgress(progress);
        const frame = await captureOneFrame(elapsedMs, { lightweight: true });
        if (frame) {
          const nextFrames = [...holdRecordingRef.current.frames, frame].slice(-MAX_FRAMES);
          holdRecordingRef.current.frames = nextFrames;
          setLastFrames(nextFrames);
          const edgeSelection = selectEdgeTriggeredFrames(nextFrames, MAX_CLOUD_FRAMES, { allowFallback: false });
          const triggerSeen = edgeSelection.summary.objectFrames > 0 || edgeSelection.summary.motionFrames > 0;
          const triggerLabel = edgeSelection.summary.objectFrames > 0 ? "object trigger" : edgeSelection.summary.motionFrames > 0 ? "motion trigger" : "watching";
          setStatus(`smart recording - ${triggerLabel} - ${nextFrames.length} edge frame${nextFrames.length === 1 ? "" : "s"}`);
          if (elapsedMs >= SMART_RECORDING_MIN_MS && triggerSeen && nextFrames.length >= 2) {
            holdRecordingRef.current.stopRequested = true;
          }
        }
        if (elapsedMs >= SMART_RECORDING_MAX_MS) {
          holdRecordingRef.current.stopRequested = true;
        }
        await new Promise((resolve) => window.setTimeout(resolve, SMART_RECORDING_INTERVAL_MS));
      }

      if (!holdRecordingRef.current.frames.length) {
        const frame = await captureOneFrame(Math.max(0, performance.now() - holdRecordingRef.current.startedAt), { lightweight: true });
        if (frame) {
          holdRecordingRef.current.frames = [frame];
          setLastFrames([frame]);
        }
      }

      const frames = holdRecordingRef.current.frames;
      const edgeSelection = selectEdgeTriggeredFrames(frames, MAX_CLOUD_FRAMES, { allowFallback: forceCloudAnalysis });
      holdRecordingRef.current.active = false;
      holdRecordingRef.current.stopRequested = false;
      holdRecordingRef.current.pointerId = null;
      setHoldRecording(false);
      setRecordingProgress(1);
      setEdgeGateSummary(edgeSelection.summary);
      setStatus(`local edge burst ready - ${frames.length} frame${frames.length === 1 ? "" : "s"} captured, ${edgeSelection.summary.selectedFrames} would go to cloud`);
    } catch (recordError) {
      holdRecordingRef.current.active = false;
      holdRecordingRef.current.stopRequested = false;
      holdRecordingRef.current.pointerId = null;
      setHoldRecording(false);
      setRecordingProgress(0);
      setError(readableError(recordError));
      setStatus("analysis error");
    } finally {
      setAnalyzing(false);
      setRecordingProgress(0);
    }
  }, [analyzing, captureOneFrame, forceCloudAnalysis, running, source, startCamera]);

  const finishHoldRecording = useCallback((event?: PointerEvent<HTMLButtonElement>) => {
    if (event) {
      safelyReleasePointerCapture(event.currentTarget, event.pointerId);
    }
    if (!holdRecordingRef.current.active) {
      return;
    }
    holdRecordingRef.current.stopRequested = true;
    holdRecordingRef.current.pointerId = null;
    setStatus("recording stopped - local edge frames ready");
  }, []);

  const recordOrAnalyze = useCallback(async () => {
    if (source === "camera") {
      if (!running) {
        await startCamera();
      }
      return;
    }
    await analyze();
  }, [analyze, running, source, startCamera]);

  const reset = useCallback(() => {
    holdRecordingRef.current.active = false;
    holdRecordingRef.current.stopRequested = true;
    setHoldRecording(false);
    setRecordingProgress(0);
    setError(null);
    setAnalysis(null);
    setLastFrames([]);
    setPreviewDetections([]);
    latestPreviewDetectionsRef.current = [];
    motionSnapshotRef.current = null;
    setEdgePreviewMetrics(null);
    setEdgePreviewFrameCount(0);
    setEdgeDetectionLatencyMs(0);
    setEdgeLoopFps(0);
    setEdgeGateSummary(null);
    setStatus(running ? `preview only - ${cameraFacing === "environment" ? "back" : "front"} camera - no cloud upload` : "camera off - no recording");
    previousFrameRef.current = null;
  }, [cameraFacing, running]);

  const resetLimit = useCallback(() => {
    analysesUsedRef.current = 0;
    setAnalysesUsed(0);
    window.sessionStorage.removeItem("cloud-video-analyzer-analyses-used");
    setError(null);
  }, []);

  const stopGeminiLive = useCallback(() => {
    if (liveTimerRef.current) {
      window.clearInterval(liveTimerRef.current);
      liveTimerRef.current = null;
    }
    liveSocketRef.current?.close();
    liveSocketRef.current = null;
    liveStreamingStartedRef.current = false;
    liveStartedAtRef.current = 0;
    liveCloudBusyRef.current = false;
    liveLastCloudAtRef.current = 0;
    setLiveRunning(false);
    setLiveStatus("Live edge camera stopped");
  }, []);

  const startGeminiLive = useCallback(async () => {
    const startLocalLiveStream = (reason: string) => {
      if (liveTimerRef.current) {
        window.clearInterval(liveTimerRef.current);
        liveTimerRef.current = null;
      }
      liveStreamingStartedRef.current = true;
      liveStartedAtRef.current = performance.now();
      liveCloudBusyRef.current = false;
      liveLastCloudAtRef.current = 0;
      setLiveRunning(true);
      setLiveStatus("Gemini live commentary active");
      setStatus("Live camera analytics active - local edge plus periodic Gemini commentary");
      setLiveTranscript((items) => addRecent(items, `${reason} Local edge boxes update continuously; Gemini receives selected snapshots for live commentary.`));

      const sendLocalFrame = async () => {
        if (!liveStreamingStartedRef.current) {
          return;
        }
        const now = performance.now();
        const frame = await captureOneFrame(Math.max(0, now - liveStartedAtRef.current), { lightweight: true });
        if (!frame) {
          return;
        }
        setLastFrames((frames) => [...frames, frame].slice(-MAX_FRAMES));
        setLiveFrameCount((count) => count + 1);
        const labels = frame.localDetections.length
          ? frame.localDetections.slice(0, 3).map((detection) => `${detection.label} ${Math.round(detection.score * 100)}%`).join(", ")
          : "no object boxes";
        setLiveTranscript((items) =>
          addRecent(
            items,
            `Edge frame: ${labels}; motion ${Math.round(frame.edgeMetrics.motionScore)}; ${frame.edgeMetrics.usable ? "usable for cloud" : frame.edgeMetrics.rejectionReason ?? "low quality"}.`,
          ),
        );

        const cloudIntervalMs = 4200;
        if (providerStatus.gemini && !liveCloudBusyRef.current && now - liveLastCloudAtRef.current >= cloudIntervalMs) {
          liveCloudBusyRef.current = true;
          liveLastCloudAtRef.current = now;
          setLiveStatus("Gemini reading latest snapshot");
          try {
            const response = await fetch("/api/analyze-video", {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({
                provider: "gemini",
                mode,
                source: "camera",
                objective: `Live camera commentary. In one short sentence, describe what changed or what is important now. User objective: ${objective}`,
                zones,
                frames: [frame],
                sampling: {
                  requestedFps: 1,
                  maxFrames: 1,
                  jpegQuality: JPEG_QUALITY,
                  payloadBytes: payloadBytes([frame]),
                  forceCloud: true,
                  edgeGate: {
                    strategy: "live snapshot",
                    inputFrames: 1,
                    selectedFrames: 1,
                    skippedFrames: 0,
                    objectFrames: frame.localDetections.length ? 1 : 0,
                    motionFrames: frame.edgeMetrics.motionScore >= EDGE_MOTION_TRIGGER_SCORE ? 1 : 0,
                    staticFrames: frame.localDetections.length || frame.edgeMetrics.motionScore >= EDGE_MOTION_TRIGGER_SCORE ? 0 : 1,
                  },
                },
              }),
            });
            const payload = await response.json().catch(() => null);
            if (response.ok && payload?.commentary) {
              const nextAnalysis = payload as VideoAnalysisResponse;
              setAnalysis(nextAnalysis);
              setLiveStatus(`Gemini live response - ${nextAnalysis.usage.latencyMs}ms`);
              setLiveGeminiInsight(`Gemini: ${nextAnalysis.commentary}`);
              setLiveTranscript((items) => addRecent(items, `Gemini: ${nextAnalysis.commentary}`, 5));
              setLiveUsage((usage) => ({
                requests: usage.requests + 1,
                frames: usage.frames + nextAnalysis.edgeAssessment.framesAnalyzed,
                costUsd: usage.costUsd + nextAnalysis.usage.estimatedCostUsd,
                lastLatencyMs: nextAnalysis.usage.latencyMs,
              }));
              setLiveEvents((items) => addRecent(items, `${nextAnalysis.edgeAssessment.framesAnalyzed} frame / ${formatCost(nextAnalysis.usage.estimatedCostUsd)}`, 5));
            } else {
              setLiveStatus("Gemini live snapshot skipped");
              if (payload?.detail || payload?.error) {
                setLiveEvents((items) => addRecent(items, `Cloud skipped: ${payload.detail ?? payload.error}`, 5));
              }
            }
          } catch {
            setLiveStatus("Gemini live retrying");
            setLiveEvents((items) => addRecent(items, "Cloud commentary retrying on next snapshot", 5));
          } finally {
            liveCloudBusyRef.current = false;
          }
        }
      };

      void sendLocalFrame();
      liveTimerRef.current = window.setInterval(() => {
        void sendLocalFrame();
      }, 1000);
    };

    setError(null);
    setLiveTranscript([]);
    setLiveGeminiInsight(null);
    setLiveEvents([]);
    setLiveFrameCount(0);
    setLiveUsage({ requests: 0, frames: 0, costUsd: 0, lastLatencyMs: 0 });
    liveStreamingStartedRef.current = false;
    liveStartedAtRef.current = 0;
    setLiveStatus("starting live edge camera");
    try {
      if (source !== "camera" || !running) {
        setLiveStatus(`starting ${cameraFacing === "environment" ? "back" : "front"} camera`);
        await startCamera(cameraFacing, null);
      }
      const video = videoRef.current;
      if (!video) {
        throw new Error("Video element is not ready.");
      }
      await waitForVideoData(video);
      startLocalLiveStream("Live edge camera started.");
    } catch (liveError) {
      setLiveRunning(false);
      setLiveStatus("Live camera error");
      setError(readableError(liveError));
      setLiveTranscript((items) => addRecent(items, readableError(liveError)));
    }
  }, [cameraFacing, captureOneFrame, mode, objective, providerStatus.gemini, running, source, startCamera, zones]);

  const applyPreset = useCallback((preset: (typeof OBJECTIVE_PRESETS)[number]) => {
    setMode(preset.mode);
    setObjective(preset.objective);
  }, []);

  const pointerPosition = useCallback((event: PointerEvent<HTMLElement>) => {
    const rect = videoFrameRef.current?.getBoundingClientRect() ?? event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  }, []);

  const startZoneDraw = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (!zoneToolOpen) {
        return;
      }
      zoneInteractionRef.current = { kind: "draw", start: pointerPosition(event) };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [pointerPosition, zoneToolOpen],
  );

  const startZoneResize = useCallback(
    (zone: Zone, corner: ZoneCorner, event: PointerEvent<HTMLElement>) => {
      event.stopPropagation();
      zoneInteractionRef.current = { kind: "resize", corner, zone };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [],
  );

  const updateZoneDraw = useCallback(
    (event: PointerEvent<HTMLElement>) => {
      const interaction = zoneInteractionRef.current;
      if (!interaction) {
        return;
      }
      const current = pointerPosition(event);

      let x1: number;
      let y1: number;
      let x2: number;
      let y2: number;

      if (interaction.kind === "draw") {
        x1 = interaction.start.x;
        y1 = interaction.start.y;
        x2 = current.x;
        y2 = current.y;
      } else {
        x1 = interaction.zone.x;
        y1 = interaction.zone.y;
        x2 = interaction.zone.x + interaction.zone.w;
        y2 = interaction.zone.y + interaction.zone.h;

        if (interaction.corner === "nw" || interaction.corner === "sw") {
          x1 = current.x;
        } else {
          x2 = current.x;
        }
        if (interaction.corner === "nw" || interaction.corner === "ne") {
          y1 = current.y;
        } else {
          y2 = current.y;
        }
      }

      const x = Math.max(0, Math.min(x1, x2));
      const y = Math.max(0, Math.min(y1, y2));
      const right = Math.min(1, Math.max(x1, x2));
      const bottom = Math.min(1, Math.max(y1, y2));
      const w = right - x;
      const h = bottom - y;
      if (w >= 0.02 && h >= 0.02) {
        setZones([{ id: "zone-1", label: "Restricted zone", x, y, w, h }]);
      }
    },
    [pointerPosition],
  );

  const endZoneDraw = useCallback((event: PointerEvent<HTMLElement>) => {
    zoneInteractionRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const remaining = Math.max(0, DEMO_ANALYSIS_LIMIT - analysesUsed);
  const isCloudSending = analyzing && (status.startsWith("sending") || status.startsWith("edge gate selected"));
  const isLiveSource = source === "camera";
  const isRecordingLocal = holdRecording && !isCloudSending && isLiveSource;
  const providerLabel = PROVIDER_COPY[provider].label;
  const sourceCaptureDetail = isLiveSource
    ? `Live camera: tap Record, draw edge boxes immediately, sample up to ${MAX_FRAMES} local frames, then send one edge-selected cloud request.`
    : `Uploaded/demo video: scan up to ${MAX_FRAMES} keyframes across the clip duration, then send only selected JPEGs.`;
  const analyzeLabel = isLiveSource
    ? "Record edge burst"
    : `Analyze clip with ${providerLabel}`;
  const cameraFacingLabel = cameraFacing === "environment" ? "Back camera" : "Front camera";
  const hasLocalEdgeFrames = lastFrames.length > 0;
  const canSendCloud = hasLocalEdgeFrames && !analyzing && !holdRecording;
  const cameraCaptureState = isCloudSending
    ? "Sending sampled frames to cloud"
    : analyzing
      ? isLiveSource
        ? `Smart recording - edge boxes live`
        : `Sampling full clip locally - not uploaded yet`
      : running
        ? `Preview only - ${cameraFacingLabel} - no recording`
        : source === "camera"
          ? `Camera off - ${cameraFacingLabel} selected`
          : source === "file"
            ? "Local file loaded - no cloud call"
            : "Sample clip loaded - no cloud call";
  const videoHint = isCloudSending
    ? "Cloud request running. Hold this view until the response returns."
    : analyzing
      ? isLiveSource
        ? `Recording locally. Keep pointing at the scene; cloud is not called yet.`
        : `Sampling keyframes across the clip. Full video stays local.`
      : analysis
        ? "Cloud response received. Adjust the prompt or scene, then analyze again."
        : `Local only. ${sourceCaptureDetail}`;
  const analyzeButtonText = analyzing
    ? isCloudSending
      ? `Analyzing with ${providerLabel}...`
      : isLiveSource
        ? "Recording live... stop now"
        : "Sampling frames..."
    : analyzeLabel;
  const videoProcessingTitle = isCloudSending
      ? "Cloud reasoning"
      : analyzing
        ? isLiveSource
          ? "Recording locally"
          : "Sampling keyframes"
        : analysis
          ? "Latest result ready"
          : running
            ? "Camera preview"
            : "Ready";
  const videoProcessingDetail = isCloudSending
      ? "Selected JPEG frames are being analyzed."
      : analyzing
        ? isLiveSource
          ? "Edge boxes update live; cloud waits for your explicit click."
          : "Keep the scene steady while the browser samples frames."
        : analysis
          ? "Result card has the latest scene output."
          : "Press the red button to begin.";
  const quickPresets = OBJECTIVE_PRESETS.slice(0, 6);
  const localObjectChips = summarizeEdgeDetections(lastFrames);
  const actionPreview = analysis?.recommendations.slice(0, 3) ?? [];
  const objectChips = analysis?.objects.length ? fallbackObjectChips(analysis) : localObjectChips;
  const resultStatus = analysis ? "Scene result" : analyzing ? "Analyzing scene" : "Ready for scene result";
  const evidenceFrame = lastFrames.find((frame) => frame.localDetections.length > 0) ?? lastFrames[0] ?? null;
  const evidenceDetections = evidenceFrame ? selectDisplayDetections(evidenceFrame.localDetections, 2) : [];
  const liveOverlayFrame = lastFrames.length ? lastFrames[lastFrames.length - 1] : null;
  const liveOverlayDetections = selectDisplayDetections(source === "camera" && previewDetections.length ? previewDetections : liveOverlayFrame?.localDetections ?? [], 2);
  const hybridEdgeContext = summarizeHybridEdgeContext(lastFrames);
  const cloudConfirmedLabels = new Set(analysis?.objects.map((object) => object.label.toLowerCase()) ?? []);
  const edgeGateLabel = edgeGateSummary
    ? `${edgeGateSummary.selectedFrames}/${edgeGateSummary.inputFrames} sent`
    : "waiting";
  const edgeGateDetail = edgeGateSummary
    ? `Edge gate: ${edgeGateSummary.objectFrames} object frame${edgeGateSummary.objectFrames === 1 ? "" : "s"}, ${edgeGateSummary.motionFrames} motion frame${edgeGateSummary.motionFrames === 1 ? "" : "s"}, ${edgeGateSummary.skippedFrames} skipped.`
    : forceCloudAnalysis
      ? "Edge gate prefers object/motion frames; force cloud fallback is on."
      : "Edge gate waits for local objects or motion before sending frames.";
  const edgePreviewObjectTrigger = previewDetections.some((detection) => detection.score >= EDGE_OBJECT_TRIGGER_SCORE);
  const edgePreviewMotionTrigger = (edgePreviewMetrics?.motionScore ?? 0) >= EDGE_MOTION_TRIGGER_SCORE;
  const edgePreviewDecision = edgePreviewObjectTrigger
    ? "object trigger"
    : edgePreviewMotionTrigger
      ? "motion trigger"
      : edgePreviewMetrics
        ? "watching locally"
        : "waiting for video";
  const edgePreviewQuality = edgePreviewMetrics
    ? edgePreviewMetrics.usable
      ? "usable frame"
      : edgePreviewMetrics.rejectionReason ?? "low-quality frame"
    : "no frame yet";
  const transformersRuntime = getTransformersDetectorRuntime();
  const edgeModelLabel = transformersRuntime.ready
    ? transformersRuntime.device === "webgpu"
      ? "MediaPipe + Transformers WebGPU"
      : "MediaPipe + Transformers WASM"
    : transformersRuntime.loading
      ? "MediaPipe + Transformers warming"
      : "MediaPipe edge";
  const edgeHudDetail = `${edgeModelLabel} - ${previewDetections.length} box${previewDetections.length === 1 ? "" : "es"} - ${edgeLoopFps || "--"} fps - ${edgeDetectionLatencyMs || "--"}ms - frame ${edgePreviewFrameCount}`;
  const edgeMetadataDetail = edgeGateSummary
    ? `${hybridEdgeContext.detections} local box${hybridEdgeContext.detections === 1 ? "" : "es"} sent as metadata with ${edgeGateSummary.selectedFrames} image frame${edgeGateSummary.selectedFrames === 1 ? "" : "s"}`
    : hybridEdgeContext.frames
      ? `${hybridEdgeContext.detections} local box${hybridEdgeContext.detections === 1 ? "" : "es"} ready for cloud context`
      : "Edge metadata will appear while recording";
  const hybridPipelineStatus = analysis
    ? "Cloud enhanced the selected edge frames"
    : hasLocalEdgeFrames
      ? "Edge buffer is live; cloud send is ready"
      : running
        ? "Browser edge is proposing live regions"
        : "Start camera to begin edge proposals";
  const recordingProgressPercent = Math.round(recordingProgress * 100);
  const latestGeminiCommentary = liveGeminiInsight ?? liveTranscript.find((item) => item.startsWith("Gemini:"));
  const liveCommentary = latestGeminiCommentary ?? (liveRunning ? "Gemini is reading selected live snapshots." : null);
  const primaryLiveFeedItems = latestGeminiCommentary
    ? [latestGeminiCommentary]
    : liveTranscript.filter((item) => !item.startsWith("Edge frame:")).slice(0, 1);
  const liveCostLabel = liveUsage.requests ? formatCost(liveUsage.costUsd) : "$0.0000";
  const displayedCloudFrames = liveUsage.requests ? liveUsage.frames : analysis?.edgeAssessment.framesAnalyzed ?? (lastFrames.length || "--");
  const displayedLatency = liveUsage.requests ? `${liveUsage.lastLatencyMs}ms` : analysis ? `${analysis.usage.latencyMs}ms` : "--";
  const displayedCost = liveUsage.requests ? liveCostLabel : analysis ? formatCost(analysis.usage.estimatedCostUsd) : "$0.0000";
  const topRecommendedAction = actionPreview[0] ?? null;
  const displayHeadline = analysis?.headline ?? (liveRunning ? "Gemini live commentary is running" : "Point camera or upload a clip. Ask anything.");
  const displayCommentary =
    analysis?.commentary ??
    liveCommentary ??
    "The browser runs fast edge detection first. Press record/sample, then send selected evidence frames to Gemini/OpenAI for the full answer.";
  const springTransition = prefersReducedMotion ? { duration: 0 } : { type: "spring" as const, stiffness: 260, damping: 30 };
  const fadeUp = prefersReducedMotion
    ? {}
    : {
        initial: { opacity: 0, y: 12 },
        animate: { opacity: 1, y: 0 },
        transition: springTransition,
      };

  return (
    <motion.main className="industrial-shell" {...fadeUp}>
      <motion.header className="industrial-header" layout transition={springTransition}>
        <div>
          <p className="eyebrow">Cloud video analytics API showcase</p>
          <h1>Ask Gemini/OpenAI anything about live video.</h1>
          <p className="app-subtitle">Point a camera or upload video. Browser edge models detect objects and motion first; selected evidence frames go to Gemini/OpenAI for structured video analytics.</p>
        </div>
        <div className="loop">
          <span>video</span>
          <span>edge object + motion gate</span>
          <span>plain-language objective</span>
          <span>Gemini/OpenAI Vision/Cosmos</span>
          <span>structured answer</span>
        </div>
      </motion.header>

      <motion.section className="industrial-grid" layout transition={springTransition}>
        <motion.div className="video-workbench panel" layout transition={springTransition}>
          <motion.div
            className={`industrial-video-frame ${videoOrientation} ${source === "camera" ? "live-camera-frame" : "clip-video-frame"}`}
            layout
            onPointerDown={startZoneDraw}
            onPointerMove={updateZoneDraw}
            onPointerUp={endZoneDraw}
            ref={videoFrameRef}
            transition={springTransition}
          >
            <video muted onLoadedMetadata={updateVideoOrientation} onResize={updateVideoOrientation} playsInline ref={videoRef} />
            <AnimatePresence>
              {liveOverlayDetections.length ? (
              <motion.div className="live-detection-overlay" aria-label="Browser edge detection overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {liveOverlayDetections.map((detection, index) => (
                  <motion.div
                    layout
                    className={`live-detection-box ${isProposalDetection(detection) ? "edge-proposal" : ""} ${cloudConfirmedLabels.has(detection.label.toLowerCase()) ? "cloud-confirmed" : ""}`}
                    key={`${detection.label}-${index}-${detection.x}-${detection.y}`}
                    initial={prefersReducedMotion ? false : { opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={prefersReducedMotion ? undefined : { opacity: 0, scale: 0.96 }}
                    style={overlayStyleForDetection(detection, videoViewport)}
                    transition={springTransition}
                  >
                    <span>{detection.label} {Math.round(detection.score * 100)}%{cloudConfirmedLabels.has(detection.label.toLowerCase()) ? " cloud" : ""}</span>
                  </motion.div>
                ))}
              </motion.div>
            ) : null}
            </AnimatePresence>
            <motion.div className={`edge-live-hud ${edgePreviewObjectTrigger || edgePreviewMotionTrigger ? "triggered" : ""}`} layout transition={springTransition}>
              <div>
                <span>Live edge processing</span>
                <strong>{edgePreviewDecision}</strong>
              </div>
              <small>{edgeHudDetail}</small>
              <em>{edgePreviewQuality}. Green = edge proposal; blue = cloud-confirmed.</em>
            </motion.div>
            <AnimatePresence>
            {liveRunning && liveCommentary ? (
              <motion.div className="live-commentary-overlay" aria-live="polite" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={springTransition}>
                <span>Live commentary</span>
                <strong>{liveCommentary}</strong>
              </motion.div>
            ) : null}
            </AnimatePresence>
            <motion.div className={`video-processing-badge ${analyzing ? "active" : analysis ? "ready" : ""}`} layout transition={springTransition}>
              <strong>{videoProcessingTitle}</strong>
              <span>{videoProcessingDetail}</span>
            </motion.div>
            {zoneToolOpen ? zones.map((zone) => (
              <div
                className="drawn-zone"
                key={zone.id}
                style={{
                  left: `${zone.x * 100}%`,
                  top: `${zone.y * 100}%`,
                  width: `${zone.w * 100}%`,
                  height: `${zone.h * 100}%`,
                }}
              >
                {zone.label}
                {(["nw", "ne", "sw", "se"] as ZoneCorner[]).map((corner) => (
                  <button
                    aria-label={`Resize ${zone.label} ${corner}`}
                    className={`zone-handle ${corner}`}
                    key={corner}
                    onPointerDown={(event) => startZoneResize(zone, corner, event)}
                    onPointerMove={updateZoneDraw}
                    onPointerUp={endZoneDraw}
                    type="button"
                  />
                ))}
              </div>
            )) : null}
            <div className="camera-status" title={status}>{cameraCaptureState}</div>
            <div className={`video-hold-hint ${analyzing ? "active" : ""}`}>{videoHint}</div>
          </motion.div>
          <canvas className="hidden-canvas" ref={canvasRef} />

          <motion.div className="first-fold-console" layout transition={springTransition}>
            <motion.div className="native-workflow-rail" aria-label="Video analytics workflow status" layout transition={springTransition}>
              <motion.div className={source !== "camera" || running ? "active" : ""} layout transition={springTransition}>
                <span>01</span>
                <strong>{source === "camera" ? running ? "Camera live" : "Start camera" : "Clip loaded"}</strong>
              </motion.div>
              <motion.div className={hasLocalEdgeFrames || analyzing ? "active" : ""} layout transition={springTransition}>
                <span>02</span>
                <strong>{hasLocalEdgeFrames ? `${lastFrames.length} edge frames` : analyzing ? "Sampling edge" : "Capture evidence"}</strong>
              </motion.div>
              <motion.div className={analysis || isCloudSending ? "active" : ""} layout transition={springTransition}>
                <span>03</span>
                <strong>{analysis ? "Cloud answer ready" : isCloudSending ? "Cloud reasoning" : "Send to cloud"}</strong>
              </motion.div>
            </motion.div>

            <div className="first-fold-header">
              <div>
                <span>On-the-fly video analytics</span>
                <strong>{source === "camera" ? "Live edge camera + cloud answer" : "Uploaded video + cloud answer"}</strong>
              </div>
              <small>{isLiveSource ? "Live camera burst" : "Full clip keyframe scan"}</small>
            </div>

            <motion.div className={`capture-command-center ${liveRunning ? "live" : ""} ${analysis ? "ready" : ""}`} aria-label="Unified live capture and cloud feedback" layout transition={springTransition}>
              <div className="capture-command-header">
                <div>
                  <span>Capture and live answer</span>
                  <strong>{liveRunning ? "Gemini is commenting on selected live frames" : analysis ? "Latest cloud answer is ready" : "Record, stream live, or upload a clip"}</strong>
                </div>
                <small>{liveFrameCount || lastFrames.length} edge / {liveUsage.requests || (analysis ? 1 : 0)} cloud</small>
              </div>
              <div className="capture-command-actions">
                <motion.button
                  aria-pressed={holdRecording}
                  className={`analyze-button first-fold-analyze ${isRecordingLocal ? "recording" : ""}`}
                  disabled={source === "camera" ? isCloudSending : analyzing}
                  whileTap={prefersReducedMotion ? undefined : { scale: 0.98 }}
                  onClick={(event) => {
                    if (source === "camera") {
                      event.preventDefault();
                      if (holdRecording) {
                        finishHoldRecording();
                      } else {
                        void beginHoldRecording();
                      }
                      return;
                    }
                    void recordOrAnalyze();
                  }}
                  type="button"
                >
                  <span className="record-dot" /> {analyzeButtonText}
                </motion.button>
                {isLiveSource ? (
                  <motion.button className="send-cloud-button" disabled={!canSendCloud} onClick={() => void sendLatestFramesToCloud()} type="button" whileTap={prefersReducedMotion ? undefined : { scale: 0.98 }}>
                    Send evidence to cloud
                    <span>{hasLocalEdgeFrames ? `${lastFrames.length} local frame${lastFrames.length === 1 ? "" : "s"} ready` : "record first"}</span>
                  </motion.button>
                ) : null}
                <button className={`live-command-button ${liveRunning ? "active" : ""}`} disabled={liveRunning} onClick={() => void startGeminiLive()} type="button">
                  <Radar size={16} /> Start Gemini Live
                </button>
                <button className="stop-live-command" disabled={!liveRunning} onClick={stopGeminiLive} type="button">
                  <Pause size={16} /> Stop Live
                </button>
              </div>
              {isLiveSource ? (
                <div className="smart-record-progress" aria-label={`Recording progress ${recordingProgressPercent}%`}>
                  <span style={{ width: `${recordingProgressPercent}%` }} />
                </div>
              ) : null}
              <div className="command-result-preview" aria-live="polite">
                <div className="command-result-topline">
                  <span>{liveRunning ? liveStatus : resultStatus}</span>
                  <strong>{analysis ? `${Math.round(analysis.confidence)}% confidence` : liveRunning ? "Live" : "Ready"}</strong>
                </div>
                <h2>{displayHeadline}</h2>
                <p>{displayCommentary}</p>
                {(primaryLiveFeedItems.length ? primaryLiveFeedItems : ["Press Start Gemini Live. The camera opens here, edge boxes update locally, and Gemini comments on selected live snapshots."]).map((item) => (
                  <p className="command-live-line" key={item}>{item}</p>
                ))}
                {topRecommendedAction ? (
                  <div className="command-next-action">
                    <span>P{topRecommendedAction.priority} / {topRecommendedAction.owner}</span>
                    <strong>{topRecommendedAction.action}</strong>
                    <p>{topRecommendedAction.reason}</p>
                  </div>
                ) : null}
                <div className="gemini-live-usage command-usage-grid" aria-label="Live Gemini usage">
                  <span><strong>{displayedCost}</strong> total cost</span>
                  <span><strong>{liveUsage.requests || (analysis ? 1 : 0)}</strong> cloud call{liveUsage.requests === 1 ? "" : "s"}</span>
                  <span><strong>{displayedCloudFrames}</strong> cloud frame{displayedCloudFrames === 1 ? "" : "s"}</span>
                  <span><strong>{displayedLatency}</strong> latency</span>
                </div>
                {liveEvents.some((item) => item.startsWith("Cloud")) ? (
                  <div className="gemini-live-events">
                    {liveEvents.filter((item) => item.startsWith("Cloud")).slice(0, 1).map((item) => (
                      <small key={item}>{item}</small>
                    ))}
                  </div>
                ) : null}
              </div>
            </motion.div>

            <motion.div className="hybrid-pipeline-strip" aria-label="Hybrid edge and cloud pipeline" layout transition={springTransition}>
              <motion.div className={running ? "active" : ""} layout transition={springTransition}>
                <span>Local edge</span>
                <strong>{edgeModelLabel}</strong>
              </motion.div>
              <motion.div className={hasLocalEdgeFrames ? "active" : ""} layout transition={springTransition}>
                <span>Frame buffer</span>
                <strong>{lastFrames.length || "--"} current frame{lastFrames.length === 1 ? "" : "s"}</strong>
              </motion.div>
              <motion.div className={analysis ? "active" : ""} layout transition={springTransition}>
                <span>Cloud answer</span>
                <strong>{analysis ? `${Math.round(analysis.confidence)}%` : "manual send"}</strong>
              </motion.div>
              <small>{hybridPipelineStatus}. Edge filters the video; cloud reasons over selected evidence.</small>
            </motion.div>

            <motion.div className={`scene-result-card secondary-result-card ${liveRunning ? "live-secondary" : ""} ${analysis ? "ready" : analyzing ? "loading" : ""}`} layout transition={springTransition}>
              <div className="scene-result-topline">
                <span>{resultStatus}</span>
                <strong>{analysis ? `${Math.round(analysis.confidence)}% confidence` : analyzing ? "Working..." : "No cloud call yet"}</strong>
              </div>
              <AnimatePresence mode="popLayout">
                <motion.h2 key={displayHeadline} layout initial={prefersReducedMotion ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={prefersReducedMotion ? undefined : { opacity: 0, y: -8 }} transition={springTransition}>{displayHeadline}</motion.h2>
              </AnimatePresence>
              <p>{displayCommentary}</p>
              <AnimatePresence>
              {error ? (
                <motion.div className="primary-error-banner" role="alert" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={springTransition}>
                  <AlertTriangle size={16} />
                  <span>{error}</span>
                </motion.div>
              ) : null}
              </AnimatePresence>
              <div className={`priority-actions-panel ${actionPreview.length ? "ready" : ""}`} aria-label="Recommended actions">
                <div className="priority-actions-heading">
                  <span>Recommended actions</span>
                  <strong>{actionPreview.length ? `${actionPreview.length} next step${actionPreview.length === 1 ? "" : "s"}` : "Waiting for analysis"}</strong>
                </div>
                {actionPreview.length ? (
                  <div className="priority-action-list">
                    {actionPreview.map((action) => (
                      <motion.div className="priority-action-item" key={`${action.priority}-${action.action}`} layout initial={prefersReducedMotion ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={springTransition}>
                        <span>P{action.priority} / {action.owner}</span>
                        <strong>{action.action}</strong>
                        <p>{action.reason}</p>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <p>Actions will appear here first: what to do, who should do it, and why the video evidence supports it.</p>
                )}
              </div>
              <div className="scene-result-metrics">
                <div>
                  <span>Edge gate</span>
                  <strong>{edgeGateLabel}</strong>
                </div>
                <div>
                  <span>Local boxes</span>
                  <strong>{hybridEdgeContext.detections || "--"} boxes</strong>
                </div>
                <div>
                  <span>Cloud frames</span>
                  <strong>{displayedCloudFrames}</strong>
                </div>
                <div>
                  <span>Latency</span>
                  <strong>{displayedLatency}</strong>
                </div>
                <div>
                  <span>{liveUsage.requests ? "Live cost" : "Cost"}</span>
                  <strong>{displayedCost}</strong>
                </div>
              </div>
              <div className="object-chip-row detected-object-row" aria-label="Detected objects">
                <strong>Detected objects</strong>
                {objectChips.map((object) => (
                  <span key={`${object.label}-${object.count}`}>
                    {object.label} {object.count > 1 ? `x${object.count}` : ""} {object.score ? `${Math.round(object.score * 100)}%` : ""}
                  </span>
                ))}
                {!objectChips.length ? <span>No objects captured yet</span> : null}
                <small>{detectorStatus}. {edgeMetadataDetail}. {edgeGateDetail}</small>
              </div>
              <AnimatePresence>
              {evidenceFrame && !liveRunning ? (
                <motion.div className="evidence-frame-panel" aria-label="Evidence frame with detections" layout initial={prefersReducedMotion ? false : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} transition={springTransition}>
                  <div className="evidence-frame-topline">
                    <span>Evidence frame</span>
                    <strong>{evidenceDetections.length ? `${evidenceDetections.length} box${evidenceDetections.length === 1 ? "" : "es"}` : "No boxes yet"}</strong>
                  </div>
                  <div className="evidence-frame-image">
                    <Image alt="Sampled evidence frame" fill sizes="(max-width: 720px) 100vw, 420px" src={evidenceFrame.imageDataUrl} unoptimized />
                    {evidenceDetections.map((detection, index) => (
                      <div
                        className={`evidence-box ${cloudConfirmedLabels.has(detection.label.toLowerCase()) ? "cloud-confirmed" : ""}`}
                        key={`${detection.label}-${index}-${detection.x}`}
                        style={overlayStyleForDetection(detection, { x: 0, y: 0, w: 100, h: 100 })}
                      >
                        <span>{detection.label} {Math.round(detection.score * 100)}%{cloudConfirmedLabels.has(detection.label.toLowerCase()) ? " cloud" : ""}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              ) : null}
              </AnimatePresence>
              <AnimatePresence>
              {analysis && !liveRunning ? (
                <motion.div className="full-response-panel" layout initial={prefersReducedMotion ? false : { opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }} transition={springTransition}>
                  <div className="response-section">
                    <span>Full response</span>
                    <strong>{analysis.scene.summary}</strong>
                    <p>{analysis.scene.environment} {analysis.scene.activity}</p>
                  </div>
                  {analysis.objects.length ? (
                    <div className="detected-object-grid">
                      {analysis.objects.slice(0, 6).map((object) => (
                        <div key={`${object.label}-${object.evidence}`}>
                          <strong>{object.label} {object.count > 1 ? `x${object.count}` : ""}</strong>
                          <span>{object.locations.join(", ")}</span>
                          <p>{object.evidence}</p>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {analysis.alerts.length ? (
                    <div className="response-section compact-response-list">
                      <span>Alerts</span>
                      {analysis.alerts.slice(0, 3).map((alert) => (
                        <p key={`${alert.label}-${alert.zoneId ?? "global"}`}>
                          <strong>{alert.triggered ? "Triggered" : "Not triggered"}:</strong> {alert.label} - {alert.evidence}
                        </p>
                      ))}
                    </div>
                  ) : null}
                  {analysis.timeline.length ? (
                    <div className="response-section compact-response-list">
                      <span>Timeline</span>
                      {analysis.timeline.slice(0, 4).map((item) => (
                        <p key={`${item.frame}-${item.observation}`}>
                          <strong>Frame {item.frame}:</strong> {item.observation}
                        </p>
                      ))}
                    </div>
                  ) : null}
                </motion.div>
              ) : null}
              </AnimatePresence>
            </motion.div>

            <label className="main-objective-box">
              <span>Plain-language analytics question</span>
              <textarea onChange={(event) => setObjective(event.target.value)} placeholder={DEFAULT_OBJECTIVE} value={objective} />
            </label>

            <div className="quick-preset-row" aria-label="Example analytics">
              <button className={objective === DEFAULT_OBJECTIVE ? "active" : ""} onClick={() => {
                setObjective(DEFAULT_OBJECTIVE);
                setMode("industrial_general");
              }} type="button">
                General query
              </button>
              {quickPresets.map((preset) => (
                <button className={objective === preset.objective ? "active" : ""} key={`quick-${preset.label}`} onClick={() => applyPreset(preset)} type="button">
                  {preset.label}
                </button>
              ))}
            </div>

            <div className="first-fold-actions">
              <button className={source === "camera" && running ? "active" : ""} disabled={running} onClick={() => void startCamera()} type="button">
                <Camera size={16} />
                <span>{running ? "Camera on" : `Start ${cameraFacing === "environment" ? "back" : "front"} camera`}</span>
              </button>
              <button className={source === "camera" ? "active" : ""} onClick={() => void switchCamera()} type="button">
                <RotateCcw size={16} />
                <span>Flip</span>
              </button>
              <label className={`upload-card compact ${source === "file" ? "active" : ""}`}>
                <FileVideo size={16} />
                <span>Upload clip</span>
                <input
                  accept="video/*"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) {
                      void loadFile(file);
                    }
                  }}
                  type="file"
                />
              </label>
              <button disabled={!running && !holdRecording} onClick={stopCamera} type="button">
                <Pause size={16} />
                <span>{holdRecording ? "Cancel" : "Stop"}</span>
              </button>
            </div>

            <div className="sample-strip primary-samples" aria-label="Quick demo clips">
              {SAMPLE_CLIPS.filter((clip) => clip.url).map((clip) => (
                <button className={sampleUrl === clip.url && source === "sample" ? "active" : ""} key={`primary-${clip.url}`} onClick={() => void loadSampleClip(clip)} type="button">
                  Use {clip.label}
                </button>
              ))}
            </div>

            <div className={`capture-disclosure ${isCloudSending ? "cloud" : analyzing ? "local" : ""}`}>
              <strong>{cameraCaptureState}</strong>
              <span>
                {isCloudSending
                  ? `Only edge-triggered JPEG keyframes are in the cloud now. ${edgeGateDetail}`
                  : `${sourceCaptureDetail} ${detectorStatus}. ${edgeGateDetail} Full video is not uploaded.`}
              </span>
            </div>

            <label className="force-cloud-toggle">
              <input checked={forceCloudAnalysis} onChange={(event) => setForceCloudAnalysis(event.target.checked)} type="checkbox" />
              <span>
                <strong>Force cloud if edge gate is quiet</strong>
                <small>{forceCloudAnalysis ? "Demo-safe: always returns an answer." : "Strict efficiency: cloud only after local object/motion trigger."}</small>
              </span>
            </label>
          </motion.div>

          <p className="panel-note">
            Camera mode records local edge evidence first. Cloud analysis only runs when you press Send selected frames to cloud. Full video stays in the browser; only object/motion-triggered JPEG keyframes are sent.
            Upload mode works without camera access on Safari, Edge, Chrome, and desktop browsers that support video canvas extraction.
          </p>
        </motion.div>

        <aside className="analytics-control panel">
          <details className="optional-config-panel">
            <summary>
              <div>
                <p className="eyebrow">Advanced settings</p>
                <h2>Model, zones and utilities</h2>
              </div>
              <span>{remaining} runs left</span>
            </summary>

            <div className="optional-config-body">
              <div className="mission-section compact">
                <div className="section-title">
                  <span>1</span>
                  <strong>Choose model</strong>
                </div>
                <div className="provider-toggle" role="group" aria-label="Provider">
                  {(["gemini", "openai", "nvidia"] as ProviderId[]).map((candidate) => (
                    <button
                      className={provider === candidate ? "active" : ""}
                      key={candidate}
                      onClick={() => setProvider(candidate)}
                      type="button"
                    >
                      {PROVIDER_COPY[candidate].label}
                      <small>{providerStatus[candidate] ? "ready" : "missing key"}</small>
                    </button>
                  ))}
                </div>
              </div>

              {devices.length > 0 ? (
                <label className="device-select">
                  Camera device
                  <select disabled={running} onChange={(event) => setSelectedDeviceId(event.target.value || null)} value={selectedDeviceId ?? ""}>
                    {devices.map((device, index) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${index + 1}`}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}

              <div className="advanced-grid">
                <div className="zone-row">
                  <Crosshair size={16} />
                  <span>
                    Zone tool: {zoneToolOpen ? `on (${Math.round((zones[0]?.w ?? 0) * 100)}% x ${Math.round((zones[0]?.h ?? 0) * 100)}%)` : "off"}
                  </span>
                  <button onClick={() => setZoneToolOpen((value) => !value)} type="button">
                    {zoneToolOpen ? "Hide zone" : "Draw zone"}
                  </button>
                </div>

                <div className="source-picker">
                  <select onChange={(event) => setSampleUrl(event.target.value)} value={sampleUrl}>
                    {SAMPLE_CLIPS.map((clip) => (
                      <option key={clip.label} value={clip.url}>
                        {clip.label}
                      </option>
                    ))}
                  </select>
                  <input onChange={(event) => setSampleUrl(event.target.value)} placeholder="https://example.com/video.mp4" value={sampleUrl} />
                  <button onClick={loadSampleUrl} type="button">
                    <Play size={16} /> Load URL
                  </button>
                </div>

                <div className="utility-row">
                  <button className="usage-reset" onClick={resetLimit} type="button">
                    Reset limit
                  </button>
                  <button onClick={reset} type="button">
                    <RotateCcw size={16} /> Clear result
                  </button>
                </div>
              </div>
            </div>
          </details>

          {error && !analysis ? (
            <div className="inline-error">
              <AlertTriangle size={18} />
              <p>{error}</p>
            </div>
          ) : null}
        </aside>
      </motion.section>
    </motion.main>
  );
}
