"use client";

import { AlertTriangle, Camera, Crosshair, FileVideo, Pause, Play, Radar, RotateCcw, Send, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import { buildVideoConstraints, listVideoInputDevices, stopStream, type CameraFacing } from "@/lib/camera";
import { canvasToJpegDataUrl, captureVideoFrame, computeEdgeMetrics, type EdgeMetricResult } from "@/lib/edgeMetrics";
import { createObjectDetector, detectObjectsForVideo } from "@/lib/mediaPipeDetector";
import type { LocalDetection } from "@/lib/types";
import type { ProviderId, SampledFrame, VideoAnalysisResponse, VideoMode, VideoSource, Zone } from "@/lib/videoSchema";

type ApiError = {
  error?: string;
  detail?: string;
};

type ZoneCorner = "nw" | "ne" | "sw" | "se";
type ZoneInteraction =
  | { kind: "draw"; start: { x: number; y: number } }
  | { kind: "resize"; corner: ZoneCorner; zone: Zone };

const MAX_FRAMES = 5;
const CAMERA_CAPTURE_WINDOW_MS = 2000;
const JPEG_QUALITY = 0.58;
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

function modeLabel(mode: VideoMode): string {
  const labels: Record<VideoMode, string> = {
    industrial_general: "Industrial",
    person_zone: "Person zone",
    ppe: "PPE",
    safety: "Safety",
    operations: "Operations",
  };
  return labels[mode];
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
}: {
  providerStatus: Record<ProviderId, boolean>;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoFrameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const liveSocketRef = useRef<WebSocket | null>(null);
  const liveTimerRef = useRef<number | null>(null);
  const previousFrameRef = useRef<EdgeMetricResult["snapshot"] | null>(null);
  const zoneInteractionRef = useRef<ZoneInteraction | null>(null);

  const [provider, setProvider] = useState<ProviderId>(providerStatus.gemini ? "gemini" : providerStatus.openai ? "openai" : "nvidia");
  const [mode, setMode] = useState<VideoMode>("person_zone");
  const [source, setSource] = useState<VideoSource>("camera");
  const [objective, setObjective] = useState(OBJECTIVE_PRESETS[0].objective);
  const [zones, setZones] = useState<Zone[]>([
    { id: "zone-1", label: "Restricted zone", x: 0.58, y: 0.2, w: 0.3, h: 0.58 },
  ]);
  const [zoneToolOpen, setZoneToolOpen] = useState(false);
  const [status, setStatus] = useState("camera off - no recording");
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>("environment");
  const [sampleUrl, setSampleUrl] = useState(SAMPLE_CLIPS[1].url);
  const [analysis, setAnalysis] = useState<VideoAnalysisResponse | null>(null);
  const [analysesUsed, setAnalysesUsed] = useState(initialAnalysesUsed);
  const [videoAspectRatio, setVideoAspectRatio] = useState("16 / 9");
  const [liveStatus, setLiveStatus] = useState("Gemini Live idle");
  const [liveRunning, setLiveRunning] = useState(false);
  const [liveEvents, setLiveEvents] = useState<string[]>([]);

  useEffect(() => {
    return () => {
      if (liveTimerRef.current) {
        window.clearInterval(liveTimerRef.current);
      }
      liveSocketRef.current?.close();
      stopStream(streamRef.current);
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  const refreshDevices = useCallback(async () => {
    const nextDevices = await listVideoInputDevices();
    setDevices(nextDevices);
    if (!selectedDeviceId && nextDevices[0]?.deviceId) {
      setSelectedDeviceId(nextDevices[0].deviceId);
    }
  }, [selectedDeviceId]);

  const updateAspectRatio = useCallback(() => {
    const video = videoRef.current;
    if (video?.videoWidth && video.videoHeight) {
      setVideoAspectRatio(`${video.videoWidth} / ${video.videoHeight}`);
    }
  }, []);

  const clearVideoObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  const startCamera = useCallback(async (nextFacing: CameraFacing = cameraFacing, deviceId: string | null = selectedDeviceId) => {
    setError(null);
    setCameraFacing(nextFacing);
    setStatus(`requesting ${nextFacing === "environment" ? "back" : "front"} camera permission`);
    setSource("camera");
    try {
      clearVideoObjectUrl();
      stopStream(streamRef.current);
      const stream = await navigator.mediaDevices.getUserMedia(buildVideoConstraints("webcam_coach", deviceId, nextFacing));
      streamRef.current = stream;
      if (!videoRef.current) {
        throw new Error("Video element is not ready.");
      }
      videoRef.current.srcObject = stream;
      videoRef.current.controls = false;
      await videoRef.current.play();
      updateAspectRatio();
      await refreshDevices();
      previousFrameRef.current = null;
      setRunning(true);
      setStatus(`preview only - ${nextFacing === "environment" ? "back" : "front"} camera - no cloud upload`);
    } catch (startError) {
      setRunning(false);
      setStatus("camera off - no recording");
      setError(readableError(startError));
    }
  }, [cameraFacing, clearVideoObjectUrl, refreshDevices, selectedDeviceId, updateAspectRatio]);

  const stopCamera = useCallback(() => {
    setRunning(false);
    setStatus("camera off - no recording");
    stopStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

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
      if (videoRef.current) {
        videoRef.current.crossOrigin = "";
        videoRef.current.srcObject = null;
        videoRef.current.src = objectUrl;
        videoRef.current.controls = true;
        videoRef.current.muted = true;
        await videoRef.current.load();
      }
      setRunning(false);
      setStatus("local video ready");
    },
    [clearVideoObjectUrl],
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
    if (videoRef.current) {
      videoRef.current.crossOrigin = "anonymous";
      videoRef.current.srcObject = null;
      videoRef.current.src = sampleUrl.trim();
      videoRef.current.controls = true;
      videoRef.current.muted = true;
      videoRef.current.load();
    }
    setRunning(false);
    setStatus("sample url loaded; press play or analyze");
  }, [clearVideoObjectUrl, sampleUrl]);

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
      if (videoRef.current) {
        videoRef.current.crossOrigin = "anonymous";
        videoRef.current.srcObject = null;
        videoRef.current.src = clip.url;
        videoRef.current.controls = true;
        videoRef.current.muted = true;
        videoRef.current.load();
      }
      setRunning(false);
      setStatus(`${clip.label.toLowerCase()} sample ready`);
    },
    [clearVideoObjectUrl],
  );

  const captureOneFrame = useCallback(async (timestampMs: number): Promise<SampledFrame | null> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const imageData = captureVideoFrame(video, canvas, 720);
    if (!imageData) {
      return null;
    }

    const result = computeEdgeMetrics(imageData, previousFrameRef.current);
    previousFrameRef.current = result.snapshot;

    let localDetections: LocalDetection[] = [];
    try {
      localDetections = await detectObjectsForVideo(video, performance.now());
    } catch {
      localDetections = [];
    }

    return {
      imageDataUrl: canvasToJpegDataUrl(canvas, JPEG_QUALITY),
      timestampMs,
      edgeMetrics: result.metrics,
      localDetections,
    };
  }, []);

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
    await createObjectDetector();
    previousFrameRef.current = null;
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

  const analyze = useCallback(async () => {
    if (!providerStatus[provider]) {
      setError(`${provider.toUpperCase()} API key is not configured on the server.`);
      return;
    }
    if (analysesUsed >= DEMO_ANALYSIS_LIMIT) {
      setError("Demo analysis limit reached for this browser session.");
      return;
    }

    setAnalyzing(true);
    setError(null);
    setAnalysis(null);
    setStatus("extracting relevant frames in browser");
    try {
      const frames = await sampleFrames();
      setStatus(`sending ${frames.length} sampled frames to ${provider}`);
      const response = await fetch("/api/analyze-video", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          provider,
          mode,
          source,
          objective,
          zones,
          frames,
          sampling: {
            requestedFps: 1,
            maxFrames: MAX_FRAMES,
            jpegQuality: JPEG_QUALITY,
            payloadBytes: payloadBytes(frames),
          },
        }),
      });

      const payload: unknown = await response.json();
      if (!response.ok) {
        if (isApiError(payload)) {
          throw new Error(payload.detail ?? payload.error ?? "Analysis route failed.");
        }
        throw new Error("Analysis route failed.");
      }

      const nextAnalysis = payload as VideoAnalysisResponse;
      const nextUsage = analysesUsed + 1;
      setAnalysesUsed(nextUsage);
      window.sessionStorage.setItem("cloud-video-analyzer-analyses-used", String(nextUsage));
      setAnalysis(nextAnalysis);
      setStatus(`${provider} analysis complete in ${nextAnalysis.usage.latencyMs}ms`);
    } catch (analysisError) {
      setError(readableError(analysisError));
      setStatus("analysis error");
    } finally {
      setAnalyzing(false);
    }
  }, [analysesUsed, mode, objective, provider, providerStatus, sampleFrames, source, zones]);

  const reset = useCallback(() => {
    setError(null);
    setAnalysis(null);
    setStatus(running ? `preview only - ${cameraFacing === "environment" ? "back" : "front"} camera - no cloud upload` : "camera off - no recording");
    previousFrameRef.current = null;
  }, [cameraFacing, running]);

  const resetLimit = useCallback(() => {
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
    setLiveRunning(false);
    setLiveStatus("Gemini Live stopped");
  }, []);

  const startGeminiLive = useCallback(async () => {
    if (!providerStatus.gemini) {
      setError("Gemini API key is required for Gemini Live.");
      return;
    }
    setError(null);
    setLiveStatus("minting ephemeral token");
    try {
      const tokenResponse = await fetch("/api/gemini-live-token", { method: "POST" });
      const tokenPayload = (await tokenResponse.json()) as { token?: string; model?: string; detail?: string; error?: string };
      if (!tokenResponse.ok || !tokenPayload.token || !tokenPayload.model) {
        throw new Error(tokenPayload.detail ?? tokenPayload.error ?? "Could not create Gemini Live token.");
      }

      const wsUrl = `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained?access_token=${encodeURIComponent(tokenPayload.token)}`;
      const websocket = new WebSocket(wsUrl);
      liveSocketRef.current = websocket;
      setLiveEvents([]);
      setLiveStatus("connecting to Gemini Live");

      websocket.onopen = () => {
        websocket.send(
          JSON.stringify({
            setup: {
              model: `models/${tokenPayload.model}`,
              responseModalities: ["AUDIO"],
              systemInstruction: {
                parts: [
                  {
                    text: "You are an industrial live video analytics assistant. Watch streamed frames, follow the user's analytics objective, and respond concisely when alerts or useful observations appear.",
                  },
                ],
              },
            },
          }),
        );
        websocket.send(
          JSON.stringify({
            realtimeInput: {
              text: `Analytics objective: ${objective}. Zones: ${zones.map((zone) => `${zone.id} ${zone.label}`).join(", ") || "none"}.`,
            },
          }),
        );
        setLiveRunning(true);
        setLiveStatus("live websocket open; streaming <=1 FPS");

        liveTimerRef.current = window.setInterval(() => {
          void captureOneFrame(Date.now()).then((frame) => {
            if (!frame || websocket.readyState !== WebSocket.OPEN) {
              return;
            }
            websocket.send(
              JSON.stringify({
                realtimeInput: {
                  video: {
                    data: frame.imageDataUrl.slice(frame.imageDataUrl.indexOf(",") + 1),
                    mimeType: "image/jpeg",
                  },
                },
              }),
            );
            setLiveEvents((items) => [`sent frame ${new Date().toLocaleTimeString([], { minute: "2-digit", second: "2-digit" })}`, ...items].slice(0, 4));
          });
        }, 1000);
      };

      websocket.onmessage = (event) => {
        const raw = typeof event.data === "string" ? event.data : "binary response";
        setLiveEvents((items) => [`received ${raw.slice(0, 90)}`, ...items].slice(0, 4));
      };

      websocket.onerror = () => {
        setLiveStatus("Gemini Live websocket error");
      };

      websocket.onclose = () => {
        if (liveTimerRef.current) {
          window.clearInterval(liveTimerRef.current);
          liveTimerRef.current = null;
        }
        setLiveRunning(false);
        setLiveStatus("Gemini Live closed");
      };
    } catch (liveError) {
      setLiveRunning(false);
      setLiveStatus("Gemini Live error");
      setError(readableError(liveError));
    }
  }, [captureOneFrame, objective, providerStatus.gemini, zones]);

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
  const triggeredAlerts = analysis?.alerts.filter((alert) => alert.triggered).length ?? 0;
  const primaryAlert = triggeredAlerts ? `${triggeredAlerts} alert${triggeredAlerts === 1 ? "" : "s"}` : analysis ? "clear" : "waiting";
  const cloudFrameCount = analysis ? String(analysis.edgeAssessment.framesAnalyzed) : analyzing && status.startsWith("sending") ? "sending" : "0";
  const estimatedCost = analysis ? formatCost(analysis.usage.estimatedCostUsd) : "--";
  const isCloudSending = analyzing && status.startsWith("sending");
  const captureWindowSeconds = Math.round(CAMERA_CAPTURE_WINDOW_MS / 1000);
  const isLiveSource = source === "camera";
  const sourceCaptureDetail = isLiveSource
    ? `Live camera: capture a ${captureWindowSeconds}s local burst, sample up to ${MAX_FRAMES} frames, then send selected JPEGs.`
    : `Uploaded/demo video: sample up to ${MAX_FRAMES} keyframes across the full clip duration, then send selected JPEGs.`;
  const analyzeLabel = isLiveSource ? `Capture ${captureWindowSeconds}s + Analyze` : "Sample clip + Analyze";
  const cameraFacingLabel = cameraFacing === "environment" ? "Back camera" : "Front camera";
  const cameraCaptureState = isCloudSending
    ? "Sending sampled frames to cloud"
    : analyzing
      ? isLiveSource
        ? `Capturing local ${captureWindowSeconds}s burst - not uploaded yet`
        : `Sampling full clip locally - not uploaded yet`
      : running
        ? `Preview only - ${cameraFacingLabel} - no recording`
        : source === "camera"
          ? `Camera off - ${cameraFacingLabel} selected`
          : source === "file"
            ? "Local file loaded - no cloud call"
            : "Sample clip loaded - no cloud call";
  const cloudState = isCloudSending ? "Cloud analyzing" : analyzing ? "Local preprocessing" : analysis ? "Cloud response" : "Local only";
  const videoHint = isCloudSending
    ? "Cloud request running. Hold this view until the response returns."
    : analyzing
      ? isLiveSource
        ? `Capturing a local ${captureWindowSeconds}s burst. Keep pointing at the scene.`
        : `Sampling keyframes across the clip. Full video stays local.`
      : analysis
        ? "Cloud response received. Adjust the prompt or scene, then analyze again."
        : `Local only. ${sourceCaptureDetail}`;
  const analyzeButtonText = analyzing ? (isCloudSending ? "Cloud analyzing..." : isLiveSource ? `Capturing ${captureWindowSeconds}s burst...` : "Sampling clip...") : analyzeLabel;
  const quickPresets = OBJECTIVE_PRESETS.slice(0, 6);

  return (
    <main className="industrial-shell">
      <header className="industrial-header">
        <div>
          <p className="eyebrow">Cloud video analytics API showcase</p>
          <h1>Describe any video analytics. Get structured alerts.</h1>
          <p className="app-subtitle">Point the camera, upload a clip, or run a sample. The browser samples keyframes locally, then sends only selected JPEG frames to Gemini/OpenAI for alerts, evidence, cost, and actions.</p>
        </div>
        <div className="loop">
          <span>video</span>
          <span>browser edge sampler</span>
          <span>plain-language objective</span>
          <span>Gemini/OpenAI Vision/Cosmos</span>
          <span>alert API</span>
        </div>
      </header>

      <section className="industrial-grid">
        <div className="video-workbench panel">
          <div
            className="industrial-video-frame"
            onPointerDown={startZoneDraw}
            onPointerMove={updateZoneDraw}
            onPointerUp={endZoneDraw}
            ref={videoFrameRef}
            style={{ aspectRatio: videoAspectRatio }}
          >
            <video muted onLoadedMetadata={updateAspectRatio} playsInline ref={videoRef} />
            {analysis ? (
              <div className="analysis-annotation-overlay" aria-label="Rendered AI annotations">
                <strong>{analysis.headline}</strong>
                <span>{analysis.alerts.some((alert) => alert.triggered) ? "Alert annotation" : "No alert annotation"}</span>
              </div>
            ) : null}
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
          </div>
          <canvas className="hidden-canvas" ref={canvasRef} />

          <div className="first-fold-console">
            <div className="first-fold-header">
              <div>
                <span>On-the-fly video analytics</span>
                <strong>Type what to detect, choose video, get outcomes</strong>
              </div>
              <small>{isLiveSource ? "Live camera burst" : "Full clip keyframe scan"}</small>
            </div>

            <label className="main-objective-box">
              <span>What analytics do you want?</span>
              <textarea onChange={(event) => setObjective(event.target.value)} placeholder="Example: detect if a person enters the marked area and recommend an action." value={objective} />
            </label>

            <div className="quick-preset-row" aria-label="Example analytics">
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
                <span>Upload</span>
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
              <button disabled={!running} onClick={stopCamera} type="button">
                <Pause size={16} />
                <span>Stop</span>
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
                  ? "Only selected JPEG keyframes are in the cloud now."
                  : `${sourceCaptureDetail} Full video is not uploaded.`}
              </span>
            </div>

            <button className="analyze-button first-fold-analyze" disabled={analyzing} onClick={analyze} type="button">
              <Send size={17} /> {analyzeButtonText}
            </button>

            <div className={`inline-result-preview ${analysis ? "ready" : ""}`}>
              <span>Result preview</span>
              <strong>{analysis?.headline ?? "No result yet"}</strong>
              <p>{analysis?.commentary ?? "Run the workflow and the first structured result appears here immediately."}</p>
            </div>
          </div>

          <div className="video-insights-row" aria-label="Current analysis status">
            <div>
              <span>Cloud state</span>
              <strong>{cloudState}</strong>
            </div>
            <div>
              <span>Frames to cloud</span>
              <strong>{cloudFrameCount}</strong>
            </div>
            <div>
              <span>Alert state</span>
              <strong>{primaryAlert}</strong>
            </div>
            <div>
              <span>Est. cost</span>
              <strong>{estimatedCost}</strong>
            </div>
          </div>

          <p className="panel-note">
            Hold the camera on the scene until the response appears. Full video stays in the browser; only selected JPEG keyframes and edge signals are sent. Cost is estimated from provider token usage.
          </p>
        </div>

        <aside className="analytics-control panel">
          <details className="optional-config-panel">
            <summary>
              <div>
                <p className="eyebrow">Advanced settings</p>
                <h2>Model, zones and live mode</h2>
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

                <div className="live-controls">
                  <button disabled={liveRunning} onClick={startGeminiLive} type="button">
                    <Radar size={16} /> Gemini Live
                  </button>
                  <button disabled={!liveRunning} onClick={stopGeminiLive} type="button">
                    <Pause size={16} /> Stop Live
                  </button>
                </div>
                <div className="live-status">
                  <strong>{liveStatus}</strong>
                  {liveEvents.map((item) => (
                    <small key={item}>{item}</small>
                  ))}
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

          {error ? (
            <div className="inline-error">
              <AlertTriangle size={18} />
              <p>{error}</p>
            </div>
          ) : null}
        </aside>
      </section>

      <section className="result-grid">
        <div className="panel result-hero">
          <div className="panel-heading">
            <h2>{analysis?.headline ?? "Structured video analytics output"}</h2>
            <span>{analysis ? `${analysis.provider} / ${analysis.model}` : modeLabel(mode)}</span>
          </div>
          <p className="summary">
            {analysis?.commentary ??
              "Run a clip to get scene understanding, person/PPE/zone alerts, timeline evidence, safety risks, and recommended human or robotic actions."}
          </p>
          {analysis ? (
            <div className="metric-grid">
              <div className="metric">
                <span>Confidence</span>
                <strong>{Math.round(analysis.confidence)}</strong>
              </div>
              <div className="metric">
                <span>Latency</span>
                <strong>{analysis.usage.latencyMs}ms</strong>
              </div>
            </div>
          ) : null}
        </div>

        <div className="panel">
          <div className="panel-heading">
            <h2>Alerts</h2>
            <ShieldCheck size={17} />
          </div>
          <div className="alert-list">
            {(analysis?.alerts ?? []).length ? (
              analysis?.alerts.map((alert) => (
                <div className={`alert-card ${alert.triggered ? "triggered" : ""}`} key={`${alert.label}-${alert.zoneId ?? "global"}`}>
                  <strong>{alert.label}</strong>
                  <span>{alert.triggered ? "Triggered" : "Not triggered"} / {alert.severity}</span>
                  <p>{alert.evidence}</p>
                </div>
              ))
            ) : (
              <p className="empty">No alerts yet.</p>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <h2>Timeline evidence</h2>
            <Radar size={17} />
          </div>
          <div className="commentary-list">
            {(analysis?.timeline ?? []).map((item) => (
              <div className="commentary-item" key={`${item.frame}-${item.observation}`}>
                <time>Frame {item.frame}</time>
                <p>{item.observation}</p>
              </div>
            ))}
            {!analysis?.timeline.length ? <p className="empty">Sampled-frame observations appear here.</p> : null}
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <h2>Actions</h2>
            <span>{analysis?.edgeAssessment.costControl ?? "browser edge preprocessing"}</span>
          </div>
          <div className="action-list">
            {(analysis?.recommendations ?? []).map((action) => (
              <div className="action-item" key={`${action.priority}-${action.action}`}>
                <span>P{action.priority} / {action.owner}</span>
                <strong>{action.action}</strong>
                <p>{action.reason}</p>
              </div>
            ))}
            {!analysis?.recommendations.length ? <p className="empty">Recommended human, automation, or robot actions appear here.</p> : null}
          </div>
        </div>
      </section>
    </main>
  );
}
