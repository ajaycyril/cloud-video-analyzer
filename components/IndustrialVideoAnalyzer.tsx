"use client";

import { AlertTriangle, Camera, Crosshair, FileVideo, Pause, Play, Radar, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import { buildVideoConstraints, listVideoInputDevices, stopStream, type CameraFacing } from "@/lib/camera";
import { canvasToJpegDataUrl, captureVideoFrame, computeEdgeMetrics, type EdgeMetricResult } from "@/lib/edgeMetrics";
import { detectObjectsForVideo, tryCreateObjectDetector } from "@/lib/mediaPipeDetector";
import { fallbackObjectChips, summarizeEdgeDetections } from "@/lib/resultPresentation";
import type { LocalDetection } from "@/lib/types";
import type { ProviderId, SampledFrame, VideoAnalysisResponse, VideoMode, VideoSource, Zone } from "@/lib/videoSchema";

type ApiError = {
  error?: string;
  detail?: string;
};

type ZoneCorner = "nw" | "ne" | "sw" | "se";
type VideoOrientation = "landscape-video" | "portrait-video";
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
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>("environment");
  const [sampleUrl, setSampleUrl] = useState(SAMPLE_CLIPS[1].url);
  const [analysis, setAnalysis] = useState<VideoAnalysisResponse | null>(null);
  const [lastFrames, setLastFrames] = useState<SampledFrame[]>([]);
  const [analysesUsed, setAnalysesUsed] = useState(initialAnalysesUsed);
  const [videoOrientation, setVideoOrientation] = useState<VideoOrientation>("landscape-video");
  const [liveStatus, setLiveStatus] = useState("Gemini Live idle");
  const [liveRunning, setLiveRunning] = useState(false);
  const [liveEvents, setLiveEvents] = useState<string[]>([]);

  useEffect(() => {
    return () => {
      holdRecordingRef.current.active = false;
      holdRecordingRef.current.stopRequested = true;
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

  useEffect(() => {
    analysesUsedRef.current = analysesUsed;
  }, [analysesUsed]);

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

  const updateVideoOrientation = useCallback(() => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video.videoHeight) {
      return;
    }
    setVideoOrientation(video.videoHeight > video.videoWidth ? "portrait-video" : "landscape-video");
  }, []);

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
      const stream = await navigator.mediaDevices.getUserMedia(buildVideoConstraints("webcam_coach", deviceId, nextFacing));
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
      setRunning(true);
      setStatus(`preview only - ${nextFacing === "environment" ? "back" : "front"} camera - no cloud upload`);
    } catch (startError) {
      setRunning(false);
      setStatus("camera off - no recording");
      setError(readableError(startError));
    }
  }, [cameraFacing, clearVideoObjectUrl, refreshDevices, selectedDeviceId, updateVideoOrientation]);

  const stopCamera = useCallback(() => {
    holdRecordingRef.current.active = false;
    holdRecordingRef.current.stopRequested = true;
    setHoldRecording(false);
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
    await tryCreateObjectDetector();
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
      setLastFrames(frames);
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
  }, [mode, objective, provider, providerStatus, source, zones]);

  const analyze = useCallback(async () => {
    setAnalyzing(true);
    setError(null);
    setAnalysis(null);
    setStatus("extracting relevant frames in browser");
    try {
      const frames = await sampleFrames();
      return await submitFramesForAnalysis(frames);
    } finally {
      setAnalyzing(false);
    }
  }, [sampleFrames, submitFramesForAnalysis]);

  const beginHoldRecording = useCallback(async (event?: PointerEvent<HTMLButtonElement>) => {
    if (source !== "camera" || analyzing || holdRecordingRef.current.active) {
      return;
    }
    if (event) {
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    setError(null);
    setAnalysis(null);
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
      await tryCreateObjectDetector();
      previousFrameRef.current = null;
      setStatus("recording locally - release to analyze");

      while (holdRecordingRef.current.active && !holdRecordingRef.current.stopRequested) {
        const elapsedMs = Math.max(0, performance.now() - holdRecordingRef.current.startedAt);
        const frame = await captureOneFrame(elapsedMs);
        if (frame) {
          const nextFrames = [...holdRecordingRef.current.frames, frame].slice(-MAX_FRAMES);
          holdRecordingRef.current.frames = nextFrames;
          setLastFrames(nextFrames);
          setStatus(`recording locally - ${nextFrames.length} selected frame${nextFrames.length === 1 ? "" : "s"}`);
        }
        await new Promise((resolve) => window.setTimeout(resolve, 420));
      }

      if (!holdRecordingRef.current.frames.length) {
        const frame = await captureOneFrame(Math.max(0, performance.now() - holdRecordingRef.current.startedAt));
        if (frame) {
          holdRecordingRef.current.frames = [frame];
          setLastFrames([frame]);
        }
      }

      const frames = holdRecordingRef.current.frames;
      holdRecordingRef.current.active = false;
      holdRecordingRef.current.stopRequested = false;
      setHoldRecording(false);
      await submitFramesForAnalysis(frames);
    } catch (recordError) {
      holdRecordingRef.current.active = false;
      holdRecordingRef.current.stopRequested = false;
      setHoldRecording(false);
      setError(readableError(recordError));
      setStatus("analysis error");
    } finally {
      setAnalyzing(false);
    }
  }, [analyzing, captureOneFrame, running, source, startCamera, submitFramesForAnalysis]);

  const finishHoldRecording = useCallback((event?: PointerEvent<HTMLButtonElement>) => {
    if (event && event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (!holdRecordingRef.current.active) {
      return;
    }
    holdRecordingRef.current.stopRequested = true;
    setStatus("recording stopped - preparing cloud request");
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
    setError(null);
    setAnalysis(null);
    setLastFrames([]);
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
  const isCloudSending = analyzing && status.startsWith("sending");
  const isLiveSource = source === "camera";
  const isRecordingLocal = holdRecording && !isCloudSending && isLiveSource;
  const providerLabel = PROVIDER_COPY[provider].label;
  const sourceCaptureDetail = isLiveSource
    ? `Live camera: hold Record, sample up to ${MAX_FRAMES} selected frames locally, then send one cloud request on release.`
    : `Uploaded/demo video: sample up to ${MAX_FRAMES} keyframes across the full clip duration, then send selected JPEGs.`;
  const analyzeLabel = isLiveSource
    ? "Hold to record"
    : `Ask ${providerLabel} about this clip`;
  const cameraFacingLabel = cameraFacing === "environment" ? "Back camera" : "Front camera";
  const cameraCaptureState = isCloudSending
    ? "Sending sampled frames to cloud"
    : analyzing
      ? isLiveSource
        ? `Recording while held - release to analyze`
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
        ? `Recording locally. Release the button to send one analysis.`
        : `Sampling keyframes across the clip. Full video stays local.`
      : analysis
        ? "Cloud response received. Adjust the prompt or scene, then analyze again."
        : `Local only. ${sourceCaptureDetail}`;
  const analyzeButtonText = analyzing
    ? isCloudSending
      ? `Analyzing with ${providerLabel}...`
      : isLiveSource
        ? "Recording... release to analyze"
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
          ? "Release the record button to send one cloud request."
          : "Keep the scene steady while the browser samples frames."
        : analysis
          ? "Result card has the latest scene output."
          : "Press the red button to begin.";
  const quickPresets = OBJECTIVE_PRESETS.slice(0, 6);
  const localObjectChips = summarizeEdgeDetections(lastFrames);
  const actionPreview = analysis?.recommendations.slice(0, 3) ?? [];
  const objectChips = analysis?.objects.length ? fallbackObjectChips(analysis) : localObjectChips;
  const resultStatus = analysis ? "Scene result" : analyzing ? "Analyzing scene" : "Ready for scene result";

  return (
    <main className="industrial-shell">
      <header className="industrial-header">
        <div>
          <p className="eyebrow">Cloud video analytics API showcase</p>
          <h1>Ask Gemini/OpenAI anything about live video.</h1>
          <p className="app-subtitle">Point the camera, upload a clip, or run a sample. Safari, Edge, Chrome, and desktop uploads use local canvas frame sampling; cloud models return objects, evidence, cost, and next actions.</p>
        </div>
        <div className="loop">
          <span>video</span>
          <span>browser frame sampler</span>
          <span>plain-language objective</span>
          <span>Gemini/OpenAI Vision/Cosmos</span>
          <span>alert API</span>
        </div>
      </header>

      <section className="industrial-grid">
        <div className="video-workbench panel">
          <div
            className={`industrial-video-frame ${videoOrientation} ${source === "camera" ? "live-camera-frame" : "clip-video-frame"}`}
            onPointerDown={startZoneDraw}
            onPointerMove={updateZoneDraw}
            onPointerUp={endZoneDraw}
            ref={videoFrameRef}
          >
            <video muted onLoadedMetadata={updateVideoOrientation} onResize={updateVideoOrientation} playsInline ref={videoRef} />
            <div className={`video-processing-badge ${analyzing ? "active" : analysis ? "ready" : ""}`}>
              <strong>{videoProcessingTitle}</strong>
              <span>{videoProcessingDetail}</span>
            </div>
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
                <strong>Ask anything about this video</strong>
              </div>
              <small>{isLiveSource ? "Live camera burst" : "Full clip keyframe scan"}</small>
            </div>

            <div className={`scene-result-card ${analysis ? "ready" : analyzing ? "loading" : ""}`}>
              <div className="scene-result-topline">
                <span>{resultStatus}</span>
                <strong>{analysis ? `${Math.round(analysis.confidence)}% confidence` : analyzing ? "Working..." : "No cloud call yet"}</strong>
              </div>
              <h2>{analysis?.headline ?? "Point camera, record, get the answer here"}</h2>
              <p>{analysis?.commentary ?? "Hold the red record button, point at the scene, release, and the answer appears here. No background polling."}</p>
              <div className={`priority-actions-panel ${actionPreview.length ? "ready" : ""}`} aria-label="Recommended actions">
                <div className="priority-actions-heading">
                  <span>Recommended actions</span>
                  <strong>{actionPreview.length ? `${actionPreview.length} next step${actionPreview.length === 1 ? "" : "s"}` : "Waiting for analysis"}</strong>
                </div>
                {actionPreview.length ? (
                  <div className="priority-action-list">
                    {actionPreview.map((action) => (
                      <div className="priority-action-item" key={`${action.priority}-${action.action}`}>
                        <span>P{action.priority} / {action.owner}</span>
                        <strong>{action.action}</strong>
                        <p>{action.reason}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p>Actions will appear here first: what to do, who should do it, and why the video evidence supports it.</p>
                )}
              </div>
              <div className="scene-result-metrics">
                <div>
                  <span>Frames</span>
                  <strong>{analysis?.edgeAssessment.framesAnalyzed ?? (lastFrames.length || "--")}</strong>
                </div>
                <div>
                  <span>Latency</span>
                  <strong>{analysis ? `${analysis.usage.latencyMs}ms` : "--"}</strong>
                </div>
                <div>
                  <span>Cost</span>
                  <strong>{analysis ? formatCost(analysis.usage.estimatedCostUsd) : "$0.0000"}</strong>
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
              </div>
              {analysis ? (
                <div className="full-response-panel">
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
                </div>
              ) : null}
            </div>

            <label className="main-objective-box">
              <span>Plain-language analytics question</span>
              <textarea onChange={(event) => setObjective(event.target.value)} placeholder={DEFAULT_OBJECTIVE} value={objective} />
            </label>

            <button
              aria-pressed={holdRecording}
              className={`analyze-button first-fold-analyze ${isRecordingLocal ? "recording" : ""}`}
              disabled={source === "camera" ? isCloudSending : analyzing}
              onClick={(event) => {
                if (source === "camera") {
                  event.preventDefault();
                  return;
                }
                void recordOrAnalyze();
              }}
              onPointerCancel={(event) => {
                if (source === "camera") {
                  finishHoldRecording(event);
                }
              }}
              onPointerDown={(event) => {
                if (source === "camera") {
                  event.preventDefault();
                  void beginHoldRecording(event);
                }
              }}
              onPointerLeave={(event) => {
                if (source === "camera" && holdRecordingRef.current.pointerId === event.pointerId) {
                  finishHoldRecording(event);
                }
              }}
              onPointerUp={(event) => {
                if (source === "camera") {
                  event.preventDefault();
                  finishHoldRecording(event);
                }
              }}
              type="button"
            >
              <span className="record-dot" /> {analyzeButtonText}
            </button>

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
                  ? "Only selected JPEG keyframes are in the cloud now."
                  : `${sourceCaptureDetail} Full video is not uploaded.`}
              </span>
            </div>
          </div>

          <p className="panel-note">
            Camera mode sends one request after you release the record button. Full video stays in the browser; only selected JPEG keyframes are sent.
            Upload mode works without camera access on Safari, Edge, Chrome, and desktop browsers that support video canvas extraction.
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

          {error && !analysis ? (
            <div className="inline-error">
              <AlertTriangle size={18} />
              <p>{error}</p>
            </div>
          ) : null}
        </aside>
      </section>
    </main>
  );
}
