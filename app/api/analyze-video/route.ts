import { NextResponse } from "next/server";
import { analyzeVideo } from "@/lib/videoProviders";
import { videoAnalysisRequestSchema, type SampledFrame } from "@/lib/videoSchema";

export const runtime = "nodejs";

const MAX_REQUEST_BYTES = Number(process.env.DEMO_MAX_REQUEST_BYTES ?? 2_500_000);
const RATE_LIMIT_WINDOW_MS = Number(process.env.DEMO_RATE_LIMIT_WINDOW_MS ?? 60_000);
const RATE_LIMIT_MAX_REQUESTS = Number(process.env.DEMO_RATE_LIMIT_MAX_REQUESTS ?? 24);

type RateLimitBucket = {
  count: number;
  resetAt: number;
};

const rateLimitBuckets = new Map<string, RateLimitBucket>();

function typedError(error: string, detail: string, status: number) {
  return NextResponse.json({ error, detail }, { status });
}

function clientKey(request: Request): string {
  const forwardedFor = request.headers.get("x-forwarded-for");
  if (forwardedFor) {
    return forwardedFor.split(",")[0]?.trim() || "unknown";
  }
  return request.headers.get("x-real-ip") ?? "unknown";
}

function checkRateLimit(request: Request): boolean {
  const now = Date.now();
  const key = clientKey(request);
  const bucket = rateLimitBuckets.get(key);

  if (!bucket || bucket.resetAt <= now) {
    rateLimitBuckets.set(key, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return true;
  }

  if (bucket.count >= RATE_LIMIT_MAX_REQUESTS) {
    return false;
  }

  bucket.count += 1;
  return true;
}

function publicModelError(message: string): string {
  if (message.endsWith("_MISSING")) {
    return message;
  }
  if (message.includes("insufficient_quota")) {
    return "Provider quota is exhausted or billing is not enabled for this API key.";
  }
  if (message.includes("invalid_api_key")) {
    return "Provider API key is invalid.";
  }
  if (message.includes("model_not_found") || message.includes("does not exist")) {
    return "Configured provider model is not available for this API key.";
  }
  if (process.env.NODE_ENV !== "production" || process.env.DEMO_EXPOSE_MODEL_ERRORS === "true") {
    return message;
  }
  return "Cloud model request failed. Try fewer frames or a clearer clip.";
}

function modelErrorCode(message: string): string {
  if (message.endsWith("_MISSING")) {
    return message;
  }
  if (message.includes("insufficient_quota")) {
    return "PROVIDER_QUOTA";
  }
  if (message.includes("invalid_api_key")) {
    return "PROVIDER_KEY_INVALID";
  }
  if (message.includes("model_not_found") || message.includes("does not exist")) {
    return "PROVIDER_MODEL_UNAVAILABLE";
  }
  if (message.includes("MODEL_RESPONSE_SCHEMA_MISMATCH")) {
    return "MODEL_RESPONSE_SCHEMA_MISMATCH";
  }
  if (message.includes("MODEL_RESPONSE_INVALID_JSON")) {
    return "MODEL_RESPONSE_INVALID_JSON";
  }
  return "MODEL_REQUEST_FAILED";
}

function frameQualityScore(frame: SampledFrame): number {
  return frame.edgeMetrics.sharpness + frame.edgeMetrics.visualComplexity + frame.edgeMetrics.brightness * 0.25 - frame.edgeMetrics.motionScore * 0.35;
}

function selectCloudFrames(frames: SampledFrame[]): SampledFrame[] {
  const targetFrameCount = Math.min(frames.length, 3);
  const usableFrames = frames.filter((frame) => frame.edgeMetrics.usable);
  const candidateFrames = usableFrames.length ? usableFrames : frames;
  const selected = new Set(
    candidateFrames
      .slice()
      .sort((left, right) => frameQualityScore(right) - frameQualityScore(left))
      .slice(0, targetFrameCount),
  );

  return frames.filter((frame) => selected.has(frame));
}

export async function POST(request: Request) {
  if (!checkRateLimit(request)) {
    return typedError("RATE_LIMITED", "Demo request limit reached. Wait briefly before analyzing again.", 429);
  }

  let body: unknown;
  try {
    const rawBody = await request.text();
    if (rawBody.length > MAX_REQUEST_BYTES) {
      return typedError("REQUEST_TOO_LARGE", "Sampled frame payload is too large for this demo endpoint.", 413);
    }
    body = JSON.parse(rawBody);
  } catch {
    return typedError("REQUEST_JSON_INVALID", "Request body must be valid JSON.", 400);
  }

  const parsed = videoAnalysisRequestSchema.safeParse(body);
  if (!parsed.success) {
    return typedError("REQUEST_INVALID", parsed.error.issues.map((issue) => issue.message).join("; "), 400);
  }

  const cloudFrames = selectCloudFrames(parsed.data.frames);

  try {
    const analysis = await analyzeVideo({
      ...parsed.data,
      frames: cloudFrames,
    });
    return NextResponse.json(analysis);
  } catch (error) {
    const message = error instanceof Error ? error.message : "MODEL_REQUEST_FAILED";
    console.warn("video analysis provider error", {
      provider: parsed.data.provider,
      mode: parsed.data.mode,
      source: parsed.data.source,
      frames: cloudFrames.length,
      code: modelErrorCode(message),
    });
    return typedError("MODEL_REQUEST_FAILED", publicModelError(message), 502);
  }
}
