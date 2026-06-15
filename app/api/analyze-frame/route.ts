import { NextResponse } from "next/server";
import { analyzeFrameWithGemini } from "../../../lib/gemini";
import { analyzeFrameRequestSchema } from "../../../lib/schema";

export const runtime = "nodejs";

const MAX_REQUEST_BYTES = Number(process.env.DEMO_MAX_REQUEST_BYTES ?? 1_250_000);
const RATE_LIMIT_WINDOW_MS = Number(process.env.DEMO_RATE_LIMIT_WINDOW_MS ?? 60_000);
const RATE_LIMIT_MAX_REQUESTS = Number(process.env.DEMO_RATE_LIMIT_MAX_REQUESTS ?? 18);

type RateLimitBucket = {
  count: number;
  resetAt: number;
};

const rateLimitBuckets = new Map<string, RateLimitBucket>();

function typedError(error: string, detail: string, status: number) {
  return NextResponse.json({ error, detail }, { status });
}

function modelFailureDetail(message: string): string {
  if (process.env.NODE_ENV !== "production") {
    return message;
  }
  return "Model request failed. Try a stable, well-lit frame.";
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

export async function POST(request: Request) {
  if (!process.env.GEMINI_API_KEY) {
    return typedError("GEMINI_API_KEY_MISSING", "Add GEMINI_API_KEY to run physical reasoning.", 500);
  }

  if (!checkRateLimit(request)) {
    return typedError("RATE_LIMITED", "Demo request limit reached. Wait briefly before scanning again.", 429);
  }

  let body: unknown;
  try {
    const rawBody = await request.text();
    if (rawBody.length > MAX_REQUEST_BYTES) {
      return typedError("REQUEST_TOO_LARGE", "Frame payload is too large for this demo endpoint.", 413);
    }
    body = JSON.parse(rawBody);
  } catch {
    return typedError("REQUEST_JSON_INVALID", "Request body must be valid JSON.", 400);
  }

  const parsed = analyzeFrameRequestSchema.safeParse(body);
  if (!parsed.success) {
    return typedError("REQUEST_INVALID", parsed.error.issues.map((issue) => issue.message).join("; "), 400);
  }

  if (!parsed.data.edgeMetrics.usable) {
    return typedError(
      "FRAME_NOT_USABLE",
      parsed.data.edgeMetrics.rejectionReason ?? "The selected frame is not usable for analysis.",
      422,
    );
  }

  try {
    const analysis = await analyzeFrameWithGemini(parsed.data);
    return NextResponse.json(analysis);
  } catch (error) {
    const message = error instanceof Error ? error.message : "UNKNOWN_ERROR";
    if (message === "MODEL_RESPONSE_INVALID") {
      return typedError("MODEL_RESPONSE_INVALID", "Gemini response did not match schema.", 502);
    }
    if (message === "MODEL_RESPONSE_EMPTY") {
      return typedError("MODEL_RESPONSE_EMPTY", "Gemini returned an empty response.", 502);
    }
    return typedError("GEMINI_REQUEST_FAILED", modelFailureDetail(message), 502);
  }
}
