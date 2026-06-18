import { NextResponse } from "next/server";
import { z } from "zod";
import { localDetectionSchema } from "@/lib/schema";

export const runtime = "nodejs";

const MAX_IMAGE_CHARS = Number(process.env.ROBOFLOW_MAX_IMAGE_CHARS ?? 1_600_000);
const DEFAULT_CONFIDENCE = 45;
const DEFAULT_OVERLAP = 30;

const requestSchema = z.object({
  imageDataUrl: z.string().min(100).max(MAX_IMAGE_CHARS),
  timestampMs: z.number().finite().min(0).optional(),
});

const predictionSchema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  width: z.number().finite().positive(),
  height: z.number().finite().positive(),
  class: z.string().min(1),
  confidence: z.number().finite().min(0),
});

const roboflowResponseSchema = z.object({
  image: z
    .object({
      width: z.number().finite().positive(),
      height: z.number().finite().positive(),
    })
    .optional(),
  predictions: z.array(predictionSchema).default([]),
});

function typedError(error: string, detail: string, status: number) {
  return NextResponse.json({ error, detail }, { status });
}

function stripBoundarySlashes(value: string): string {
  let start = 0;
  let end = value.length;
  while (value[start] === "/") {
    start += 1;
  }
  while (end > start && value[end - 1] === "/") {
    end -= 1;
  }
  return value.slice(start, end);
}

function dataUrlToBase64(dataUrl: string): string {
  const commaIndex = dataUrl.indexOf(",");
  return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
}

function boundedPercent(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(100, Math.max(0, Math.round(parsed)));
}

function normalize(value: number, denominator: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(denominator) || denominator <= 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, value / denominator));
}

export async function POST(request: Request) {
  const apiKey = (process.env.ROBOFLOW_API_KEY || process.env.ROBOFLOW_INFERENCE_API_KEY)?.trim();
  const model = stripBoundarySlashes(process.env.ROBOFLOW_MODEL?.trim() ?? "");
  if (!apiKey || !model) {
    return NextResponse.json({
      configured: false,
      provider: "roboflow",
      model: model || null,
      detections: [],
    });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return typedError("REQUEST_JSON_INVALID", "Request body must be valid JSON.", 400);
  }

  const parsed = requestSchema.safeParse(body);
  if (!parsed.success) {
    return typedError("REQUEST_INVALID", parsed.error.issues.map((issue) => issue.message).join("; "), 400);
  }

  const confidence = boundedPercent(process.env.ROBOFLOW_CONFIDENCE, DEFAULT_CONFIDENCE);
  const overlap = boundedPercent(process.env.ROBOFLOW_OVERLAP, DEFAULT_OVERLAP);
  const url = new URL(`https://detect.roboflow.com/${model}`);
  url.searchParams.set("api_key", apiKey);
  url.searchParams.set("confidence", String(confidence));
  url.searchParams.set("overlap", String(overlap));
  url.searchParams.set("format", "json");
  url.searchParams.set("name", "sampled-frame.jpg");

  const response = await fetch(url, {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/x-www-form-urlencoded",
    },
    body: dataUrlToBase64(parsed.data.imageDataUrl),
  });

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return typedError("ROBOFLOW_RESPONSE_INVALID", "Roboflow did not return JSON predictions.", 502);
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "message" in payload
        ? String((payload as { message: unknown }).message)
        : "Roboflow request failed.";
    return typedError("ROBOFLOW_REQUEST_FAILED", detail, 502);
  }

  const parsedResponse = roboflowResponseSchema.safeParse(payload);
  if (!parsedResponse.success) {
    return typedError("ROBOFLOW_RESPONSE_INVALID", "Roboflow response did not match the expected detection schema.", 502);
  }

  const imageWidth = parsedResponse.data.image?.width ?? 1;
  const imageHeight = parsedResponse.data.image?.height ?? 1;
  const detections = parsedResponse.data.predictions
    .map((prediction) => {
      const raw = {
        label: prediction.class,
        score: prediction.confidence <= 1 ? prediction.confidence : prediction.confidence / 100,
        x: normalize(prediction.x - prediction.width / 2, imageWidth),
        y: normalize(prediction.y - prediction.height / 2, imageHeight),
        w: normalize(prediction.width, imageWidth),
        h: normalize(prediction.height, imageHeight),
      };
      return localDetectionSchema.safeParse(raw).success ? raw : null;
    })
    .filter((detection) => detection !== null);

  return NextResponse.json({
    configured: true,
    provider: "roboflow",
    model,
    detections,
  });
}
