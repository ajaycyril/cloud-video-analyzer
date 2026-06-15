import { GoogleGenAI, MediaResolution, ThinkingLevel } from "@google/genai";
import { estimateTokenCostUsd } from "./providerPricing";
import { VIDEO_ANALYTICS_SYSTEM_PROMPT, buildVideoAnalysisPrompt } from "./videoPrompts";
import {
  geminiVideoResponseSchema,
  videoAnalysisJsonSchema,
  videoAnalysisResponseSchema,
  type ProviderId,
  type VideoAnalysisRequest,
  type VideoAnalysisResponse,
} from "./videoSchema";

const DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview";
const DEFAULT_OPENAI_MODEL = "gpt-5.4-mini";
const DEFAULT_NVIDIA_MODEL = "nvidia/cosmos3-nano-reasoner";
const DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1";

type ModelOutput = Omit<VideoAnalysisResponse, "provider" | "model" | "usage">;

const modelOutputSchema = videoAnalysisResponseSchema.omit({
  provider: true,
  model: true,
  usage: true,
});

let geminiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY_MISSING");
  }
  if (!geminiClient) {
    geminiClient = new GoogleGenAI({ apiKey });
  }
  return geminiClient;
}

function configuredModel(provider: ProviderId): string {
  if (provider === "gemini") {
    return (process.env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL).trim();
  }
  if (provider === "openai") {
    return (process.env.OPENAI_MODEL || DEFAULT_OPENAI_MODEL).trim();
  }
  if (provider === "nvidia") {
    return (process.env.NVIDIA_MODEL || DEFAULT_NVIDIA_MODEL).trim();
  }
  return (process.env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL).trim();
}

function dataUrlToBase64(dataUrl: string): string {
  const commaIndex = dataUrl.indexOf(",");
  return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
}

function withUsage(
  provider: ProviderId,
  model: string,
  output: ModelOutput,
  request: VideoAnalysisRequest,
  inputTokens: number,
  outputTokens: number,
  latencyMs: number,
): VideoAnalysisResponse {
  const totalTokens = inputTokens + outputTokens;
  return {
    ...output,
    provider,
    model,
    edgeAssessment: {
      ...output.edgeAssessment,
      framesAnalyzed: request.frames.length,
      cloudFramesSkipped: Math.max(0, request.sampling.maxFrames - request.frames.length),
    },
    usage: {
      inputTokens,
      outputTokens,
      totalTokens,
      estimatedCostUsd: estimateTokenCostUsd(provider, model, inputTokens, outputTokens),
      latencyMs,
    },
  };
}

function parseModelJson(text: string): ModelOutput {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("MODEL_RESPONSE_INVALID_JSON");
  }

  const result = modelOutputSchema.safeParse(parsed);
  if (!result.success) {
    throw new Error("MODEL_RESPONSE_SCHEMA_MISMATCH");
  }
  return {
    ...result.data,
    confidence: result.data.confidence <= 1 ? result.data.confidence * 100 : result.data.confidence,
  };
}

function shouldRetryOpenAIError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return true;
  }
  return !(
    error.message.includes("invalid_api_key") ||
    error.message.includes("insufficient_quota") ||
    error.message.includes("model_not_found") ||
    error.message.includes("does not exist")
  );
}

export async function analyzeVideoWithGemini(request: VideoAnalysisRequest): Promise<VideoAnalysisResponse> {
  const ai = getGeminiClient();
  const model = configuredModel("gemini");
  const startedAt = Date.now();
  const response = await ai.models.generateContent({
    model,
    contents: [
      {
        role: "user",
        parts: [
          { text: buildVideoAnalysisPrompt(request) },
          ...request.frames.map((frame) => ({
            inlineData: {
              mimeType: "image/jpeg",
              data: dataUrlToBase64(frame.imageDataUrl),
            },
          })),
        ],
      },
    ],
    config: {
      systemInstruction: VIDEO_ANALYTICS_SYSTEM_PROMPT,
      responseMimeType: "application/json",
      responseSchema: geminiVideoResponseSchema,
      mediaResolution: MediaResolution.MEDIA_RESOLUTION_LOW,
      thinkingConfig: {
        thinkingLevel: ThinkingLevel.LOW,
      },
      temperature: 0.2,
      maxOutputTokens: 1800,
    },
  });

  if (!response.text) {
    throw new Error("MODEL_RESPONSE_EMPTY");
  }

  const usage = response.usageMetadata;
  return withUsage(
    "gemini",
    model,
    parseModelJson(response.text),
    request,
    usage?.promptTokenCount ?? 0,
    usage?.candidatesTokenCount ?? 0,
    Date.now() - startedAt,
  );
}

export async function analyzeVideoWithOpenAI(request: VideoAnalysisRequest): Promise<VideoAnalysisResponse> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY_MISSING");
  }

  const model = configuredModel("openai");
  const startedAt = Date.now();
  let lastError: unknown;

  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      return await requestOpenAIAnalysis(apiKey, model, request, startedAt);
    } catch (error) {
      lastError = error;
      if (attempt === 1 || !shouldRetryOpenAIError(error)) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 350));
    }
  }

  throw lastError instanceof Error ? lastError : new Error("OPENAI_REQUEST_FAILED");
}

async function requestOpenAIAnalysis(apiKey: string, model: string, request: VideoAnalysisRequest, startedAt: number): Promise<VideoAnalysisResponse> {
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      instructions: VIDEO_ANALYTICS_SYSTEM_PROMPT,
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: buildVideoAnalysisPrompt(request) },
            ...request.frames.map((frame) => ({
              type: "input_image",
              image_url: frame.imageDataUrl,
              detail: "low",
            })),
          ],
        },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "video_analysis",
          strict: true,
          schema: videoAnalysisJsonSchema,
        },
      },
      max_output_tokens: 1800,
    }),
  });

  const payload: unknown = await response.json();
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? JSON.stringify((payload as { error: unknown }).error)
        : "OPENAI_REQUEST_FAILED";
    throw new Error(message);
  }

  const typed = payload as {
    output_text?: string;
    output?: Array<{ content?: Array<{ type?: string; text?: string }> }>;
    usage?: { input_tokens?: number; output_tokens?: number };
  };
  const text =
    typed.output_text ??
    typed.output
      ?.flatMap((item) => item.content ?? [])
      .find((content) => content.type === "output_text" || typeof content.text === "string")?.text;
  if (!text) {
    throw new Error("MODEL_RESPONSE_EMPTY");
  }

  return withUsage(
    "openai",
    model,
    parseModelJson(text),
    request,
    typed.usage?.input_tokens ?? 0,
    typed.usage?.output_tokens ?? 0,
    Date.now() - startedAt,
  );
}

export async function analyzeVideoWithNvidia(request: VideoAnalysisRequest): Promise<VideoAnalysisResponse> {
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    throw new Error("NVIDIA_API_KEY_MISSING");
  }

  const model = configuredModel("nvidia");
  const baseUrl = (process.env.NVIDIA_BASE_URL || DEFAULT_NVIDIA_BASE_URL).replace(/\/$/, "");
  const startedAt = Date.now();
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "system",
          content: `${VIDEO_ANALYTICS_SYSTEM_PROMPT} Return only valid JSON with these top-level keys: ${videoAnalysisJsonSchema.required.join(", ")}.`,
        },
        {
          role: "user",
          content: [
            { type: "text", text: buildVideoAnalysisPrompt(request) },
            ...request.frames.map((frame) => ({
              type: "image_url",
              image_url: {
                url: frame.imageDataUrl,
              },
            })),
          ],
        },
      ],
      temperature: 0.2,
      max_tokens: 1800,
    }),
  });

  const payload: unknown = await response.json();
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? JSON.stringify((payload as { error: unknown }).error)
        : "NVIDIA_REQUEST_FAILED";
    throw new Error(message);
  }

  const typed = payload as {
    choices?: Array<{ message?: { content?: string } }>;
    usage?: { prompt_tokens?: number; completion_tokens?: number };
  };
  const text = typed.choices?.[0]?.message?.content;
  if (!text) {
    throw new Error("MODEL_RESPONSE_EMPTY");
  }

  return withUsage(
    "nvidia",
    model,
    parseModelJson(text),
    request,
    typed.usage?.prompt_tokens ?? 0,
    typed.usage?.completion_tokens ?? 0,
    Date.now() - startedAt,
  );
}

export async function analyzeVideo(request: VideoAnalysisRequest): Promise<VideoAnalysisResponse> {
  if (request.provider === "gemini") {
    return analyzeVideoWithGemini(request);
  }
  if (request.provider === "nvidia") {
    return analyzeVideoWithNvidia(request);
  }
  return analyzeVideoWithOpenAI(request);
}
