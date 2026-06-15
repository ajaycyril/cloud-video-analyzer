import { GoogleGenAI, MediaResolution, ThinkingLevel } from "@google/genai";
import { analysisResponseSchema, geminiResponseSchema } from "./schema";
import { buildAnalysisPrompt, PHYSICAL_AI_SYSTEM_PROMPT } from "./prompts";
import type { AnalysisResponse, AnalyzeFrameRequest } from "./types";

const DEFAULT_MODEL = "gemini-3-flash-preview";
const GEMINI_3_FLASH_INPUT_USD_PER_MILLION = 0.5;
const GEMINI_3_FLASH_OUTPUT_USD_PER_MILLION = 3;

let client: GoogleGenAI | null = null;

function getClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY_MISSING");
  }
  if (!client) {
    client = new GoogleGenAI({ apiKey });
  }
  return client;
}

function dataUrlToBase64(dataUrl: string): string {
  const commaIndex = dataUrl.indexOf(",");
  if (commaIndex >= 0) {
    return dataUrl.slice(commaIndex + 1);
  }
  return dataUrl;
}

function costEstimate(inputTokens: number, outputTokens: number): number {
  return (
    (inputTokens / 1_000_000) * GEMINI_3_FLASH_INPUT_USD_PER_MILLION +
    (outputTokens / 1_000_000) * GEMINI_3_FLASH_OUTPUT_USD_PER_MILLION
  );
}

export async function analyzeFrameWithGemini(request: AnalyzeFrameRequest): Promise<AnalysisResponse> {
  const ai = getClient();
  const model = process.env.GEMINI_MODEL || DEFAULT_MODEL;
  const imageData = dataUrlToBase64(request.frameBase64);

  const response = await ai.models.generateContent({
    model,
    contents: [
      {
        role: "user",
        parts: [
          { text: buildAnalysisPrompt(request) },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: imageData,
            },
          },
        ],
      },
    ],
    config: {
      systemInstruction: PHYSICAL_AI_SYSTEM_PROMPT,
      responseMimeType: "application/json",
      responseSchema: geminiResponseSchema,
      mediaResolution: MediaResolution.MEDIA_RESOLUTION_LOW,
      thinkingConfig: {
        thinkingLevel: ThinkingLevel.LOW,
      },
      temperature: 0.25,
      maxOutputTokens: 1536,
    },
  });

  const text = response.text;
  if (!text) {
    throw new Error("MODEL_RESPONSE_EMPTY");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("MODEL_RESPONSE_INVALID");
  }
  const result = analysisResponseSchema.safeParse(parsed);
  if (!result.success) {
    throw new Error("MODEL_RESPONSE_INVALID");
  }

  const usageMetadata = response.usageMetadata;
  const inputTokens = usageMetadata?.promptTokenCount ?? 0;
  const outputTokens = usageMetadata?.candidatesTokenCount ?? 0;
  const totalTokens = usageMetadata?.totalTokenCount ?? inputTokens + outputTokens;

  return {
    ...result.data,
    usage: {
      inputTokens,
      outputTokens,
      totalTokens,
      estimatedCostUsd: costEstimate(inputTokens, outputTokens),
    },
  };
}
