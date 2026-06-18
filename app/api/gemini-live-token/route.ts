import { GoogleGenAI, Modality } from "@google/genai";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const DEFAULT_LIVE_MODEL = "gemini-3.1-flash-live-preview";

function typedError(error: string, detail: string, status: number) {
  return NextResponse.json({ error, detail }, { status });
}

export async function POST() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return typedError("GEMINI_API_KEY_MISSING", "Add GEMINI_API_KEY before starting Gemini Live.", 500);
  }

  const model = process.env.GEMINI_LIVE_MODEL || DEFAULT_LIVE_MODEL;
  const expireTime = new Date(Date.now() + 30 * 60 * 1000).toISOString();
  const newSessionExpireTime = new Date(Date.now() + 60 * 1000).toISOString();

  try {
    const client = new GoogleGenAI({
      apiKey,
      httpOptions: { apiVersion: "v1alpha" },
    } as ConstructorParameters<typeof GoogleGenAI>[0]);
    const token = await client.authTokens.create({
      config: {
        uses: 1,
        expireTime,
        newSessionExpireTime,
        liveConnectConstraints: {
          model,
          config: {
            responseModalities: [Modality.AUDIO],
            outputAudioTranscription: {},
            temperature: 0.2,
          },
        },
        httpOptions: { apiVersion: "v1alpha" },
      },
    });

    return NextResponse.json({
      token: token.name,
      model,
      expiresAt: expireTime,
    });
  } catch (error) {
    const detail = error instanceof Error && process.env.NODE_ENV !== "production" ? error.message : "Could not create Gemini Live token.";
    return typedError("GEMINI_LIVE_TOKEN_FAILED", detail, 502);
  }
}
