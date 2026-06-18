import type { ProviderId } from "./videoSchema";

type TokenPrice = {
  inputUsdPerMillion: number;
  outputUsdPerMillion: number;
};

const DEFAULT_PRICES: Record<ProviderId, TokenPrice> = {
  gemini: {
    inputUsdPerMillion: 1.5,
    outputUsdPerMillion: 9,
  },
  openai: {
    inputUsdPerMillion: 0.75,
    outputUsdPerMillion: 4.5,
  },
  nvidia: {
    inputUsdPerMillion: 0,
    outputUsdPerMillion: 0,
  },
};

const OPENAI_MODEL_PRICES: Record<string, TokenPrice> = {
  "gpt-4o-mini": {
    inputUsdPerMillion: 0.15,
    outputUsdPerMillion: 0.6,
  },
  "gpt-5.4-mini": {
    inputUsdPerMillion: 0.75,
    outputUsdPerMillion: 4.5,
  },
};

const GEMINI_MODEL_PRICES: Record<string, TokenPrice> = {
  "gemini-3.5-flash": {
    inputUsdPerMillion: 1.5,
    outputUsdPerMillion: 9,
  },
  "gemini-3-flash-preview": {
    inputUsdPerMillion: 0.5,
    outputUsdPerMillion: 3,
  },
  "gemini-3.1-flash-lite": {
    inputUsdPerMillion: 0.25,
    outputUsdPerMillion: 1.5,
  },
  "gemini-3.1-flash-live-preview": {
    inputUsdPerMillion: 1,
    outputUsdPerMillion: 4.5,
  },
};

export function estimateTokenCostUsd(provider: ProviderId, model: string, inputTokens: number, outputTokens: number): number {
  const normalizedModel = model.trim().toLowerCase();
  const price =
    provider === "openai"
      ? OPENAI_MODEL_PRICES[normalizedModel] ?? DEFAULT_PRICES.openai
      : provider === "gemini"
        ? GEMINI_MODEL_PRICES[normalizedModel] ?? DEFAULT_PRICES.gemini
        : DEFAULT_PRICES[provider];
  return (inputTokens / 1_000_000) * price.inputUsdPerMillion + (outputTokens / 1_000_000) * price.outputUsdPerMillion;
}
