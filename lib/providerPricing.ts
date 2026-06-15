import type { ProviderId } from "./videoSchema";

type TokenPrice = {
  inputUsdPerMillion: number;
  outputUsdPerMillion: number;
};

const PRICES: Record<ProviderId, TokenPrice> = {
  gemini: {
    inputUsdPerMillion: 0.5,
    outputUsdPerMillion: 3,
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

export function estimateTokenCostUsd(provider: ProviderId, inputTokens: number, outputTokens: number): number {
  const price = PRICES[provider];
  return (inputTokens / 1_000_000) * price.inputUsdPerMillion + (outputTokens / 1_000_000) * price.outputUsdPerMillion;
}
