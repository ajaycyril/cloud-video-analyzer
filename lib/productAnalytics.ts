import { track } from "@vercel/analytics";

type AnalyticsValue = string | number | boolean | null;
type AnalyticsProperties = Record<string, AnalyticsValue>;

function cleanProperties(properties: AnalyticsProperties): AnalyticsProperties {
  return Object.fromEntries(
    Object.entries(properties).filter(([, value]) => value !== null && value !== undefined),
  );
}

export function trackProductEvent(name: string, properties: AnalyticsProperties = {}): void {
  try {
    track(name, cleanProperties(properties));
  } catch {
    // Product analytics should never affect the demo path.
  }
}

