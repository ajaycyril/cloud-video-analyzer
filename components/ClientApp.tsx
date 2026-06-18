"use client";

import { useEffect, useState } from "react";
import { CapabilityGate } from "./CapabilityGate";
import { IndustrialVideoAnalyzer } from "./IndustrialVideoAnalyzer";
import { getBrowserCapabilities } from "@/lib/browserCapabilities";
import type { CapabilityResult } from "@/lib/types";
import type { ProviderId } from "@/lib/videoSchema";

export function ClientApp({ providerStatus, roboflowReady }: { providerStatus: Record<ProviderId, boolean>; roboflowReady: boolean }) {
  const [capabilityResult, setCapabilityResult] = useState<CapabilityResult | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setCapabilityResult(getBrowserCapabilities());
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <CapabilityGate result={capabilityResult}>
      <IndustrialVideoAnalyzer providerStatus={providerStatus} roboflowReady={roboflowReady} />
    </CapabilityGate>
  );
}
