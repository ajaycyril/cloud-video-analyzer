import type { z } from "zod";
import type {
  analysisResponseSchema,
  analyzeFrameRequestSchema,
  edgeMetricsSchema,
  localDetectionSchema,
  modeSchema,
  scanPhaseSchema,
  worldStateSchema,
} from "./schema";

export type AppMode = z.infer<typeof modeSchema>;
export type Mode = AppMode;
export type ScanPhase = z.infer<typeof scanPhaseSchema>;
export type EdgeMetrics = z.infer<typeof edgeMetricsSchema>;
export type LocalDetection = z.infer<typeof localDetectionSchema>;
export type WorldState = z.infer<typeof worldStateSchema>;
export type AnalysisResponse = z.infer<typeof analysisResponseSchema>;
export type AnalyzeFrameRequest = z.infer<typeof analyzeFrameRequestSchema>;

export type CapabilityResult = {
  supported: boolean;
  browserName: string;
  isChromeFamily: boolean;
  isSecureContext: boolean;
  hasMediaDevices: boolean;
  hasWebAssembly: boolean;
  hasCanvasImageData: boolean;
  hasWebGPU: boolean;
  issues: string[];
};
