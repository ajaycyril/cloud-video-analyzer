import type { LocalDetection } from "./types";

export function isProposalDetectionLabel(label: string): boolean {
  const normalized = label.toLowerCase();
  return normalized.includes("proposal") || normalized.includes("region") || normalized.includes("attention") || normalized.includes("candidate");
}

export function isRealObjectDetection(detection: LocalDetection): boolean {
  return !isProposalDetectionLabel(detection.label);
}
