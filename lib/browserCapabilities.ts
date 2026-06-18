import type { CapabilityResult } from "./types";

type UserAgentDataLike = {
  brands?: Array<{ brand: string; version: string }>;
};

function browserNameFromBrands(brands: Array<{ brand: string }> | undefined): string {
  if (!brands || brands.length === 0) {
    return "Unknown";
  }
  const edge = brands.find((brand) => brand.brand === "Microsoft Edge");
  if (edge) {
    return "Microsoft Edge";
  }
  const chrome = brands.find((brand) => brand.brand === "Google Chrome" || brand.brand === "Chromium");
  return chrome?.brand ?? brands[0]?.brand ?? "Unknown";
}

function browserNameFromUserAgent(userAgent: string): string {
  if (userAgent.includes("CriOS")) {
    return "Chrome on iOS";
  }
  if (userAgent.includes("Edg/")) {
    return "Microsoft Edge";
  }
  if (userAgent.includes("Chrome/") || userAgent.includes("Chromium/") || userAgent.includes("HeadlessChrome/")) {
    return "Google Chrome";
  }
  if (userAgent.includes("FxiOS")) {
    return "Firefox on iOS";
  }
  if (userAgent.includes("EdgiOS")) {
    return "Edge on iOS";
  }
  if (userAgent.includes("Safari") && userAgent.includes("Mobile")) {
    return "Mobile Safari";
  }
  if (userAgent.includes("Safari")) {
    return "Safari";
  }
  return "Unknown";
}

function isChromeFamilyFromBrands(brands: Array<{ brand: string }> | undefined): boolean {
  if (!brands) {
    return false;
  }
  return brands.some((brand) => brand.brand === "Google Chrome" || brand.brand === "Chromium" || brand.brand === "Microsoft Edge");
}

function isChromeFamilyFromUserAgent(userAgent: string): boolean {
  return userAgent.includes("Chrome/") || userAgent.includes("Chromium/") || userAgent.includes("HeadlessChrome/") || userAgent.includes("Edg/");
}

export function getBrowserCapabilities(): CapabilityResult {
  if (typeof window === "undefined") {
    return {
      supported: false,
      browserName: "Server",
      isChromeFamily: false,
      isSecureContext: false,
      hasMediaDevices: false,
      hasWebAssembly: false,
      hasCanvasImageData: false,
      hasWebGPU: false,
      issues: ["Browser APIs are not available on the server."],
      warnings: [],
    };
  }

  const nav = navigator as Navigator & { userAgentData?: UserAgentDataLike; gpu?: unknown };
  const brands = nav.userAgentData?.brands;
  const browserName = brands ? browserNameFromBrands(brands) : browserNameFromUserAgent(navigator.userAgent);
  const isChromeFamily = brands ? isChromeFamilyFromBrands(brands) : isChromeFamilyFromUserAgent(navigator.userAgent);
  const hasMediaDevices = Boolean(nav.mediaDevices?.getUserMedia);
  const hasWebAssembly = typeof WebAssembly !== "undefined";
  const hasWebGPU = Boolean(nav.gpu);
  let hasCanvasImageData = false;

  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const context = canvas.getContext("2d");
  if (context) {
    const imageData = context.getImageData(0, 0, 1, 1);
    hasCanvasImageData = imageData.data.length === 4;
  }

  const issues: string[] = [];
  const warnings: string[] = [];
  if (!hasCanvasImageData) {
    issues.push("Canvas ImageData is required to sample frames from camera, uploaded clips, or demo videos.");
  }
  if (!window.isSecureContext) {
    warnings.push("Camera access requires HTTPS or localhost. Uploaded clips can still be analyzed when canvas is available.");
  }
  if (!hasMediaDevices) {
    warnings.push("Camera access is not available in this browser. Upload a clip or use a demo video.");
  }
  if (!hasWebAssembly) {
    warnings.push("Local object hints are disabled because WebAssembly is unavailable. Cloud analysis still works from sampled frames.");
  }

  return {
    supported: issues.length === 0,
    browserName,
    isChromeFamily,
    isSecureContext: window.isSecureContext,
    hasMediaDevices,
    hasWebAssembly,
    hasCanvasImageData,
    hasWebGPU,
    issues,
    warnings,
  };
}
