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
    };
  }

  const nav = navigator as Navigator & { userAgentData?: UserAgentDataLike; gpu?: unknown };
  const brands = nav.userAgentData?.brands;
  const browserName = brands ? browserNameFromBrands(brands) : browserNameFromUserAgent(navigator.userAgent);
  const isChromeFamily = isChromeFamilyFromBrands(brands);
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
  if (!window.isSecureContext) {
    issues.push("A secure context is required. Use HTTPS or localhost.");
  }
  if (!hasMediaDevices) {
    issues.push("Camera access is not available in this browser.");
  }
  if (!hasWebAssembly) {
    issues.push("WebAssembly is required for local MediaPipe inference.");
  }
  if (!hasCanvasImageData) {
    issues.push("Canvas ImageData is required for edge video metrics.");
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
  };
}
