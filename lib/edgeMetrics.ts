import type { EdgeMetrics } from "./types";

type PreviousFrame = {
  width: number;
  height: number;
  luminance: Uint8ClampedArray;
};

export type EdgeMetricResult = {
  metrics: EdgeMetrics;
  snapshot: PreviousFrame;
};

function clampScore(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, value));
}

function luminanceAt(data: Uint8ClampedArray, pixelIndex: number): number {
  const offset = pixelIndex * 4;
  return 0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2];
}

function entropyFromBuckets(buckets: number[], total: number): number {
  if (total === 0) {
    return 0;
  }
  let entropy = 0;
  for (const count of buckets) {
    if (count === 0) {
      continue;
    }
    const probability = count / total;
    entropy -= probability * Math.log2(probability);
  }
  return entropy;
}

export function computeEdgeMetrics(imageData: ImageData, previous: PreviousFrame | null): EdgeMetricResult {
  const { width, height, data } = imageData;
  const totalPixels = width * height;
  const luminance = new Uint8ClampedArray(totalPixels);
  const buckets = Array.from({ length: 16 }, () => 0);
  let sum = 0;
  let min = 255;
  let max = 0;

  for (let i = 0; i < totalPixels; i += 1) {
    const lum = luminanceAt(data, i);
    const rounded = Math.round(lum);
    luminance[i] = rounded;
    sum += lum;
    min = Math.min(min, lum);
    max = Math.max(max, lum);
    buckets[Math.min(15, Math.floor(lum / 16))] += 1;
  }

  const mean = sum / totalPixels;
  let variance = 0;
  let gradientSum = 0;
  let edgeCount = 0;
  let compared = 0;
  let frameDiffSum = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const lum = luminance[index];
      const dx = Math.abs(lum - luminance[index - 1]);
      const dy = Math.abs(lum - luminance[index - width]);
      const gradient = dx + dy;
      gradientSum += gradient;
      if (gradient > 42) {
        edgeCount += 1;
      }
      variance += (lum - mean) * (lum - mean);

      if (previous && previous.width === width && previous.height === height) {
        frameDiffSum += Math.abs(lum - previous.luminance[index]);
        compared += 1;
      }
    }
  }

  const innerPixels = Math.max(1, (width - 2) * (height - 2));
  const brightness = clampScore((mean / 255) * 100);
  const contrast = clampScore((Math.sqrt(variance / innerPixels) / 80) * 100);
  const sharpness = clampScore((gradientSum / innerPixels / 30) * 100);
  const edgeDensity = clampScore((edgeCount / innerPixels) * 250);
  const visualComplexity = clampScore((entropyFromBuckets(buckets, totalPixels) / 4) * 100);
  const frameDifference = compared > 0 ? clampScore((frameDiffSum / compared / 48) * 100) : 0;
  const motionScore = frameDifference;
  const exposureRange = max - min;
  const stability = clampScore(100 - motionScore);

  let usable = true;
  let rejectionReason: string | null = null;
  if (brightness < 18) {
    usable = false;
    rejectionReason = "Frame is too dark for reliable analysis.";
  } else if (brightness > 88 && exposureRange < 40) {
    usable = false;
    rejectionReason = "Frame is overexposed.";
  } else if (sharpness < 10) {
    usable = false;
    rejectionReason = "Frame is too blurry.";
  } else if (motionScore > 72) {
    usable = false;
    rejectionReason = "Camera is moving too quickly.";
  }

  return {
    metrics: {
      brightness,
      contrast,
      sharpness,
      edgeDensity,
      visualComplexity,
      frameDifference,
      motionScore,
      stability,
      usable,
      rejectionReason,
    },
    snapshot: {
      width,
      height,
      luminance,
    },
  };
}

export function captureVideoFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement, maxWidth = 560): ImageData | null {
  if (!video.videoWidth || !video.videoHeight) {
    return null;
  }

  const scale = Math.min(1, maxWidth / video.videoWidth);
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    return null;
  }
  context.drawImage(video, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
}

export function canvasToJpegDataUrl(canvas: HTMLCanvasElement, quality = 0.62): string {
  return canvas.toDataURL("image/jpeg", quality);
}
