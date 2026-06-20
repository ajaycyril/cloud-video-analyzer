# Implementation Notes

## API Route

`app/api/analyze-video/route.ts` validates request size, rate limits, schema, and frame usability before calling a provider.

## Browser Sampler

`components/IndustrialVideoAnalyzer.tsx` samples frames from:

- `getUserMedia` camera streams.
- Local uploaded video files.
- CORS-enabled MP4 URLs.

The sampler sends JPEG keyframes, edge metrics, local detections, plain-language objective, and zones.

## Live Edge Intelligence

The live camera loop separates three concepts:

- Scanned frames: low-cost local frames inspected by the browser.
- Evidence frames: qualified frames that contain object, strong motion, or material scene-change signals.
- Cloud frames: selected evidence frames sent to the provider for higher-level reasoning.

Bounding boxes are stabilized across repeated detections of the same object, so live overlays do not pulse just because confidence or coordinates jitter slightly. After a cloud call, the app keeps the last analyzed scene signature and suppresses repeat calls until the camera view changes meaningfully.

## No Silent Fallbacks

The app does not fabricate analytics if a model fails. Missing API keys, invalid model output, tainted canvas, and bad requests are visible errors.

## Extending Providers

Add a new provider by:

1. Extending `providerSchema`.
2. Adding pricing in `providerPricing.ts`.
3. Implementing an adapter in `videoProviders.ts`.
4. Adding provider status in `app/page.tsx`.
