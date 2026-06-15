# Implementation Notes

## API Route

`app/api/analyze-video/route.ts` validates request size, rate limits, schema, and frame usability before calling a provider.

## Browser Sampler

`components/IndustrialVideoAnalyzer.tsx` samples frames from:

- `getUserMedia` camera streams.
- Local uploaded video files.
- CORS-enabled MP4 URLs.

The sampler sends JPEG keyframes, edge metrics, local detections, plain-language objective, and zones.

## No Silent Fallbacks

The app does not fabricate analytics if a model fails. Missing API keys, invalid model output, tainted canvas, and bad requests are visible errors.

## Extending Providers

Add a new provider by:

1. Extending `providerSchema`.
2. Adding pricing in `providerPricing.ts`.
3. Implementing an adapter in `videoProviders.ts`.
4. Adding provider status in `app/page.tsx`.
