# Cloud Video Analyzer

Production-oriented browser and cloud demo for configurable industrial video analytics.

The core idea: a customer writes the analytics they need in plain language, the browser extracts a small number of relevant frames plus edge signals, and Gemini or OpenAI Vision returns structured alerts, evidence, risks, and recommended actions. This shows how general-purpose multimodal APIs can cover many video analytics workflows that previously required narrow custom models.

## Use Cases

- Person detection and restricted-zone alerts
- PPE visibility checks for hard hats, vests, gloves, and uncertainty reporting
- Industrial safety hazard review
- Operations monitoring and bottleneck summaries
- Queue/crowd buildup and asset/layout review
- Customer-defined analytics objectives without retraining

## Demo Modes

- Live camera: point a phone or laptop camera at a scene, choose a prompt, and analyze sampled frames.
- Upload: upload site footage; the browser samples frames locally instead of sending the full video.
- Built-in clips: factory people, road safety, and outdoor activity samples are bundled under `public/samples`.
- Drawn zones: drag a restricted zone on top of the video before running person/zone analytics.

## Architecture

1. Video source: live webcam, uploaded video, or CORS-enabled MP4 URL.
2. Browser edge preprocessing: keyframe sampling, compression, quality checks, motion/complexity metrics, optional local object detection.
3. Customer objective: plain-language analytics request plus optional drawn zones.
4. Provider router: Gemini, OpenAI Vision through the Responses API, or NVIDIA Cosmos server-side adapter.
5. Structured response: alerts, evidence, timeline, risks, recommended human/automation/robot actions, usage, latency, and cost estimate.

## Environment

```bash
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3-flash-preview
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
NVIDIA_API_KEY=
NVIDIA_MODEL=nvidia/cosmos3-nano-reasoner
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NEXT_PUBLIC_DEMO_ANALYSIS_LIMIT=20
DEMO_MAX_REQUEST_BYTES=2500000
DEMO_RATE_LIMIT_WINDOW_MS=60000
DEMO_RATE_LIMIT_MAX_REQUESTS=24
```

## Local Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. Camera access requires a secure context in production; use the deployed HTTPS URL for phone demos.

## Notes

- The browser does not upload full video by default. It sends sampled JPEG frames and metadata.
- The server caps cloud analysis to the best three sampled frames per request for latency and cost control.
- Internet-hosted videos often block canvas access unless they provide CORS headers. Uploaded local clips are the reliable demo path.
- The frontend includes PWA metadata and a standalone manifest for phone demos.
- `public/samples` includes reproducible demo media for factory people flow, road safety review, outdoor activity analysis, and person/zone API tests.
- Cloud cost is a model-aware estimate based on returned token usage and the local pricing table. Provider dashboards remain the billing source of truth.
- API keys are only read server-side.
- NVIDIA Cosmos integration targets the NIM/OpenAI-compatible endpoint and is isolated in one adapter for key-based validation.
- There are no mock model fallbacks. Missing keys and model failures return explicit errors.

## Verified Production Flows

- Chrome capability check: HTTPS, camera API, WebAssembly, canvas, and WebGPU capability detected.
- Gemini Vision: uploaded road, factory/person, and general activity clips.
- OpenAI Vision: uploaded road, factory/person, and general activity clips through sampled image frames.
- iPhone viewport: uploaded clips and built-in sample-card flow.
- Camera path: verified with Chrome fake camera in production automation; physical phone camera access requires the user to grant browser permission.
