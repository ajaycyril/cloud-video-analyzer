# Cloud Video Analyzer

Production-oriented browser and cloud demo for configurable industrial video analytics.

The core idea: a customer writes the analytics they need in plain language, the browser extracts a small number of relevant frames plus edge signals, and Gemini or OpenAI returns structured alerts, evidence, risks, and recommended actions. This shows how general-purpose multimodal APIs can cover many video analytics workflows that previously required narrow custom models.

## Use Cases

- Person detection and restricted-zone alerts
- PPE visibility checks for hard hats, vests, gloves, and uncertainty reporting
- Industrial safety hazard review
- Operations monitoring and bottleneck summaries
- Customer-defined analytics objectives without retraining

## Architecture

1. Video source: live webcam, uploaded video, or CORS-enabled MP4 URL.
2. Browser edge preprocessing: keyframe sampling, compression, quality checks, motion/complexity metrics, optional local object detection.
3. Customer objective: plain-language analytics request plus optional drawn zones.
4. Provider router: Gemini, OpenAI, or NVIDIA Cosmos server-side adapter.
5. Structured response: alerts, evidence, timeline, risks, recommended human/automation/robot actions, usage, latency, and cost estimate.

## Environment

```bash
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3-flash-preview
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
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
- Internet-hosted videos often block canvas access unless they provide CORS headers. Uploaded local clips are the reliable demo path.
- The frontend includes PWA metadata and a standalone manifest for phone demos.
- API keys are only read server-side.
- NVIDIA Cosmos integration targets the NIM/OpenAI-compatible endpoint and is isolated in one adapter for key-based validation.
- There are no mock model fallbacks. Missing keys and model failures return explicit errors.
