# Cloud Video Analyzer

Mobile-first physical AI and cloud video analytics demo.

Live demo: https://cloud-video-analyzer.vercel.app

This project shows how a full edge-to-cloud video analytics stack can replace brittle single-purpose camera rules. A user can point a phone camera at a scene, upload a clip, or select a demo video, describe the analytics they need in plain language, and receive structured alerts, timeline evidence, recommended actions, latency, and estimated cloud cost.

The core architecture is intentionally hybrid: the browser acts as the edge device. It does lightweight video preprocessing, local person/vehicle/object detection, and motion gating. Only object/motion-triggered JPEG keyframes and edge signals are sent to Gemini/OpenAI/NVIDIA for higher-level reasoning. The full video is not uploaded by default.

## Screenshots

### Mobile workflow

![Mobile workflow](docs/assets/mobile-workflow.png)

### Desktop workflow

![Desktop workflow](docs/assets/desktop-workflow.png)

### Structured analysis result

![Analysis result](docs/assets/analysis-result.png)

## What It Demonstrates

- Physical AI workflow design for real-world camera input.
- Browser edge preprocessing before cloud inference: local person, vehicle, object, and motion triggers decide which frames deserve cloud reasoning.
- Plain-language configurable analytics instead of fixed rules.
- Industrial-style use cases: person detection, restricted-zone alerts, PPE checks, safety hazards, operations review, queue/crowd monitoring, traffic buildup, and asset/layout review.
- Multimodal provider integration with Gemini and OpenAI Vision.
- Structured output suitable for dashboards, alert APIs, robot tasking, surveillance triage, and human review.
- Production concerns: API-key isolation, request caps, frame sampling, payload limits, rate limiting, latency/cost display, and deployable PWA UI.

## Core User Flow

1. Preview a live camera feed, upload a clip, or select a built-in demo video.
2. For live video, capture a short local burst in the browser.
3. For uploaded or sample video, sample keyframes across the full clip duration.
4. Run the object/motion edge gate and skip static frames before cloud analysis.
5. Send only triggered JPEG keyframes and metadata to the configured cloud model.
6. Render structured alerts, evidence, actions, and a browser-side annotation overlay.

## Demo Modes

- Live camera: phone or laptop camera preview, front/back camera toggle, short local capture window.
- Uploaded clip: local browser keyframe sampling across the clip; full video is not uploaded.
- Built-in clips: factory people flow, road safety, and outdoor activity samples.
- Drawn zone: resize a restricted zone directly over the video for person/zone analytics.
- Custom objective: write any analytics request in plain language.

## Architecture

```text
Camera / uploaded clip / sample video
        |
        v
Browser preprocessing
- capture window or full-clip keyframe sampling
- JPEG compression
- visual quality and motion metrics
- local person/vehicle/object detection
- object/motion edge gate: send only evidence frames
        |
        v
Server-side provider router
- request validation
- payload and rate limits
- key isolation
- Gemini / OpenAI / optional NVIDIA-compatible adapter
        |
        v
Structured video analytics response
- scene summary
- objects and evidence
- alerts and severity
- timeline observations
- recommended human, automation, or robot actions
- token usage, latency, estimated cost
```

Example: a city camera does not need to send every frame to the cloud. The browser/edge layer can detect a person, vehicle, or motion event locally, then send only those evidence frames to the cloud model. The cloud layer answers the harder question: whether traffic is piling up, whether a person entered a restricted area, what behavior changed, and what action should be taken.

## Tech Stack

- Next.js App Router
- React
- TypeScript
- Tailwind CSS v4 foundation with custom production CSS
- PWA manifest, install metadata, app icons, and mobile viewport tuning
- Gemini API via `@google/genai`
- OpenAI-compatible Responses API flow for vision frames
- MediaPipe Tasks Vision for browser-side object detection hooks
- Optional Roboflow hosted object detection enrichment
- Object/motion edge gate before cloud model calls
- Browser canvas keyframe extraction
- Vercel production deployment

## Environment

```bash
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
NVIDIA_API_KEY=
NVIDIA_MODEL=nvidia/cosmos3-nano-reasoner
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
ROBOFLOW_API_KEY=
ROBOFLOW_INFERENCE_API_KEY=
ROBOFLOW_MODEL=
ROBOFLOW_CONFIDENCE=45
ROBOFLOW_OVERLAP=30
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

Open `http://localhost:3000`.

Camera access requires a secure context in production, so phone demos should use the HTTPS Vercel URL.

## Production Notes

- API keys are server-side only.
- The browser does not upload full video by default.
- The server caps cloud analysis to selected sampled frames for latency and cost control.
- Internet-hosted videos may block canvas extraction unless CORS headers are present; uploaded files and bundled samples are reliable.
- Cost is a model-aware estimate based on returned token usage and the local pricing table. Provider dashboards remain the billing source of truth.
- There are no mock model fallbacks. Missing keys and provider failures return explicit errors.

## Verified Production Flows

- Production deployment on Vercel.
- iPhone viewport workflow.
- Live camera path with fake-camera automation.
- Built-in sample clip workflow.
- Gemini Vision analysis on road/factory/general activity clips.
- OpenAI Vision analysis on road/factory/general activity clips.
- Browser-rendered result and annotation overlay.
