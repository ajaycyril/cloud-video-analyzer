# Cloud Video Analyzer

Mobile-first edge-to-cloud video analytics platform for physical AI demos.

Live demo: https://cloud-video-analyzer.vercel.app

Cloud Video Analyzer shows how general-purpose multimodal AI can turn cameras into programmable analytics systems. A user can point a phone camera at a scene, upload a clip, or select a demo video, type the analytics they need in plain language, and receive structured alerts, scene evidence, recommended actions, latency, and estimated cloud cost.

The core architecture is intentionally hybrid: the browser acts as the edge device. It performs live object detection, motion analysis, scene-quality checks, evidence buffering, and stateful edge gating. Only qualified evidence frames are sent to Gemini/OpenAI/NVIDIA for higher-level reasoning. The full video is not uploaded by default.

This is the core thesis: most video analytics should not be a hard-coded rule for one camera problem. Edge perception can cheaply decide when something changed, and cloud vision-language models can reason about what it means.

## Screenshots

### Mobile workflow

![Mobile workflow](docs/assets/mobile-workflow.png)

### Desktop workflow

![Desktop workflow](docs/assets/desktop-workflow.png)

### Structured analysis result

![Analysis result](docs/assets/analysis-result.png)

## What It Demonstrates

- Physical AI product architecture for real camera streams, not just static image prompting.
- Edge-first video analytics: browser-side object detection, motion scoring, visual quality checks, and scene-change gating before cloud inference.
- Stateful live perception: stable bounding boxes stay on screen without noisy pulsing, and evidence frames do not pile up while the scene is unchanged.
- Plain-language analytics: users ask for person-in-zone, PPE, queue buildup, safety risk, traffic behavior, or any custom objective without rebuilding the pipeline.
- Multimodal provider routing across Gemini, OpenAI Vision, optional NVIDIA-compatible physical-world reasoning, and optional Roboflow detector enrichment.
- Structured outputs for dashboards, alert APIs, operator workflows, robot tasking, surveillance triage, and executive review.
- Production-grade concerns: API-key isolation, request caps, payload limits, model-aware cost estimates, no mock fallbacks, PWA install metadata, and Vercel deployment.

## Core User Flow

1. Preview a live camera feed, upload a clip, or select a built-in demo video.
2. The browser runs local object detection and motion/quality metrics continuously.
3. The UI draws stable live bounding boxes for edge detections.
4. The edge gate queues evidence only when an object, motion event, or material scene change occurs.
5. The app sends only selected JPEG evidence frames plus edge metadata to the configured cloud model.
6. Gemini/OpenAI/NVIDIA returns structured scene reasoning, alerts, evidence, recommended actions, usage, latency, and estimated cost.
7. The UI shows live edge detections, cloud-ready evidence, cloud-confirmed objects, and operator actions in one workflow.

## Demo Modes

- Live camera: phone or laptop camera preview, front/back camera toggle, continuous edge detection, stateful cloud gating, and manual "Ask cloud now" override.
- Uploaded clip: local browser keyframe sampling across the clip; full video is not uploaded.
- Built-in clips: factory people flow, road safety, outdoor activity, and general motion/activity samples.
- Drawn zone: resize a restricted zone directly over the video for person/zone analytics.
- Custom objective: write any analytics request in plain language.

## Capability Map

| Layer | Capability | Why it matters |
| --- | --- | --- |
| Browser edge | MediaPipe EfficientDet object detection, motion grid, visual quality metrics, scene delta scoring | Reduces cloud calls and makes the demo feel live |
| Edge gate | Object/motion/scene-change evidence selection, cooldowns, same-scene suppression | Prevents cloud flooding and controls cost |
| Cloud reasoning | Gemini/OpenAI/NVIDIA-compatible multimodal analysis over selected frames | Converts visual evidence into human-readable decisions |
| Specialist detection | Optional Roboflow enrichment on selected frames | Adds domain detector hooks without replacing the core architecture |
| UX | Stable bounding boxes, first-fold result card, cost/latency counters, manual override | Makes the edge-to-cloud workflow understandable to non-technical users |
| API contract | Zod-validated structured alerts, risks, events, recommendations, usage | Makes outputs usable by dashboards, agents, and downstream automation |

## Architecture

```text
Camera / uploaded clip / sample video
        |
        v
Browser preprocessing
- live camera loop or full-clip keyframe sampling
- JPEG compression
- visual quality and motion metrics
- local person/vehicle/object detection
- stable bounding boxes and local detection summaries
- stateful object/motion/scene-change gate
- same-scene suppression and cooldowns
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

## Representative Use Cases

- City surveillance: edge detects people/vehicles; cloud reasons about congestion, unsafe behavior, or escalation priority.
- Industrial safety: edge detects people/motion; cloud evaluates PPE, restricted-zone entry, blocked walkways, and supervisor actions.
- Operations monitoring: edge filters static frames; cloud explains queue buildup, bottlenecks, and layout changes.
- Robotics supervision: edge selects relevant visual evidence; cloud turns perception into taskable recommendations for a human, automation system, or robot.
- General video analytics: a user uploads a clip, writes the objective, and receives a structured analysis without building a custom model pipeline.

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
- Stateful detection stabilization for non-pulsing live bounding boxes
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
- The browser maintains a qualified evidence buffer; steady scenes do not keep adding cloud-bound frames.
- The server caps cloud analysis to selected sampled frames for latency and cost control.
- Manual cloud override is explicit and visible in the UI.
- Vercel Web Analytics tracks pageviews and privacy-light product events such as stream starts, uploads, sample selection, provider selection, manual cloud requests, edge-gate blocks, and cloud analysis outcomes.
- Internet-hosted videos may block canvas extraction unless CORS headers are present; uploaded files and bundled samples are reliable.
- Cost is a model-aware estimate based on returned token usage and the local pricing table. Provider dashboards remain the billing source of truth.
- There are no mock model fallbacks. Missing keys and provider failures return explicit errors.

## Verified Production Flows

- Production deployment on Vercel.
- iPhone viewport workflow.
- Live camera path with fake-camera automation.
- Stateful edge evidence gating: steady scenes scan locally without repeated cloud calls.
- Stable live bounding-box overlay.
- Manual cloud override path.
- Built-in sample clip workflow.
- Gemini Vision analysis on road/factory/general activity clips.
- OpenAI Vision analysis on road/factory/general activity clips.
- Browser-rendered result and annotation overlay.
