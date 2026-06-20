# Architecture

Cloud Video Analyzer is organized around a provider-neutral edge-to-cloud video analytics contract.

The product pattern is the same one used in scalable physical AI systems: run cheap perception at the edge first, then spend cloud reasoning only when the edge sees something worth explaining. For example, a city camera can detect people, vehicles, or motion locally, then ask a cloud model whether traffic is building up, whether a person entered a restricted area, or what action an operator should take.

The architecture is designed to demonstrate a practical future for video analytics: use local perception for continuous monitoring, use cloud vision-language models for flexible reasoning, and keep the interface plain-language so the analytic can change without retraining or rewriting a bespoke rule engine.

## Flow

```text
camera/file/sample video
  -> browser keyframe sampler
  -> edge quality + motion metrics
  -> local person / vehicle / object detections
  -> stable live detection overlay
  -> stateful object/motion/scene-change edge gate
  -> optional Roboflow specialist detections on selected frames
  -> plain-language objective + zones
  -> /api/analyze-video
  -> Gemini / OpenAI Vision / NVIDIA Cosmos adapter
  -> structured alert/evidence/action response
```

## Browser Edge Processing

The browser reduces cloud cost and latency before any API call:

- Samples up to five frames per request in the browser.
- Compresses frames as JPEG.
- Computes brightness, contrast, sharpness, edge density, motion, stability, and visual complexity.
- Runs local object detection with a bundled EfficientDet Lite model when browser support is available.
- Uses Transformers.js WebGPU object detection on capable desktop Chrome devices for a heavier local perception layer.
- Separates true object classes from heuristic motion/attention proposals, so generic edge proposals can trigger cloud review without being displayed as fake object labels.
- Stabilizes live detections so repeated sightings of the same object do not pulse or remount the bounding box.
- Maintains a qualified evidence buffer instead of treating every scanned frame as cloud-ready evidence.
- Applies an object/motion/scene-change gate before any cloud reasoning request; static or unchanged scenes are suppressed.
- Applies cooldowns and same-scene suppression after a cloud call, then waits for material motion, object, or scene change before sending again.
- Optionally enriches only edge-selected frames through Roboflow hosted object detection when `ROBOFLOW_API_KEY` or `ROBOFLOW_INFERENCE_API_KEY` plus `ROBOFLOW_MODEL` are configured.
- Sends drawn zones as normalized coordinates.
- The API caps cloud reasoning calls to the best three object/motion-triggered frames.

## Live Edge State Machine

The live camera path separates scanning, evidence, and cloud reasoning:

```text
scanning
  -> local detector sees object / strong motion / scene delta
  -> queue qualified evidence frame
  -> cloud call if provider is available and cooldown allows it
  -> hold analyzed scene signature
  -> suppress repeats until object, motion, or scene changes materially
```

This keeps the UI live while avoiding the common failure mode where a camera stream becomes a cloud-call loop. The user can still press "Ask cloud now" for a one-time manual override; that path is explicit and marked as a manual cloud request.

## Provider Contract

`lib/videoSchema.ts` defines the request and response shape. The response includes:

- `alerts`: triggered or non-triggered alerts with evidence and optional zone ID.
- `events`: timestamped observations.
- `risks`: safety and operational risks.
- `recommendations`: actions assigned to human, automation, or robot.
- `usage`: tokens, latency, and estimated cost.

## Provider Adapters

`lib/videoProviders.ts` contains separate Gemini, OpenAI Vision, and NVIDIA Cosmos adapters. All providers return through the same Zod-validated response contract. Invalid JSON or schema mismatches fail the request instead of being silently repaired; OpenAI gets one bounded retry for transient provider or schema-contract failures.

OpenAI uses the Responses API with `input_text` for the analytics objective and `input_image` entries for sampled video frames. It is not a native full-video upload path.

NVIDIA Cosmos is integrated as a physical-world reasoning provider using the NVIDIA NIM endpoint shape. Its default model is `nvidia/cosmos3-nano-reasoner`, which NVIDIA Build lists as a free endpoint for development and a vision-language model for physical-world reasoning on video or images.

Roboflow is integrated as a specialist detection layer, not a general reasoning provider. The browser still samples frames locally, then `/api/roboflow-detect` can add detector boxes before the selected frames are sent to Gemini/OpenAI/NVIDIA for natural-language analysis, alerts, and recommended actions. Roboflow requires an API key and a concrete hosted model endpoint such as `project/version`.

## Why This Matters

Traditional video analytics often ships as a fixed detector plus hand-written rules. That works for narrow tasks, but it does not adapt quickly when the user changes the question from "is there a person?" to "is this area becoming unsafe?" or "what should an operator do next?"

This project demonstrates a different pattern:

- Edge perception answers "did anything worth looking at happen?"
- Cloud multimodal reasoning answers "what does it mean?"
- A typed response contract makes the answer usable by dashboards, alerts, agents, and robotics workflows.
- Plain-language objectives make the analytic configurable at runtime.

## Production Considerations

- Add durable rate limiting with Redis or Vercel Firewall for public deployments.
- Persist requests/responses for auditability only with explicit customer consent.
- Keep full-video upload as an optional path for long clips; browser keyframes should remain the default low-cost path.
- Add evaluation clips per customer objective before claiming production accuracy.
