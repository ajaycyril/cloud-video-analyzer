# Architecture

Cloud Video Analyzer is organized around a provider-neutral video analytics contract.

## Flow

```text
camera/file/sample video
  -> browser keyframe sampler
  -> edge quality + motion metrics
  -> local object detections
  -> object/motion edge gate
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
- Applies an object/motion gate before any cloud reasoning request; static frames without local objects or motion are skipped.
- Optionally enriches only edge-selected frames through Roboflow hosted object detection when `ROBOFLOW_API_KEY` or `ROBOFLOW_INFERENCE_API_KEY` plus `ROBOFLOW_MODEL` are configured.
- Sends drawn zones as normalized coordinates.
- The API caps cloud reasoning calls to the best three object/motion-triggered frames.

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

## Production Considerations

- Add durable rate limiting with Redis or Vercel Firewall for public deployments.
- Persist requests/responses for auditability only with explicit customer consent.
- Keep full-video upload as an optional path for long clips; browser keyframes should remain the default low-cost path.
- Add evaluation clips per customer objective before claiming production accuracy.
