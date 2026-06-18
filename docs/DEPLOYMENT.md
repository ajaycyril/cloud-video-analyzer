# Deployment

## Vercel

Set these environment variables in the Vercel project:

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

Deploy with:

```bash
vercel --prod
```

## Phone Demo

Use the HTTPS production URL. Mobile browsers generally require a secure context for camera permissions.

## Public Demo Hardening

- Keep the session demo limit enabled.
- Add provider spend limits in provider dashboards.
- Add persistent rate limiting before sharing broadly.
- Avoid logging frame payloads unless a customer explicitly agrees.
