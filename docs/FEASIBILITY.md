# Feasibility And Cost

General-purpose multimodal APIs are increasingly viable for configurable video analytics when the task tolerates sampled-frame reasoning and second-level latency.

## Why This Can Replace Narrow Analytics In Some Cases

- The customer can describe a new objective without training a detector.
- One provider contract supports zone alerts, PPE checks, safety review, and operations summaries.
- Browser preprocessing reduces cloud payloads by sending a few useful frames instead of high-FPS video.
- Structured outputs make the result usable as an API response, not just a chat answer.

## Where Dedicated CV Still Wins

- Millisecond latency.
- High-FPS tracking.
- Deterministic bounding boxes for every frame.
- Regulated safety systems where a missed detection has legal or life-safety impact.
- On-prem or offline requirements.
- Large camera fleets where specialized edge models are cheaper at scale.

## Current Pricing Assumptions

- Gemini 3 Flash Preview pricing is token-based for text/image/video input and output.
- OpenAI vision inputs are billed as image/token input to the selected model.
- NVIDIA Build lists `cosmos3-nano-reasoner` as a free development endpoint; production terms and pricing should be confirmed in the NVIDIA account/API catalog before customer commitments.
- The app estimates cost from returned token usage when available. NVIDIA is shown as zero estimated cost until account-specific pricing is configured.

Use the app's token/cost panel as a directional estimate, then validate with provider billing dashboards before customer commitments.
