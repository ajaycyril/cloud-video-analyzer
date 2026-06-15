# Sample Footage

The app supports three demo sources:

1. Live camera.
2. Uploaded local video.
3. Public MP4/WebM URL with CORS enabled.

## Recommended Demo Path

Use uploaded local footage for industrial demos. It avoids browser canvas security restrictions and is the most reliable way to test end-to-end.

The app includes:

- A CORS-safe browser sample clip for plumbing checks.
- A public-domain factory sample link for person/zone style demos when browser CORS permits it.

## Internet Video Caveat

Browser frame extraction uses a canvas. If an internet video does not send CORS headers that allow anonymous browser access, the canvas becomes tainted and frame sampling fails. That is expected browser security behavior, not an app bug.

## Suggested Test Clips

- Record a short phone clip of a person walking into a marked area.
- Record a PPE-style clip where a helmet or vest is visible, then one where it is unclear.
- Record a short operations clip with a queue, blocked path, tool movement, or repeated process.

Keep clips between 5 and 20 seconds for fast demos.
