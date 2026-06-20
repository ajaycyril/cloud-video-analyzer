import type { Mode } from "./types";

export type CameraFacing = "user" | "environment";

export async function listVideoInputDevices(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === "videoinput");
}

export function buildVideoConstraints(mode: Mode, deviceId: string | null, facingMode: CameraFacing = mode === "webcam_coach" ? "user" : "environment"): MediaStreamConstraints {
  const wantsWebcam = mode === "webcam_coach";
  const wantsRearCamera = facingMode === "environment";
  const fullFieldOfViewHint = { resizeMode: { ideal: "none" } } as unknown as Partial<MediaTrackConstraints>;
  const videoSizing =
    wantsWebcam && !wantsRearCamera
      ? {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          aspectRatio: { ideal: 16 / 9 },
        }
      : {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          aspectRatio: { ideal: 16 / 9 },
        };

  if (deviceId) {
    return {
      video: {
        deviceId: { exact: deviceId },
        ...videoSizing,
        ...fullFieldOfViewHint,
      },
      audio: false,
    };
  }

  return {
    video: {
      facingMode: { ideal: facingMode },
      ...videoSizing,
      ...fullFieldOfViewHint,
    },
    audio: false,
  };
}

export function stopStream(stream: MediaStream | null): void {
  if (!stream) {
    return;
  }
  for (const track of stream.getTracks()) {
    track.stop();
  }
}
