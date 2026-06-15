import type { Mode } from "./types";

export async function listVideoInputDevices(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === "videoinput");
}

export function buildVideoConstraints(mode: Mode, deviceId: string | null): MediaStreamConstraints {
  const wantsWebcam = mode === "webcam_coach";
  const videoSizing = wantsWebcam
    ? {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        aspectRatio: { ideal: 16 / 9 },
      }
    : {
        width: { ideal: 1080 },
        height: { ideal: 1920 },
        aspectRatio: { ideal: 9 / 16 },
      };

  if (deviceId) {
    return {
      video: {
        deviceId: { exact: deviceId },
        ...videoSizing,
      },
      audio: false,
    };
  }

  const facingMode = wantsWebcam ? "user" : "environment";
  return {
    video: {
      facingMode: { ideal: facingMode },
      ...videoSizing,
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
