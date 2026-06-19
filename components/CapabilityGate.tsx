"use client";

import { CheckCircle2, Cpu, ShieldAlert, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import type { CapabilityResult } from "@/lib/types";

export function CapabilityGate({ result, children }: { result: CapabilityResult | null; children: ReactNode }) {
  if (!result) {
    return (
      <main className="center-screen">
        <div className="status-card">
          <Cpu />
          <h1>Checking browser capabilities</h1>
          <p>Cloud Video Analyzer is validating video, canvas frame extraction, and optional camera/object-hint support.</p>
        </div>
      </main>
    );
  }

  if (!result.supported) {
    return (
      <main className="center-screen">
        <div className="status-card error">
          <ShieldAlert />
          <h1>Video frame extraction is unavailable</h1>
          <p>This app needs canvas frame extraction to analyze camera, uploaded clips, or demo videos. Camera and local object hints are optional.</p>
          <div className="requirements-grid">
            <Requirement label="Browser" ok value={result.browserName} />
            <Requirement label="Canvas ImageData" ok={result.hasCanvasImageData} value={result.hasCanvasImageData ? "Ready" : "Missing"} />
            <Requirement label="Camera API" ok={result.hasMediaDevices} optional value={result.hasMediaDevices ? "Ready" : "Upload-only"} />
            <Requirement label="WebAssembly" ok={result.hasWebAssembly} optional value={result.hasWebAssembly ? "Object hints ready" : "Cloud-only"} />
            <Requirement label="WebGPU" ok={result.hasWebGPU} optional value={result.hasWebGPU ? "Available" : "Unavailable"} />
            <Requirement label="Secure context" ok={result.isSecureContext} optional value={result.isSecureContext ? "Ready" : "Upload-only"} />
          </div>
          <ul className="blocking-list">
            {result.issues.map((issue) => (
              <li key={issue}>{issue}</li>
            ))}
          </ul>
        </div>
      </main>
    );
  }

  return (
    <>
      <div className="capability-pill">
        <Cpu size={14} />
        <span>{result.browserName}</span>
        <span>{result.hasMediaDevices ? "camera ready" : "upload ready"}</span>
        <span>{result.hasWebAssembly ? "local object hints" : "cloud-only objects"}</span>
        <span>{result.hasWebGPU ? "WebGPU edge ready" : "WebGPU unavailable"}</span>
      </div>
      {result.warnings.length ? (
        <div className="compatibility-warning" role="status">
          {result.warnings[0]}
        </div>
      ) : null}
      {children}
    </>
  );
}

function Requirement({
  label,
  value,
  ok,
  optional,
}: {
  label: string;
  value: string;
  ok: boolean;
  optional?: boolean;
}) {
  return (
    <div className="requirement">
      {ok ? <CheckCircle2 className="ok" /> : <XCircle className={optional ? "warn" : "bad"} />}
      <div>
        <strong>{label}</strong>
        <span>{value}</span>
      </div>
    </div>
  );
}
