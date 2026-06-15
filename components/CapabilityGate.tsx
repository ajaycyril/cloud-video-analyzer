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
          <p>Clutter Scorer is validating camera, WebAssembly, and canvas requirements.</p>
        </div>
      </main>
    );
  }

  if (!result.supported) {
    return (
      <main className="center-screen">
        <div className="status-card error">
          <ShieldAlert />
          <h1>Browser AI requirements missing</h1>
          <p>Use HTTPS with camera, WebAssembly, and canvas support. Chrome, Edge, and iPhone Safari are supported when those APIs are available.</p>
          <div className="requirements-grid">
            <Requirement label="Browser" ok value={result.browserName} />
            <Requirement label="Secure context" ok={result.isSecureContext} value={result.isSecureContext ? "Ready" : "Missing"} />
            <Requirement label="Camera API" ok={result.hasMediaDevices} value={result.hasMediaDevices ? "Ready" : "Missing"} />
            <Requirement label="WebAssembly" ok={result.hasWebAssembly} value={result.hasWebAssembly ? "Ready" : "Missing"} />
            <Requirement label="Canvas ImageData" ok={result.hasCanvasImageData} value={result.hasCanvasImageData ? "Ready" : "Missing"} />
            <Requirement label="WebGPU" ok={result.hasWebGPU} optional value={result.hasWebGPU ? "Ready" : "Optional"} />
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
        <span>{result.hasWebGPU ? "WebGPU ready" : "CPU/WASM mode"}</span>
      </div>
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
