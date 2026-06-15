"use client";

import type { AppMode } from "@/lib/types";

const modeOptions: Array<{ value: AppMode; label: string; description: string }> = [
  { value: "space_scan", label: "Any Space", description: "Shelves, counters, rooms, closets, drawers, desks." },
  { value: "desk_productivity", label: "Work Setup", description: "Desk, cables, tools, focus zone, spill risk." },
  { value: "webcam_coach", label: "Camera Background", description: "Lighting, framing, background, call readiness." },
];

export function ModeSelector({
  value,
  onChange,
  disabled,
}: {
  value: AppMode;
  onChange: (mode: AppMode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="mode-selector">
      {modeOptions.map((option) => (
        <button
          key={option.value}
          type="button"
          className={option.value === value ? "mode-card active" : "mode-card"}
          disabled={disabled}
          onClick={() => onChange(option.value)}
        >
          <strong>{option.label}</strong>
          <span>{option.description}</span>
        </button>
      ))}
    </div>
  );
}
