"use client";

import type { WorldState } from "../lib/types";

export function WorldStatePanel({ worldState, events }: { worldState: WorldState | null; events: string[] }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>World State</h2>
        <span>{worldState?.objects.length ?? 0} objects</span>
      </div>
      {!worldState ? (
        <p className="empty">Scene graph will appear after analysis.</p>
      ) : (
        <>
          <p className="summary">{worldState.summary}</p>
          <div className="object-grid">
            {worldState.objects.slice(0, 8).map((object) => (
              <div key={object.id} className="object-card">
                <strong>{object.label}</strong>
                <span>{object.location}</span>
                <small>{object.affordance}</small>
              </div>
            ))}
          </div>
          <div className="relationship-list">
            {worldState.relationships.slice(0, 5).map((relationship, index) => (
              <p key={`${relationship.subject}-${index}`}>
                <strong>{relationship.subject}</strong> {relationship.relation} <strong>{relationship.object}</strong>: {relationship.implication}
              </p>
            ))}
          </div>
        </>
      )}
      {events.length > 0 ? (
        <div className="event-log">
          {events.slice(0, 6).map((event, index) => (
            <span key={`${event}-${index}`}>{event}</span>
          ))}
        </div>
      ) : null}
    </section>
  );
}
