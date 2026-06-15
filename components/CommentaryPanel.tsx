"use client";

export type CommentaryItem = {
  time: string;
  text: string;
};

export function CommentaryPanel({ items, status }: { items: CommentaryItem[]; status: string }) {
  return (
    <section className="panel commentary-panel">
      <div className="panel-heading">
        <h2>Live Commentary</h2>
        <span>{status}</span>
      </div>
      {items.length === 0 ? (
        <p className="empty">The physical AI observer will narrate stable keyframes.</p>
      ) : (
        <div className="commentary-list">
          {items.map((item, index) => (
            <div key={`${item.time}-${index}`} className="commentary-item">
              <time>{item.time}</time>
              <p>{item.text}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
