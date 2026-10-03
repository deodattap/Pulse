import { usePulse } from "../PulseContext";

export default function Ticker() {
  const { overview } = usePulse();
  const indices = overview?.indices || [];
  
  if (!indices.length) {
    return null; // Don't show ticker if no live data available
  }
  
  const items = [...indices, ...indices];

  return (
    <div className="ticker-container bg-white border-b" style={{ borderColor: "var(--border)" }}>
      <div className="ticker-track py-2">
        {items.map((item, i) => (
          <span key={i} className="inline-flex items-center gap-1.5 px-4" style={{ borderRight: i !== items.length - 1 ? "1px solid var(--border)" : "none" }}>
            <span className="text-xs font-semibold" style={{ color: "var(--foreground)", fontFamily: "Inter, sans-serif" }}>
              {item.name}
            </span>
            <span className="font-mono-data text-xs font-medium" style={{ color: "var(--foreground)" }}>
              {item.value}
            </span>
            <span
              className="text-xs font-medium font-mono-data"
              style={{ color: item.up ? "var(--up-light)" : "var(--down-light)" }}
            >
              {item.pct}%
            </span>
            <span
              className="inline-block w-0 h-0"
              style={{
                borderLeft: "3px solid transparent",
                borderRight: "3px solid transparent",
                ...(item.up
                  ? { borderBottom: "5px solid var(--up-light)", marginBottom: 1 }
                  : { borderTop: "5px solid var(--down-light)", marginTop: 1 }),
              }}
            />
          </span>
        ))}
      </div>
    </div>
  );
}
