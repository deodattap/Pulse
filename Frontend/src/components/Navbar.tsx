import { Search, Bell, Activity } from "lucide-react";
import { useState } from "react";
import { usePulse } from "../PulseContext";

type Page = "overview" | "stock" | "prediction" | "features" | "regime" | "research";

interface NavbarProps {
  page: Page;
  onNavigate: (page: Page, symbol?: string) => void;
}

const NAV_ITEMS: { label: string; page: Page }[] = [
  { label: "Overview", page: "overview" },
  { label: "Stock Analysis", page: "stock" },
  { label: "Prediction", page: "prediction" },
  { label: "Features", page: "features" },
  { label: "Market Regime", page: "regime" },
  { label: "Research", page: "research" },
];

export default function Navbar({ page, onNavigate }: NavbarProps) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const { stocks } = usePulse();

  const filtered = query.length > 1
    ? stocks.filter(s => s.symbol.includes(query.toUpperCase()) || s.name.toLowerCase().includes(query.toLowerCase())).slice(0, 6)
    : [];

  return (
    <header style={{ background: "var(--primary)", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
      <div className="max-w-screen-xl mx-auto px-4 flex items-center gap-6 h-14">
        {/* Logo */}
        <button
          onClick={() => onNavigate("overview")}
          className="flex items-center gap-2 flex-shrink-0"
        >
          <div className="flex items-center justify-center w-7 h-7 rounded" style={{ background: "var(--accent)" }}>
            <Activity size={14} className="text-white" />
          </div>
          <span style={{ fontFamily: "Manrope, sans-serif", fontWeight: 800, fontSize: 17, color: "#fff", letterSpacing: "-0.01em" }}>
            PULSE
          </span>
          <span style={{ fontSize: 10, fontWeight: 500, color: "rgba(255,255,255,0.45)", letterSpacing: "0.06em", marginTop: 1 }}>
            NSE · AI
          </span>
        </button>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-1 flex-1">
          {NAV_ITEMS.map(item => (
            <button
              key={item.page}
              onClick={() => onNavigate(item.page)}
              className="nav-link px-3 py-1.5 rounded text-sm"
              style={{
                fontFamily: "Inter, sans-serif",
                fontWeight: 500,
                fontSize: 13,
                color: page === item.page ? "#fff" : "rgba(255,255,255,0.6)",
                background: page === item.page ? "rgba(255,255,255,0.1)" : "transparent",
              }}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {/* Search */}
        <div className="relative ml-auto">
          <button
            onClick={() => setSearchOpen(v => !v)}
            className="flex items-center gap-2 px-3 py-1.5 rounded text-sm"
            style={{ background: "rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.6)", fontSize: 13 }}
          >
            <Search size={13} />
            <span style={{ fontFamily: "Inter, sans-serif" }}>Search stocks…</span>
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.3)" }}>⌘K</span>
          </button>

          {searchOpen && (
            <div className="absolute right-0 top-full mt-1 w-80 card shadow-lg z-50" style={{ background: "var(--card)" }}>
              <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: "var(--border)" }}>
                <Search size={14} style={{ color: "var(--muted-foreground)" }} />
                <input
                  autoFocus
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="Search NSE stocks…"
                  className="flex-1 outline-none text-sm bg-transparent"
                  style={{ fontFamily: "Inter, sans-serif", color: "var(--foreground)" }}
                />
              </div>
              {filtered.length > 0 && (
                <div>
                  {filtered.map(s => (
                    <button
                      key={s.symbol}
                      className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-secondary text-left"
                      style={{ borderBottom: "1px solid var(--border)" }}
                      onClick={() => { onNavigate("stock", s.symbol); setSearchOpen(false); setQuery(""); }}
                    >
                      <div>
                        <div className="text-xs font-semibold" style={{ fontFamily: "Manrope, sans-serif", color: "var(--foreground)" }}>{s.symbol}</div>
                        <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter, sans-serif" }}>{s.name}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono-data text-xs font-medium" style={{ color: "var(--foreground)" }}>₹{s.price.toFixed(2)}</div>
                        <div className="font-mono-data text-xs" style={{ color: s.up ? "var(--up-light)" : "var(--down-light)" }}>
                          {s.up ? "+" : ""}{s.pct.toFixed(2)}%
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {query.length > 1 && filtered.length === 0 && (
                <div className="px-3 py-4 text-sm text-center" style={{ color: "var(--muted-foreground)", fontFamily: "Inter, sans-serif" }}>
                  No stocks found
                </div>
              )}
              {query.length <= 1 && (
                <div className="px-3 py-2">
                  <div className="section-label mb-2">Top Stocks</div>
                  {stocks.slice(0, 4).map(s => (
                    <button
                      key={s.symbol}
                      className="w-full flex items-center justify-between py-2 hover:bg-secondary rounded text-left px-1"
                      onClick={() => { onNavigate("stock", s.symbol); setSearchOpen(false); setQuery(""); }}
                    >
                      <span className="text-xs font-semibold" style={{ fontFamily: "Manrope, sans-serif", color: "var(--foreground)" }}>{s.symbol}</span>
                      <span className="font-mono-data text-xs" style={{ color: s.up ? "var(--up-light)" : "var(--down-light)" }}>
                        {s.up ? "+" : ""}{s.pct.toFixed(2)}%
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <button className="p-1.5 rounded nav-link" style={{ color: "rgba(255,255,255,0.55)" }}>
          <Bell size={16} />
        </button>
      </div>
    </header>
  );
}
