import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { useMemo, useState } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { usePulse } from "../PulseContext";
import StatusBanner from "../components/StatusBanner";
import type { StockQuote } from "../api";

const TABS = ["Gainers", "Losers", "Most Active", "52W High", "52W Low"];

type OverviewProps = { onSelectStock: (symbol: string) => void };

function MiniChart({ data, up }: { data: { value: number }[]; up: boolean }) {
  if (!data.length) return null;
  return (
    <ResponsiveContainer width={80} height={32}>
      <AreaChart data={data} margin={{ top: 2, bottom: 2, left: 0, right: 0 }}>
        <defs>
          <linearGradient id={`g${up ? "u" : "d"}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={up ? "#16A34A" : "#DC2626"} stopOpacity={0.2} />
            <stop offset="95%" stopColor={up ? "#16A34A" : "#DC2626"} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey="value" stroke={up ? "#16A34A" : "#DC2626"} strokeWidth={1.5} fill={`url(#g${up ? "u" : "d"})`} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default function Overview({ onSelectStock }: OverviewProps) {
  const { overview, loading, error, stocks } = usePulse();
  const [activeTab, setActiveTab] = useState("Gainers");

  const gainers = [...stocks].sort((a, b) => b.pct - a.pct).slice(0, 8);
  const losers = [...stocks].sort((a, b) => a.pct - b.pct).slice(0, 8);
  const mostActive = [...stocks].sort((a, b) => (b.volume_raw || 0) - (a.volume_raw || 0)).slice(0, 8);
  const high52 = [...stocks].filter(s => s.high52 && s.price > 0).sort((a, b) => (a.high52 - a.price) / a.high52 - (b.high52 - b.price) / b.high52).slice(0, 8);
  const low52 = [...stocks].filter(s => s.low52 && s.price > 0).sort((a, b) => (a.price - a.low52) / a.low52 - (b.price - b.low52) / b.low52).slice(0, 8);

  const tabData: Record<string, StockQuote[]> = {
    Gainers: gainers,
    Losers: losers,
    "Most Active": mostActive,
    "52W High": high52,
    "52W Low": low52,
  };

  const tabStocks = tabData[activeTab];
  const indices = overview?.indices || [];
  const nifty = overview?.nifty;
  const niftyHistory = overview?.nifty_history || [];
  const stats = overview?.model_stats;
  const sectors = overview?.sectors || [];
  const recent = overview?.recent_predictions || [];
  const chartSlice = useMemo(() => niftyHistory.slice(-90), [niftyHistory]);
  
  // Show error message if no live data is available
  const hasLiveData = stocks.length > 0 && indices.length > 0;

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-6">
      <StatusBanner loading={loading} error={error} loadingText="Loading market data and research metrics from the backend…" />
      
      {!loading && !hasLiveData && (
        <div className="card p-4 text-center mb-6" style={{ borderColor: "var(--border)" }}>
          <div className="text-sm font-medium" style={{ color: "var(--foreground)", fontFamily: "Inter" }}>
            ⚠️ Live NSE data unavailable
          </div>
          <div className="text-xs mt-1" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
            Unable to fetch real-time market data. Please check your connection or try again later.
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {(hasLiveData ? indices.slice(0, 4) : []).map((idx, i) => (
          <div key={i} className="card px-4 py-3">
            <div className="section-label mb-1">{idx.name}</div>
            <div className="flex items-end justify-between">
              <div className="font-mono-data text-base font-semibold" style={{ color: "var(--foreground)" }}>
                {idx.value}
              </div>
              <div className="text-right">
                <div className="font-mono-data text-xs" style={{ color: idx.up ? "var(--up-light)" : "var(--down-light)" }}>
                  {idx.change}
                </div>
                <div
                  className="text-xs font-medium"
                  style={{
                    color: idx.up ? "var(--up-light)" : "var(--down-light)",
                    fontFamily: "JetBrains Mono, monospace",
                  }}
                >
                  {idx.pct}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className="card p-4">
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="section-label mb-0.5">NIFTY 50</div>
                <div className="flex items-baseline gap-2">
                  <span className="font-mono-data text-2xl font-bold" style={{ fontFamily: "JetBrains Mono, monospace", color: "var(--foreground)" }}>
                    {nifty?.value || "—"}
                  </span>
                  {nifty && (
                    <span className="font-mono-data text-sm font-medium" style={{ color: nifty.up ? "var(--up-light)" : "var(--down-light)" }}>
                      {nifty.up ? "▲" : "▼"} {nifty.change} ({nifty.pct}%)
                    </span>
                  )}
                </div>
                <div className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter, sans-serif" }}>
                  NSE · live via Yahoo Finance
                </div>
              </div>
            </div>
            {hasLiveData && chartSlice.length > 0 && (
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={chartSlice} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="niftyGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#1D4ED8" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#1D4ED8" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "Inter" }} tickLine={false} axisLine={false} interval={14} tickFormatter={d => String(d).slice(5)} />
                  <YAxis tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={v => v.toFixed(0)} />
                  <Tooltip
                    contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 12, fontFamily: "JetBrains Mono" }}
                    labelStyle={{ color: "var(--muted-foreground)", fontSize: 11 }}
                    formatter={(v: unknown) => [`${Number(v as number).toFixed(2)}`, "NIFTY 50"]}
                  />
                  <Area type="monotone" dataKey="value" stroke="#1D4ED8" strokeWidth={1.5} fill="url(#niftyGrad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="card">
            <div className="flex items-center border-b px-4" style={{ borderColor: "var(--border)" }}>
              {TABS.map(t => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className="text-xs font-medium py-3 px-3 mr-1 border-b-2 transition-colors"
                  style={{
                    fontFamily: "Inter, sans-serif",
                    borderBottomColor: activeTab === t ? "var(--accent)" : "transparent",
                    color: activeTab === t ? "var(--accent)" : "var(--muted-foreground)",
                  }}
                >
                  {t}
                </button>
              ))}
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Company</th>
                  <th className="num">LTP</th>
                  <th className="num">Change</th>
                  <th className="num">% Change</th>
                  <th className="num">Volume</th>
                  <th className="num" style={{ textAlign: "center" }}>Trend</th>
                </tr>
              </thead>
              <tbody>
                {hasLiveData ? tabStocks.map(s => {
                  const mini = niftyHistory.slice(-12).map(d => ({ value: d.close * (s.price / (nifty?.value_raw || s.price || 1)) }));
                  return (
                    <tr key={s.symbol}>
                      <td>
                        <button onClick={() => onSelectStock(s.symbol)} className="text-left hover:underline">
                          <div className="text-xs font-semibold" style={{ fontFamily: "Manrope, sans-serif", color: "var(--accent)" }}>{s.symbol}</div>
                          <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter, sans-serif" }}>{s.name}</div>
                        </button>
                      </td>
                      <td className="num font-mono-data">₹{s.price.toFixed(2)}</td>
                      <td className="num font-mono-data" style={{ color: s.up ? "var(--up-light)" : "var(--down-light)" }}>
                        {s.up ? "+" : ""}{s.change.toFixed(2)}
                      </td>
                      <td className="num">
                        <span
                          className="inline-flex items-center gap-1 text-xs font-medium px-1.5 py-0.5 rounded font-mono-data"
                          style={{ background: s.up ? "var(--up-bg)" : "var(--down-bg)", color: s.up ? "var(--up)" : "var(--down)" }}
                        >
                          {s.up ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                          {Math.abs(s.pct).toFixed(2)}%
                        </span>
                      </td>
                      <td className="num font-mono-data text-xs" style={{ color: "var(--muted-foreground)" }}>{s.volume}</td>
                      <td style={{ textAlign: "right" }}>
                        <MiniChart data={mini} up={s.up} />
                      </td>
                    </tr>
                  );
                }) : (
                  <tr>
                    <td colSpan={6} className="text-center py-8 text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
                      Live data unavailable
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="pulse-dot" />
              <span className="section-label">PULSE AI · Research</span>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div className="rounded p-2.5" style={{ background: "var(--secondary)" }}>
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 22, fontWeight: 700, color: "var(--foreground)" }}>
                  {stats ? `${stats.avgAccuracy}%` : "—"}
                </div>
                <div className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>Avg. Accuracy</div>
              </div>
              <div className="rounded p-2.5" style={{ background: "var(--secondary)" }}>
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 22, fontWeight: 700, color: "var(--foreground)" }}>
                  {stats ? stats.f1Score : "—"}
                </div>
                <div className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>F1 Score</div>
              </div>
              <div className="rounded p-2.5" style={{ background: "var(--secondary)" }}>
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 16, fontWeight: 700, color: "var(--up-light)" }}>
                  {stats ? stats.buySignals.toLocaleString() : "—"}
                </div>
                <div className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>BUY Signals</div>
              </div>
              <div className="rounded p-2.5" style={{ background: "var(--secondary)" }}>
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 16, fontWeight: 700, color: "var(--down-light)" }}>
                  {stats ? stats.sellSignals.toLocaleString() : "—"}
                </div>
                <div className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>SELL Signals</div>
              </div>
            </div>
            <div className="text-xs flex items-center justify-between pt-2 border-t" style={{ borderColor: "var(--border)", color: "var(--muted-foreground)", fontFamily: "Inter" }}>
              <span>{stats ? `${stats.totalStocks} NSE stocks · SRP + DFS` : "SRP + DFS"}</span>
              <span>{stats?.lastUpdated || ""}</span>
            </div>
          </div>

          <div className="card p-4">
            <div className="section-label mb-3">Sector Performance</div>
            <div className="grid grid-cols-2 gap-1.5">
              {hasLiveData ? sectors.map(s => {
                const intensity = Math.min(Math.abs(s.change) / 2.5, 1);
                const bg = s.change >= 0
                  ? `rgba(22, 163, 74, ${0.08 + intensity * 0.22})`
                  : `rgba(220, 38, 38, ${0.08 + intensity * 0.22})`;
                return (
                  <div key={s.name} className="rounded px-2.5 py-2" style={{ background: bg }}>
                    <div className="text-xs font-semibold" style={{ fontFamily: "Manrope, sans-serif", color: "var(--foreground)" }}>{s.name}</div>
                    <div className="font-mono-data text-xs font-medium" style={{ color: s.change >= 0 ? "var(--up-light)" : "var(--down-light)" }}>
                      {s.change >= 0 ? "+" : ""}{s.change.toFixed(2)}%
                    </div>
                  </div>
                );
              }) : (
                <div className="col-span-2 text-center py-4 text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
                  Live data unavailable
                </div>
              )}
            </div>
          </div>

          <div className="card p-4">
            <div className="section-label mb-3">Recent PULSE Predictions</div>
            <div className="flex flex-col gap-2">
              {recent.length > 0 ? recent.map(s => {
                const isBuy = s.signal === "BUY";
                return (
                  <div key={s.symbol} className="flex items-center justify-between py-1.5 border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                    <div>
                      <div className="text-xs font-semibold" style={{ fontFamily: "Manrope, sans-serif", color: "var(--foreground)" }}>{s.symbol}</div>
                      <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>{s.date} · {s.regime}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs font-bold px-2 py-0.5 rounded"
                        style={{
                          fontFamily: "Manrope, sans-serif",
                          background: isBuy ? "var(--up-bg)" : "var(--down-bg)",
                          color: isBuy ? "var(--up)" : "var(--down)",
                        }}
                      >
                        {s.signal}
                      </span>
                    </div>
                  </div>
                );
              }) : (
                <div className="text-center py-4 text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
                  No recent predictions available
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
