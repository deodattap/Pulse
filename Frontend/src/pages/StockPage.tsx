import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, ArrowDownRight, ChevronDown, Search } from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, CartesianGrid, ComposedChart,
} from "recharts";
import { usePulse } from "../PulseContext";
import StatusBanner from "../components/StatusBanner";
import {
  fetchStockHistory,
  isLivePrediction,
  runPredict,
  type HistoricalPrediction,
  type LivePrediction,
  type PredictionResult,
  type PriceBar,
  type StockQuote,
} from "../api";

const STOCK_TABS = ["Overview", "Chart", "Prediction", "Technicals", "Features", "Research"];

type StockPageProps = {
  symbol: string;
  onChangeSymbol: (s: string) => void;
};

function fmtVol(v: number) {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1000) return `${(v / 1000).toFixed(1)}K`;
  return String(v);
}

export default function StockPage({ symbol, onChangeSymbol }: StockPageProps) {
  const { stocks, overview } = usePulse();
  const researchStocks = overview?.research_stocks || [];
  const isResearch = researchStocks.includes(symbol.toUpperCase());

  const [tab, setTab] = useState("Overview");
  const [range, setRange] = useState("6M");
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [customTicker, setCustomTicker] = useState("");
  const [quote, setQuote] = useState<StockQuote | null>(null);
  const [history, setHistory] = useState<PriceBar[]>([]);
  const [histLoading, setHistLoading] = useState(true);
  const [histError, setHistError] = useState<string | null>(null);

  const [selectedDate, setSelectedDate] = useState("");
  const [predMode, setPredMode] = useState<"auto" | "live" | "historical">(isResearch ? "auto" : "live");
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  useEffect(() => {
    let cancelled = false;
    setHistLoading(true);
    setHistError(null);
    setPrediction(null);
    setPredError(null);
    fetchStockHistory(symbol, range)
      .then(res => {
        if (cancelled) return;
        setQuote(res.quote);
        setHistory(res.history);
      })
      .catch(err => {
        if (!cancelled) setHistError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setHistLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [symbol, range]);

  const filtered = query.length > 1
    ? stocks.filter(s => s.symbol.includes(query.toUpperCase()) || s.name.toLowerCase().includes(query.toLowerCase()))
    : stocks;

  const stock: StockQuote = quote || stocks.find(s => s.symbol === symbol) || {
    symbol,
    name: "Live data unavailable",
    sector: "—",
    price: 0,
    change: 0,
    pct: 0,
    up: true,
    volume: "—",
    volume_raw: 0,
    open: 0,
    high: 0,
    low: 0,
    high52: 0,
    low52: 0,
    date: null,
    research: false,
  };

  const chartData = useMemo(
    () => history.map(d => ({ ...d, value: d.close })),
    [history],
  );

  const live = prediction && isLivePrediction(prediction) ? prediction : null;
  const histPred = prediction && !isLivePrediction(prediction) ? prediction as HistoricalPrediction : null;

  const signal = live?.signal || histPred?.signal || "";
  const isBuy = signal === "BUY";
  const confidence = live ? Math.round(live.confidence * 100) : histPred ? Math.round(histPred.confidence * 100) : 0;
  const selectedFeatures = live?.selected_detailed || histPred?.selected_detailed || [];
  const maxImp = Math.max(...selectedFeatures.map(f => f.importance), 0.0001);

  const technicals = live?.technicals?.slice(0, 12).map(t => ({
    name: t.name,
    value: String(t.value),
    signal: t.signal,
  })) || [];

  async function generate() {
    setPredLoading(true);
    setPredError(null);
    try {
      const result = await runPredict(symbol, selectedDate || undefined, predMode);
      setPrediction(result);
      if (!isLivePrediction(result) && result.available_dates?.length && !selectedDate) {
        setSelectedDate(result.date);
      }
    } catch (err) {
      setPredError(err instanceof Error ? err.message : String(err));
    } finally {
      setPredLoading(false);
    }
  }

  return (
    <div style={{ background: "var(--background)", minHeight: "100vh" }}>
      <div className="border-b" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        <div className="max-w-screen-xl mx-auto px-4 py-3">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="relative">
              <button
                onClick={() => setSearchOpen(v => !v)}
                className="flex items-center gap-2 px-3 py-1.5 rounded border text-sm"
                style={{ border: "1px solid var(--border)", fontFamily: "Manrope, sans-serif", fontWeight: 700, color: "var(--foreground)", fontSize: 14 }}
              >
                {stock.symbol}
                <ChevronDown size={13} style={{ color: "var(--muted-foreground)" }} />
              </button>
              {searchOpen && (
                <div className="absolute left-0 top-full mt-1 w-72 card shadow-lg z-50" style={{ background: "var(--card)" }}>
                  <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: "var(--border)" }}>
                    <Search size={13} style={{ color: "var(--muted-foreground)" }} />
                    <input
                      autoFocus
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      placeholder="Search…"
                      className="flex-1 outline-none text-sm bg-transparent"
                      style={{ fontFamily: "Inter, sans-serif" }}
                    />
                  </div>
                  <div style={{ maxHeight: 240, overflowY: "auto" }}>
                    {filtered.map(s => (
                      <button
                        key={s.symbol}
                        className="w-full flex items-center justify-between px-3 py-2 hover:bg-secondary border-b text-left"
                        style={{ borderColor: "var(--border)" }}
                        onClick={() => { onChangeSymbol(s.symbol); setSearchOpen(false); setQuery(""); }}
                      >
                        <div>
                          <div className="text-xs font-semibold" style={{ fontFamily: "Manrope", color: "var(--foreground)" }}>{s.symbol}</div>
                          <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>{s.name}</div>
                        </div>
                        <div className="font-mono-data text-xs" style={{ color: s.up ? "var(--up-light)" : "var(--down-light)" }}>
                          {s.up ? "+" : ""}{s.pct.toFixed(2)}%
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <form
              className="flex items-center gap-2"
              onSubmit={e => {
                e.preventDefault();
                if (customTicker.trim()) {
                  onChangeSymbol(customTicker.trim().toUpperCase().replace(".NS", ""));
                  setCustomTicker("");
                }
              }}
            >
              <input
                value={customTicker}
                onChange={e => setCustomTicker(e.target.value)}
                placeholder="Any NSE ticker"
                className="px-2 py-1.5 rounded border text-xs outline-none"
                style={{ border: "1px solid var(--border)", fontFamily: "JetBrains Mono", width: 140, background: "var(--card)" }}
              />
              <button type="submit" className="text-xs px-2 py-1.5 rounded" style={{ background: "var(--secondary)", fontFamily: "Inter" }}>
                Open
              </button>
            </form>
            <div>
              <span className="text-sm font-medium" style={{ fontFamily: "Manrope, sans-serif", color: "var(--foreground)" }}>{stock.name}</span>
              <span className="ml-2 text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>NSE · {stock.sector}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="border-b" style={{ background: "#0D2040", borderColor: "rgba(255,255,255,0.08)" }}>
        <div className="max-w-screen-xl mx-auto px-4 py-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 style={{ fontFamily: "Manrope, sans-serif", fontWeight: 800, fontSize: 22, color: "#fff", margin: 0 }}>
                {stock.name}
              </h1>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs font-semibold px-1.5 py-0.5 rounded" style={{ background: "rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.7)", fontFamily: "Manrope" }}>
                  {stock.symbol}
                </span>
                <span className="text-xs" style={{ color: "rgba(255,255,255,0.45)", fontFamily: "Inter" }}>NSE · Equity</span>
              </div>
              <div className="flex items-baseline gap-3 mt-3">
                <span className="font-mono-data" style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 32, fontWeight: 700, color: "#fff" }}>
                  {stock.price > 0 ? `₹${stock.price.toFixed(2)}` : "—"}
                </span>
                {stock.price > 0 && (
                  <span
                    className="flex items-center gap-1 font-mono-data text-sm font-medium px-2 py-1 rounded"
                    style={{
                      fontFamily: "JetBrains Mono, monospace",
                      background: stock.up ? "rgba(22,163,74,0.18)" : "rgba(220,38,38,0.18)",
                      color: stock.up ? "#4ADE80" : "#F87171",
                    }}
                  >
                    {stock.up ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                    {stock.up ? "+" : ""}{stock.change.toFixed(2)} ({stock.up ? "+" : ""}{stock.pct.toFixed(2)}%)
                  </span>
                )}
              </div>
              <div className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.4)", fontFamily: "Inter" }}>
                {stock.date ? `As of ${stock.date}` : "Live NSE data"}
              </div>
            </div>

            <div className="flex flex-wrap gap-x-6 gap-y-2 mt-1">
              {[
                ["Open", stock.open > 0 ? `₹${stock.open.toFixed(2)}` : "—"],
                ["High", stock.high > 0 ? `₹${stock.high.toFixed(2)}` : "—"],
                ["Low", stock.low > 0 ? `₹${stock.low.toFixed(2)}` : "—"],
                ["Volume", stock.volume],
                ["52W H", stock.high52 > 0 ? `₹${stock.high52.toFixed(2)}` : "—"],
                ["52W L", stock.low52 > 0 ? `₹${stock.low52.toFixed(2)}` : "—"],
              ].map(([l, v]) => (
                <div key={l}>
                  <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255,255,255,0.4)", fontFamily: "Inter" }}>{l}</div>
                  <div className="font-mono-data text-sm font-medium" style={{ color: "rgba(255,255,255,0.85)", fontFamily: "JetBrains Mono, monospace" }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="border-b" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        <div className="max-w-screen-xl mx-auto px-4 flex gap-0">
          {STOCK_TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="text-sm font-medium py-3 px-4 border-b-2 transition-colors"
              style={{
                fontFamily: "Inter, sans-serif",
                fontWeight: 500,
                fontSize: 13,
                borderBottomColor: tab === t ? "var(--accent)" : "transparent",
                color: tab === t ? "var(--accent)" : "var(--muted-foreground)",
              }}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-screen-xl mx-auto px-4 py-6">
        <StatusBanner loading={histLoading} error={histError} loadingText="Downloading price history from Yahoo Finance…" />
        
        {!histLoading && !quote && (
          <div className="card p-4 text-center mb-4" style={{ borderColor: "var(--border)" }}>
            <div className="text-sm font-medium" style={{ color: "var(--foreground)", fontFamily: "Inter" }}>
              ⚠️ Live data unavailable for {symbol}
            </div>
            <div className="text-xs mt-1" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
              Unable to fetch real-time data. This stock may not be available on Yahoo Finance NSE data.
            </div>
          </div>
        )}

        {tab === "Overview" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 flex flex-col gap-4">
              <div className="card p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="section-label">Price Performance</div>
                  <div className="flex gap-1">
                    {["1M", "3M", "6M", "1Y"].map(r => (
                      <button key={r} onClick={() => setRange(r)} className="text-xs px-2 py-1 rounded" style={{
                        fontFamily: "Inter", fontWeight: 500,
                        background: range === r ? "var(--accent)" : "var(--secondary)",
                        color: range === r ? "#fff" : "var(--muted-foreground)",
                      }}>{r}</button>
                    ))}
                  </div>
                </div>
                {chartData.length > 0 && (
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="stockGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={stock.up ? "#16A34A" : "#DC2626"} stopOpacity={0.15} />
                          <stop offset="95%" stopColor={stock.up ? "#16A34A" : "#DC2626"} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "Inter" }} tickLine={false} axisLine={false} interval={Math.max(1, Math.floor(chartData.length / 6))} tickFormatter={d => String(d).slice(5)} />
                      <YAxis tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={v => `₹${v.toFixed(0)}`} />
                      <Tooltip
                        contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 12 }}
                        formatter={(v: unknown) => [`₹${Number(v as number).toFixed(2)}`, stock.symbol]}
                      />
                      <Area type="monotone" dataKey="value" stroke={stock.up ? "#16A34A" : "#DC2626"} strokeWidth={1.5} fill="url(#stockGrad)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>

              <div className="card p-4">
                <div className="section-label mb-3">Volume</div>
                {chartData.length > 0 && (
                  <ResponsiveContainer width="100%" height={90}>
                    <BarChart data={chartData.slice(-60)} margin={{ top: 0, right: 4, left: -20, bottom: 0 }}>
                      <XAxis dataKey="date" tick={false} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} tickFormatter={v => fmtVol(v)} />
                      <Tooltip
                        contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }}
                        formatter={(v: unknown) => [fmtVol(Number(v as number)), "Volume"]}
                      />
                      <Bar dataKey="volume" fill="#94A3B8" opacity={0.6} radius={[1, 1, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <div className="card p-4" style={{ borderLeft: prediction ? `3px solid ${isBuy ? "var(--up-light)" : "var(--down-light)"}` : "3px solid var(--border)" }}>
                <div className="flex items-center justify-between mb-3">
                  <div className="section-label">PULSE Prediction</div>
                  <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: "var(--secondary)", color: "var(--muted-foreground)", fontFamily: "Inter", fontSize: 10 }}>20-day horizon</span>
                </div>
                {prediction ? (
                  <>
                    <div className="flex items-center gap-3 mb-3">
                      <span
                        className="text-xl font-bold px-4 py-1.5 rounded"
                        style={{
                          fontFamily: "Manrope, sans-serif",
                          background: isBuy ? "var(--up-bg)" : "var(--down-bg)",
                          color: isBuy ? "var(--up)" : "var(--down)",
                        }}
                      >
                        {signal}
                      </span>
                      <div>
                        <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>Confidence</div>
                        <div className="font-mono-data text-base font-bold" style={{ fontFamily: "JetBrains Mono", color: "var(--foreground)" }}>{confidence}%</div>
                      </div>
                    </div>
                    <div className="w-full h-1.5 rounded-full" style={{ background: "var(--border)" }}>
                      <div className="h-1.5 rounded-full" style={{ width: `${confidence}%`, background: isBuy ? "var(--up-light)" : "var(--down-light)" }} />
                    </div>
                    <div className="text-xs mt-3" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
                      {live ? `SRP + DFS · ${live.selected_features.slice(0, 3).join(", ")}` : `Historical · ${histPred?.regime}`}
                    </div>
                  </>
                ) : (
                  <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
                    Generate a prediction on the Prediction tab. This runs Dynamic Feature Selection + SRPClassifier on the backend.
                  </div>
                )}
                <button
                  onClick={() => setTab("Prediction")}
                  className="mt-3 w-full text-xs py-1.5 rounded font-medium border"
                  style={{ fontFamily: "Inter", borderColor: "var(--border)", color: "var(--accent)" }}
                >
                  View Full Prediction →
                </button>
              </div>

              <div className="card p-4">
                <div className="section-label mb-3">Key Metrics</div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <tbody>
                    {[
                      ["52W High", stock.high52 > 0 ? `₹${stock.high52.toFixed(2)}` : "—"],
                      ["52W Low", stock.low52 > 0 ? `₹${stock.low52.toFixed(2)}` : "—"],
                      ["Volume", stock.volume],
                      ["Sector", stock.sector],
                      ["Research set", isResearch ? "Yes (2022–2024 eval)" : "Live only"],
                    ].map(([l, v]) => (
                      <tr key={l} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td className="py-2 text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>{l}</td>
                        <td className="py-2 text-xs font-medium text-right font-mono-data" style={{ fontFamily: "JetBrains Mono, monospace", color: "var(--foreground)" }}>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {selectedFeatures.length > 0 && (
                <div className="card p-4">
                  <div className="section-label mb-3">Active Features (DFS)</div>
                  <div className="flex flex-col gap-1.5">
                    {selectedFeatures.slice(0, 5).map(f => (
                      <div key={f.feature} className="flex items-center gap-2">
                        <div className="text-xs font-mono-data font-medium w-28 shrink-0" style={{ fontFamily: "JetBrains Mono", color: "var(--foreground)", fontSize: 11 }}>{f.feature}</div>
                        <div className="flex-1 h-1 rounded-full" style={{ background: "var(--border)" }}>
                          <div className="h-1 rounded-full" style={{ width: `${(f.importance / maxImp) * 100}%`, background: "var(--accent)" }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {tab === "Chart" && chartData.length > 0 && (
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="section-label mb-0.5">Price Chart — {stock.symbol}</div>
                <div className="font-mono-data text-lg font-bold" style={{ color: "var(--foreground)" }}>₹{stock.price.toFixed(2)}</div>
              </div>
              <div className="flex gap-1">
                {["1M", "3M", "6M", "1Y"].map(r => (
                  <button key={r} onClick={() => setRange(r)} className="text-xs px-2.5 py-1.5 rounded" style={{
                    fontFamily: "Inter", fontWeight: 500,
                    background: range === r ? "var(--accent)" : "var(--secondary)",
                    color: range === r ? "#fff" : "var(--muted-foreground)",
                  }}>{r}</button>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={360}>
              <ComposedChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "Inter" }} tickLine={false} axisLine={false} interval={Math.max(1, Math.floor(chartData.length / 8))} tickFormatter={d => String(d).slice(5)} />
                <YAxis tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={v => `₹${v.toFixed(0)}`} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 12 }} formatter={(v: unknown) => [`₹${Number(v as number).toFixed(2)}`]} />
                <defs>
                  <linearGradient id="chartGradFull" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={stock.up ? "#16A34A" : "#DC2626"} stopOpacity={0.12} />
                    <stop offset="95%" stopColor={stock.up ? "#16A34A" : "#DC2626"} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="close" stroke={stock.up ? "#16A34A" : "#DC2626"} strokeWidth={2} fill="url(#chartGradFull)" dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}

        {tab === "Prediction" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 flex flex-col gap-4">
              <div className="card p-5">
                <div className="section-label mb-4">Generate Prediction</div>
                <div className="flex items-end gap-4 mb-3 flex-wrap">
                  <div>
                    <div className="text-xs mb-1.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter", fontWeight: 500 }}>Stock</div>
                    <div className="px-3 py-2 rounded border font-semibold text-sm" style={{ border: "1px solid var(--border)", fontFamily: "Manrope", color: "var(--foreground)" }}>
                      {stock.symbol}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs mb-1.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter", fontWeight: 500 }}>Mode</div>
                    <select
                      value={predMode}
                      onChange={e => setPredMode(e.target.value as "auto" | "live" | "historical")}
                      className="px-3 py-2 rounded border text-sm outline-none"
                      style={{ border: "1px solid var(--border)", fontFamily: "Inter", background: "var(--card)" }}
                    >
                      <option value="auto">Auto</option>
                      <option value="live">Live (Yahoo + SRP)</option>
                      <option value="historical" disabled={!isResearch}>Historical research</option>
                    </select>
                  </div>
                  <div>
                    <div className="text-xs mb-1.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter", fontWeight: 500 }}>Date (historical)</div>
                    <input
                      type="date"
                      value={selectedDate}
                      onChange={e => setSelectedDate(e.target.value)}
                      className="px-3 py-2 rounded border text-sm outline-none"
                      style={{ border: "1px solid var(--border)", fontFamily: "JetBrains Mono", color: "var(--foreground)", background: "var(--card)" }}
                    />
                  </div>
                  <div>
                    <div className="text-xs mb-1.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter", fontWeight: 500 }}>Horizon</div>
                    <div className="px-3 py-2 rounded border text-sm font-mono-data" style={{ border: "1px solid var(--border)", fontFamily: "JetBrains Mono", color: "var(--foreground)" }}>
                      20 days
                    </div>
                  </div>
                  <button
                    onClick={generate}
                    disabled={predLoading}
                    className="px-5 py-2 rounded text-sm font-semibold text-white"
                    style={{ background: predLoading ? "var(--muted-foreground)" : "var(--accent)", fontFamily: "Manrope" }}
                  >
                    {predLoading ? "Running…" : "Generate"}
                  </button>
                </div>
                <p className="text-xs mb-4" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
                  Live mode downloads 2 years of NSE data, computes 33 indicators, selects top-10 features with XGBoost, then trains SRPClassifier. This can take 20–60 seconds.
                </p>
                <StatusBanner loading={predLoading} error={predError} loadingText="Running Dynamic Feature Selection + SRPClassifier…" />

                {prediction && (
                  <div>
                    <div className="rounded p-4 mb-4" style={{ background: isBuy ? "rgba(22,163,74,0.06)" : "rgba(220,38,38,0.06)", border: `1px solid ${isBuy ? "rgba(22,163,74,0.2)" : "rgba(220,38,38,0.2)"}` }}>
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <div className="section-label mb-1">PULSE Prediction · {stock.symbol} · {prediction.mode}</div>
                          <div className="flex items-center gap-3">
                            <span
                              className="text-2xl font-extrabold px-4 py-1.5 rounded"
                              style={{ fontFamily: "Manrope", background: isBuy ? "var(--up-bg)" : "var(--down-bg)", color: isBuy ? "var(--up)" : "var(--down)" }}
                            >
                              {signal}
                            </span>
                            <div>
                              <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>Model Confidence</div>
                              <div className="font-mono-data font-bold text-lg" style={{ fontFamily: "JetBrains Mono", color: "var(--foreground)" }}>{confidence}%</div>
                            </div>
                          </div>
                        </div>
                      </div>
                      {histPred && (
                        <div className="text-xs mt-2" style={{ fontFamily: "Inter", color: "var(--foreground)" }}>
                          Date {histPred.date} · Regime {histPred.regime} · Actual {histPred.actual_label} · {histPred.correct ? "Correct" : "Incorrect"}
                        </div>
                      )}
                      {live && (
                        <div className="text-xs mt-2" style={{ fontFamily: "Inter", color: "var(--muted-foreground)" }}>
                          {live.explanation.text}
                        </div>
                      )}
                    </div>

                    {live && chartData.length > 0 && (
                      <ResponsiveContainer width="100%" height={200}>
                        <AreaChart data={(live.price_history.length ? live.price_history : chartData).slice(-60)} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} vertical={false} />
                          <XAxis dataKey="date" tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} interval={9} tickFormatter={d => String(d).slice(5)} />
                          <YAxis tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={v => `₹${v.toFixed(0)}`} />
                          <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} formatter={(v: unknown) => [`₹${Number(v as number).toFixed(2)}`]} />
                          <Area type="monotone" dataKey="close" stroke={stock.up ? "#16A34A" : "#DC2626"} strokeWidth={1.5} fill="none" dot={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                    )}

                    {histPred && (
                      <div className="mt-3">
                        <div className="section-label mb-2">Recent historical predictions</div>
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Date</th>
                              <th>Signal</th>
                              <th className="num">Conf</th>
                              <th>Actual</th>
                              <th>Result</th>
                            </tr>
                          </thead>
                          <tbody>
                            {histPred.recent.map(r => (
                              <tr key={r.date}>
                                <td className="text-xs font-mono-data">{r.date}</td>
                                <td className="text-xs">{r.signal}</td>
                                <td className="num font-mono-data text-xs">{(r.confidence * 100).toFixed(1)}%</td>
                                <td className="text-xs">{r.actual}</td>
                                <td className="text-xs">{r.correct ? "Correct" : "Incorrect"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="card p-4">
              <div className="section-label mb-3">Selected Features</div>
              {selectedFeatures.length === 0 && (
                <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>Generate a prediction to see dynamically selected features.</div>
              )}
              <div className="flex flex-col gap-2">
                {selectedFeatures.map(f => (
                  <div key={f.feature} className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: "var(--up-light)" }} />
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-xs font-semibold font-mono-data" style={{ fontFamily: "JetBrains Mono", color: "var(--foreground)", fontSize: 11 }}>{f.feature}</span>
                        <span className="text-xs" style={{ fontFamily: "Inter", color: "var(--muted-foreground)", fontSize: 10 }}>{f.category}</span>
                      </div>
                      <div className="w-full h-1 rounded-full" style={{ background: "var(--border)" }}>
                        <div className="h-1 rounded-full" style={{ width: `${(f.importance / maxImp) * 100}%`, background: "var(--accent)" }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab === "Technicals" && (
          <div className="card">
            <div className="px-4 py-3 border-b section-label" style={{ borderColor: "var(--border)" }}>Technical Indicators (from latest live prediction)</div>
            {technicals.length === 0 ? (
              <div className="p-4 text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>Run a live prediction to populate indicator values from the ML pipeline.</div>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Indicator</th>
                    <th className="num">Value</th>
                    <th>Signal</th>
                  </tr>
                </thead>
                <tbody>
                  {technicals.map(t => (
                    <tr key={t.name}>
                      <td className="text-xs font-medium" style={{ fontFamily: "Manrope", color: "var(--foreground)" }}>{t.name}</td>
                      <td className="num font-mono-data text-xs">{t.value}</td>
                      <td className="text-xs" style={{ fontFamily: "Inter" }}>{t.signal}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {tab === "Features" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="card p-4">
              <div className="section-label mb-3">Feature Importance (this prediction)</div>
              {selectedFeatures.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={selectedFeatures} layout="vertical" margin={{ top: 0, right: 20, left: 80, bottom: 0 }}>
                    <XAxis type="number" tick={{ fontSize: 10, fill: "#94A3B8" }} tickLine={false} axisLine={false} />
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: "var(--foreground)", fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} width={75} />
                    <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} />
                    <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                      {selectedFeatures.map((_, i) => (
                        <Cell key={i} fill={`rgba(29,78,216,${1 - i * 0.08})`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>Generate a prediction first.</div>
              )}
            </div>
            <div className="card">
              <div className="px-4 py-3 section-label border-b" style={{ borderColor: "var(--border)" }}>Selected Feature Details</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Feature</th>
                    <th>Category</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedFeatures.map(f => (
                    <tr key={f.feature}>
                      <td className="text-xs font-mono-data">{f.rank}</td>
                      <td className="text-xs font-semibold font-mono-data" style={{ fontFamily: "JetBrains Mono" }}>{f.feature}</td>
                      <td className="text-xs">{f.category}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {tab === "Research" && (
          <StockResearchPanel symbol={symbol} />
        )}
      </div>
    </div>
  );
}

function StockResearchPanel({ symbol }: { symbol: string }) {
  const { overview } = usePulse();
  const [row, setRow] = useState<{ srp_acc: number; pulse_acc: number; diff: number; srp_f1: number; pulse_f1: number } | null>(null);
  const [yearly, setYearly] = useState<{ year: string; srp_acc: number; pulse_acc: number }[]>([]);

  useEffect(() => {
    import("../api").then(({ fetchResearch }) => {
      fetchResearch().then(data => {
        const found = data.per_stock.find(s => s.symbol.toUpperCase() === symbol.toUpperCase()) || null;
        setRow(found);
        setYearly(data.yearly);
      }).catch(() => {});
    });
  }, [symbol]);

  if (!row) {
    return (
      <div className="card p-4 text-sm" style={{ fontFamily: "Inter", color: "var(--muted-foreground)" }}>
        {overview?.research_stocks?.includes(symbol)
          ? "Loading per-stock research metrics…"
          : `${symbol} was not in the 8-stock research evaluation. Use live prediction instead.`}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="card p-4">
        <div className="section-label mb-3">Model Comparison — {symbol}</div>
        <div className="grid grid-cols-2 gap-3 mb-4">
          {[
            { label: "Static SRP", acc: row.srp_acc, f1: row.srp_f1.toFixed(3), color: "#94A3B8" },
            { label: "PULSE (DFS+SRP)", acc: row.pulse_acc, f1: row.pulse_f1.toFixed(3), color: "var(--accent)" },
          ].map(m => (
            <div key={m.label} className="rounded p-3" style={{ background: "var(--secondary)" }}>
              <div className="text-xs font-medium mb-2" style={{ fontFamily: "Manrope", color: "var(--foreground)" }}>{m.label}</div>
              <div className="font-mono-data text-2xl font-bold" style={{ fontFamily: "JetBrains Mono", color: m.color }}>{m.acc.toFixed(1)}%</div>
              <div className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>F1: {m.f1}</div>
            </div>
          ))}
        </div>
        <div className="rounded p-3 text-sm" style={{ background: "rgba(29,78,216,0.06)", border: "1px solid rgba(29,78,216,0.15)" }}>
          <span className="font-bold" style={{ fontFamily: "Manrope", color: "var(--accent)" }}>{row.diff >= 0 ? "+" : ""}{row.diff.toFixed(1)}% accuracy vs static</span>
        </div>
      </div>
      <div className="card p-4">
        <div className="section-label mb-3">Year-wise Performance (all research stocks)</div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={yearly} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} vertical={false} />
            <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#94A3B8" }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "#94A3B8" }} tickLine={false} axisLine={false} domain={[60, 90]} tickFormatter={v => `${v}%`} />
            <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} formatter={(v: unknown) => [`${Number(v as number).toFixed(1)}%`]} />
            <Bar dataKey="srp_acc" name="Static SRP" fill="#CBD5E1" radius={[3, 3, 0, 0]} />
            <Bar dataKey="pulse_acc" name="PULSE" fill="var(--accent)" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
