import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid, Cell, Legend } from "recharts";
import { fetchFeatures } from "../api";
import { useAsync } from "../useAsync";
import StatusBanner from "../components/StatusBanner";

const COLORS = ["#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD"];

export default function Features() {
  const [stock, setStock] = useState("ALL");
  const { data, loading, error } = useAsync(() => fetchFeatures(stock === "ALL" ? undefined : stock), [stock]);
  const [hoveredFeature, setHoveredFeature] = useState<string | null>(null);

  const ranking = data?.ranking || [];
  const top = ranking[0];
  const avgImp = ranking.length ? ranking.reduce((s, r) => s + r.importance, 0) / ranking.length : 0;
  const timeSeries = data?.time_series || [];
  const keys = data?.time_series_keys || [];
  const lineColors = ["#1D4ED8", "#16A34A", "#D97706", "#9333EA", "#0891B2"];

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-6">
      <div className="mb-5 flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="section-label mb-1">Dynamic Feature Selection</div>
          <h2 style={{ fontFamily: "Manrope, sans-serif", fontWeight: 800, fontSize: 22, color: "var(--foreground)", margin: 0 }}>
            Adaptive Feature Selection Engine
          </h2>
          <p className="mt-1 text-sm" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
            Feature selection frequency from the streaming evaluation logs (XGBoost importance, top-10 per window).
          </p>
        </div>
        <select
          value={stock}
          onChange={e => setStock(e.target.value)}
          className="px-3 py-2 rounded border text-sm"
          style={{ border: "1px solid var(--border)", fontFamily: "Inter", background: "var(--card)" }}
        >
          <option value="ALL">All research stocks</option>
          {(data?.research_stocks || []).map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <StatusBanner loading={loading} error={error} loadingText="Loading feature selection logs…" />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {[
          { label: "Windows", value: data ? String(data.total_predictions) : "—", sub: "Prediction windows" },
          { label: "Unique features", value: data ? String(data.unique_features) : "—", sub: "Ever selected" },
          { label: "Top feature", value: top?.feature || "—", sub: top ? `${top.frequency}% of windows` : "" },
          { label: "Avg selection rate", value: ranking.length ? `${(avgImp * 100).toFixed(1)}%` : "—", sub: "Across ranked features" },
        ].map(s => (
          <div key={s.label} className="card px-4 py-3">
            <div className="section-label mb-0.5">{s.label}</div>
            <div className="font-mono-data font-bold text-lg" style={{ fontFamily: "JetBrains Mono, monospace", color: "var(--foreground)" }}>{s.value}</div>
            <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>{s.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <div className="card p-4 lg:col-span-1">
          <div className="section-label mb-3">Selection Frequency Ranking</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={ranking.slice(0, 10)} layout="vertical" margin={{ top: 0, right: 40, left: 72, bottom: 0 }}>
              <XAxis type="number" tick={{ fontSize: 10, fill: "#94A3B8" }} tickLine={false} axisLine={false} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: "var(--foreground)", fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} width={68} />
              <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} formatter={(v: number | string | undefined) => [`${Number(v ?? 0).toFixed(1)}%`, "Frequency"]} />
              <Bar dataKey="frequency" radius={[0, 3, 3, 0]}>
                {ranking.slice(0, 10).map((_, i) => (
                  <Cell key={i} fill={COLORS[Math.min(i, COLORS.length - 1)]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-4 lg:col-span-2">
          <div className="section-label mb-3">Selection Rate Over Time (Top 5)</div>
          {timeSeries.length > 0 && keys.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={timeSeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} vertical={false} />
                <XAxis dataKey="month" tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} interval={2} />
                <YAxis tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} tickFormatter={v => `${(Number(v) * 100).toFixed(0)}%`} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "JetBrains Mono" }} />
                {keys.map((k, i) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={lineColors[i % lineColors.length]} strokeWidth={1.5} dot={false} name={k} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-xs" style={{ color: "var(--muted-foreground)" }}>No time series available.</div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card">
          <div className="px-4 py-3 border-b section-label" style={{ borderColor: "var(--border)" }}>Feature Selection Frequency</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th className="num">Freq %</th>
                <th>Category</th>
                <th>Stability</th>
              </tr>
            </thead>
            <tbody>
              {ranking.slice(0, 12).map(f => {
                const stability = f.frequency > 60 ? "High" : f.frequency > 35 ? "Medium" : "Low";
                const stabColor = stability === "High" ? "var(--up-light)" : stability === "Medium" ? "#D97706" : "var(--down-light)";
                return (
                  <tr key={f.feature}>
                    <td className="text-xs font-semibold font-mono-data" style={{ fontFamily: "JetBrains Mono", color: "var(--foreground)" }}>{f.feature}</td>
                    <td className="num">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-1 rounded-full" style={{ background: "var(--border)" }}>
                          <div className="h-1 rounded-full" style={{ width: `${Math.min(f.frequency, 100)}%`, background: "var(--accent)" }} />
                        </div>
                        <span className="font-mono-data text-xs">{f.frequency}%</span>
                      </div>
                    </td>
                    <td className="text-xs">{f.category}</td>
                    <td>
                      <span className="text-xs font-medium" style={{ color: stabColor, fontFamily: "Inter" }}>{stability}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="card p-4">
          <div className="section-label mb-3">Currently Selected Features (latest window)</div>
          <div className="grid grid-cols-2 gap-2">
            {(data?.latest_selected || []).map((feat, i) => {
              const meta = ranking.find(r => r.feature === feat);
              return (
                <div
                  key={feat}
                  className="rounded p-2.5"
                  style={{ background: hoveredFeature === feat ? "rgba(29,78,216,0.08)" : "var(--secondary)" }}
                  onMouseEnter={() => setHoveredFeature(feat)}
                  onMouseLeave={() => setHoveredFeature(null)}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-semibold font-mono-data" style={{ fontFamily: "JetBrains Mono", color: "var(--foreground)", fontSize: 11 }}>{feat}</span>
                    <span className="text-xs" style={{ fontFamily: "Inter", fontSize: 10, color: "var(--muted-foreground)" }}>{meta?.category || ""}</span>
                  </div>
                  <div className="text-xs" style={{ fontFamily: "Inter", color: "var(--muted-foreground)" }}>#{i + 1}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
