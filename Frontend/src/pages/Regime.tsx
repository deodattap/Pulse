import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from "recharts";
import { fetchRegime } from "../api";
import { useAsync } from "../useAsync";
import StatusBanner from "../components/StatusBanner";

const regimeColors: Record<string, string> = {
  Bull: "#16A34A",
  Bear: "#DC2626",
  Sideways: "#D97706",
  Recovery: "#2563EB",
  Volatile: "#9333EA",
};

const regimeBg: Record<string, string> = {
  Bull: "#DCFCE7",
  Bear: "#FEE2E2",
  Sideways: "#FEF3C7",
  Recovery: "#DBEAFE",
  Volatile: "#F3E8FF",
};

export default function Regime() {
  const { data, loading, error } = useAsync(() => fetchRegime(), []);
  const quarters = data?.quarters || [];
  const summary = data?.summary || [];

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-6">
      <div className="mb-5">
        <div className="section-label mb-1">Market Regime Analysis</div>
        <h2 style={{ fontFamily: "Manrope, sans-serif", fontWeight: 800, fontSize: 22, color: "var(--foreground)", margin: 0 }}>
          Market Condition Detection
        </h2>
        <p className="mt-1 text-sm" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
          Regimes and accuracy are computed from the streaming evaluation log (8 NSE stocks, 2022–2024).
        </p>
      </div>

      <StatusBanner loading={loading} error={error} loadingText="Loading regime analysis…" />

      <div className="flex flex-wrap gap-3 mb-6">
        {summary.map(s => (
          <div key={s.regime} className="card px-4 py-3 flex items-center gap-3">
            <div className="w-2 h-8 rounded-full" style={{ background: regimeColors[s.regime] || "#94A3B8" }} />
            <div>
              <div className="font-semibold text-sm" style={{ fontFamily: "Manrope", color: "var(--foreground)" }}>{s.regime}</div>
              <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>{s.count} predictions · {s.accuracy}% acc</div>
            </div>
          </div>
        ))}
        <div className="card px-4 py-3">
          <div className="section-label mb-0.5">Latest logged regime</div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full" style={{ background: regimeColors[data?.current || ""] || "#16A34A" }} />
            <span className="font-semibold text-sm" style={{ fontFamily: "Manrope", color: "var(--foreground)" }}>{data?.current || "—"}</span>
          </div>
        </div>
      </div>

      <div className="card p-4 mb-4">
        <div className="section-label mb-3">Prediction Accuracy by Quarter</div>
        {quarters.length > 0 && (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={quarters} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} vertical={false} />
              <XAxis dataKey="period" tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} interval={0} />
              <YAxis tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} domain={[60, 90]} tickFormatter={v => `${v}%`} />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }}
                formatter={(v: unknown, _: unknown, props: { payload?: { regime?: string } }) => [`${Number(v as number).toFixed(1)}%`, props?.payload?.regime || "Accuracy"]}
              />
              <Bar dataKey="accuracy" radius={[3, 3, 0, 0]}>
                {quarters.map((d, i) => (
                  <Cell key={i} fill={regimeColors[d.regime] || "#94A3B8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="card">
        <div className="px-4 py-3 border-b section-label" style={{ borderColor: "var(--border)" }}>Regime Period Analysis</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Period</th>
              <th>Regime</th>
              <th className="num">Accuracy</th>
              <th className="num">Predictions</th>
              <th>Performance</th>
            </tr>
          </thead>
          <tbody>
            {quarters.map((r, i) => {
              const barW = Math.min(100, Math.max(0, ((r.accuracy - 65) / 20) * 100));
              return (
                <tr key={i}>
                  <td className="text-xs font-mono-data" style={{ fontFamily: "JetBrains Mono", color: "var(--foreground)" }}>{r.period}</td>
                  <td>
                    <span
                      className="text-xs font-medium px-2 py-0.5 rounded"
                      style={{ background: regimeBg[r.regime] || "var(--secondary)", color: regimeColors[r.regime] || "var(--foreground)", fontFamily: "Manrope" }}
                    >
                      {r.regime}
                    </span>
                  </td>
                  <td className="num font-mono-data font-semibold text-xs">{r.accuracy.toFixed(1)}%</td>
                  <td className="num font-mono-data text-xs" style={{ color: "var(--muted-foreground)" }}>{r.trades.toLocaleString()}</td>
                  <td>
                    <div className="w-24 h-1.5 rounded-full" style={{ background: "var(--border)" }}>
                      <div className="h-1.5 rounded-full" style={{ width: `${barW}%`, background: regimeColors[r.regime] || "#94A3B8" }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
