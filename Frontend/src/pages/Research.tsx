import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, LineChart, Line, Legend } from "recharts";
import { fetchResearch } from "../api";
import { useAsync } from "../useAsync";
import StatusBanner from "../components/StatusBanner";

export default function Research() {
  const { data, loading, error } = useAsync(() => fetchResearch(), []);
  const baseline = data?.baseline_srp;
  const pulse = data?.pulse_dynamic;
  const improvement = data?.improvement;

  const metricsComparison = baseline && pulse ? [
    { metric: "Accuracy", srp: baseline.accuracy, pulse: pulse.accuracy },
    { metric: "F1 Score", srp: baseline.f1 * 100, pulse: pulse.f1 * 100 },
    { metric: "Precision", srp: baseline.precision * 100, pulse: pulse.precision * 100 },
    { metric: "Recall", srp: baseline.recall * 100, pulse: pulse.recall * 100 },
  ] : [];

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-6">
      <div className="mb-5">
        <div className="section-label mb-1">Research Results</div>
        <h2 style={{ fontFamily: "Manrope, sans-serif", fontWeight: 800, fontSize: 22, color: "var(--foreground)", margin: 0 }}>
          Static SRP vs Dynamic Feature Selection + SRP
        </h2>
        <p className="mt-1 text-sm" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
          Metrics loaded from the streaming evaluation CSVs (8 NSE stocks, walk-forward 2022–2024).
        </p>
      </div>

      <StatusBanner loading={loading} error={error} loadingText="Loading research metrics…" />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {[
          { label: "Accuracy Gain", value: improvement ? `${improvement.accuracy >= 0 ? "+" : ""}${improvement.accuracy}%` : "—", color: "var(--up-light)" },
          { label: "F1 Improvement", value: improvement ? `${improvement.f1 >= 0 ? "+" : ""}${improvement.f1.toFixed(3)}` : "—", color: "var(--accent)" },
          { label: "PULSE Accuracy", value: pulse ? `${pulse.accuracy}%` : "—", color: "var(--foreground)" },
          { label: "Stocks Evaluated", value: data ? String(data.per_stock.length) : "—", color: "var(--foreground)" },
        ].map(s => (
          <div key={s.label} className="card px-4 py-3">
            <div className="section-label mb-0.5">{s.label}</div>
            <div className="font-mono-data font-bold text-xl" style={{ fontFamily: "JetBrains Mono, monospace", color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <div className="card p-4">
          <div className="section-label mb-3">Metrics Comparison</div>
          {metricsComparison.length > 0 && (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={metricsComparison} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} vertical={false} />
                <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "var(--foreground)", fontFamily: "Manrope" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} domain={[60, 90]} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} formatter={(v: unknown) => [`${Number(v as number).toFixed(1)}%`]} />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "Inter" }} />
                <Bar dataKey="srp" name="Static SRP" fill="#CBD5E1" radius={[3, 3, 0, 0]} />
                <Bar dataKey="pulse" name="PULSE (DFS+SRP)" fill="var(--accent)" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="card p-4">
          <div className="section-label mb-3">Year-wise Accuracy</div>
          {data?.yearly?.length ? (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={data.yearly} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} vertical={false} />
                <XAxis dataKey="year" tick={{ fontSize: 11, fill: "#94A3B8" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: "#94A3B8" }} tickLine={false} axisLine={false} domain={[60, 90]} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} formatter={(v: unknown) => [`${Number(v as number).toFixed(1)}%`]} />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "Inter" }} />
                <Line type="monotone" dataKey="srp_acc" name="Static SRP" stroke="#CBD5E1" strokeWidth={2} dot={{ r: 3, fill: "#CBD5E1" }} />
                <Line type="monotone" dataKey="pulse_acc" name="PULSE" stroke="var(--accent)" strokeWidth={2} dot={{ r: 3, fill: "var(--accent)" }} />
              </LineChart>
            </ResponsiveContainer>
          ) : null}
        </div>
      </div>

      {data?.journey?.length ? (
        <div className="card p-4 mb-4">
          <div className="section-label mb-3">Algorithm journey</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data.journey} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
              <XAxis dataKey="algorithm" tick={{ fontSize: 9, fill: "#94A3B8" }} interval={0} />
              <YAxis domain={[40, 90]} tick={{ fontSize: 9, fill: "#94A3B8" }} />
              <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }} />
              <Bar dataKey="accuracy" fill="var(--accent)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : null}

      <div className="card mb-4">
        <div className="px-4 py-3 border-b section-label" style={{ borderColor: "var(--border)" }}>Per-Stock Comparison</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Stock</th>
              <th className="num">Static Acc</th>
              <th className="num">PULSE Acc</th>
              <th className="num">Δ Accuracy</th>
              <th className="num">Static F1</th>
              <th className="num">PULSE F1</th>
            </tr>
          </thead>
          <tbody>
            {(data?.per_stock || []).map(s => (
              <tr key={s.symbol}>
                <td className="text-xs font-semibold" style={{ fontFamily: "Manrope", color: "var(--foreground)" }}>{s.symbol}</td>
                <td className="num font-mono-data text-xs">{s.srp_acc.toFixed(1)}%</td>
                <td className="num font-mono-data text-xs font-semibold" style={{ color: "var(--accent)" }}>{s.pulse_acc.toFixed(1)}%</td>
                <td className="num">
                  <span
                    className="font-mono-data text-xs font-semibold px-1.5 py-0.5 rounded"
                    style={{ background: s.diff >= 0 ? "var(--up-bg)" : "var(--down-bg)", color: s.diff >= 0 ? "var(--up)" : "var(--down)", fontFamily: "JetBrains Mono" }}
                  >
                    {s.diff >= 0 ? "+" : ""}{s.diff.toFixed(1)}%
                  </span>
                </td>
                <td className="num font-mono-data text-xs" style={{ color: "var(--muted-foreground)" }}>{s.srp_f1.toFixed(3)}</td>
                <td className="num font-mono-data text-xs font-medium">{s.pulse_f1.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data?.financial?.length ? (
        <div className="card p-5">
          <div className="section-label mb-1">Simulated Portfolio Performance</div>
          <p className="text-xs mb-3" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
            Fixed ±0.5% per-trade simulation · Not actual trading returns
          </p>
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Static</th>
                <th>Dynamic</th>
              </tr>
            </thead>
            <tbody>
              {data.financial.map(f => (
                <tr key={f.metric}>
                  <td className="text-xs">{f.metric}</td>
                  <td className="text-xs font-mono-data">{f.static}</td>
                  <td className="text-xs font-mono-data">{f.dynamic}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
