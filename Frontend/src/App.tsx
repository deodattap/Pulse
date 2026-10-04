import { useCallback, useEffect, useState } from "react";
import Navbar from "./components/Navbar";
import Ticker from "./components/Ticker";
import Overview from "./pages/Overview";
import StockPage from "./pages/StockPage";
import Features from "./pages/Features";
import Research from "./pages/Research";
import { fetchOverview, type OverviewResponse, type StockQuote } from "./api";
import { PulseContext } from "./PulseContext";

type Page = "overview" | "stock" | "prediction" | "features" | "regime" | "research";

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [selectedSymbol, setSelectedSymbol] = useState("RELIANCE");
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchOverview()
      .then(data => {
        setOverview(data);
        if (data.quote_error) {
          setError(data.quote_error);
        } else if (!data.stocks || data.stocks.length === 0) {
          setError("Live data unavailable");
        } else {
          setError(null);
        }
      })
      .catch(err => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function navigate(p: Page, symbol?: string) {
    if (symbol) setSelectedSymbol(symbol);
    if (p === "prediction") {
      setPage("stock");
    } else {
      setPage(p);
    }
  }

  function selectStock(symbol: string) {
    setSelectedSymbol(symbol);
    setPage("stock");
  }

  const stocks = overview?.stocks || [];

  return (
    <PulseContext.Provider value={{ overview, loading, error, refresh, stocks }}>
      <div style={{ minHeight: "100vh", background: "var(--background)" }}>
        <Navbar page={page} onNavigate={navigate} />
        <Ticker />

        <main>
          {page === "overview" && <Overview onSelectStock={selectStock} />}
          {page === "stock" && (
            <StockPage
              symbol={selectedSymbol}
              onChangeSymbol={sym => setSelectedSymbol(sym)}
            />
          )}
          {page === "features" && <Features />}
          {page === "research" && <Research />}
        </main>

        <footer className="border-t mt-8 py-6" style={{ borderColor: "var(--border)", background: "var(--card)" }}>
          <div className="max-w-screen-xl mx-auto px-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span style={{ fontFamily: "Manrope, sans-serif", fontWeight: 800, fontSize: 14, color: "var(--primary)" }}>PULSE</span>
              <span className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
                Adaptive Streaming Stock Market Prediction · NSE India
              </span>
            </div>
            <div className="text-xs" style={{ color: "var(--muted-foreground)", fontFamily: "Inter" }}>
              For research purposes only. Not financial advice.
            </div>
          </div>
        </footer>
      </div>
    </PulseContext.Provider>
  );
}
