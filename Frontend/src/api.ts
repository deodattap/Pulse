const API_BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (body as { detail?: string }).detail || res.statusText || "Request failed";
    throw new ApiError(detail, res.status);
  }
  return body as T;
}

export function apiGet<T>(path: string) {
  return request<T>(path);
}

export function apiPost<T>(path: string, payload: unknown) {
  return request<T>(path, { method: "POST", body: JSON.stringify(payload) });
}

export type IndexQuote = {
  name: string;
  ticker?: string;
  value: string;
  value_raw?: number;
  change: string;
  pct: string;
  up: boolean;
};

export type StockQuote = {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  pct: number;
  up: boolean;
  volume: string;
  volume_raw?: number;
  open: number;
  high: number;
  low: number;
  high52: number;
  low52: number;
  date?: string | null;
  research?: boolean;
  pe?: number;
  mcap?: string;
};

export type PriceBar = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  value?: number;
};

export type ModelStats = {
  totalStocks: number;
  avgAccuracy: number;
  totalPredictions: number;
  activeDays: number;
  f1Score: number;
  buySignals: number;
  sellSignals: number;
  noTradeSignals?: number;
  lastUpdated: string | null;
};

export type RecentPrediction = {
  symbol: string;
  date: string;
  signal: string;
  confidence: number;
  regime: string;
};

export type OverviewResponse = {
  indices: IndexQuote[];
  stocks: StockQuote[];
  sectors: { name: string; change: number; stocks: number }[];
  model_stats: ModelStats;
  recent_predictions: RecentPrediction[];
  nifty_history: PriceBar[];
  nifty: IndexQuote | null;
  quote_error: string | null;
  research_stocks: string[];
};

export type FeatureRank = {
  rank: number;
  feature: string;
  frequency: number;
  count?: number;
  category: string;
  description?: string;
  importance: number;
};

export type FeaturesResponse = {
  stock: string;
  total_predictions: number;
  unique_features: number;
  avg_features_per_window: number;
  ranking: FeatureRank[];
  categories: { name: string; count: number }[];
  monthly_top: { month: string; top_feature: string }[];
  time_series: Array<Record<string, number | string>>;
  time_series_keys: string[];
  latest_selected: string[];
  latest_date: string;
  latest_top_feature: string | null;
  research_stocks: string[];
};

export type RegimeResponse = {
  quarters: { period: string; regime: string; accuracy: number; trades: number }[];
  summary: { regime: string; count: number; accuracy: number }[];
  current: string;
  latest_date: string;
};

export type ResearchResponse = {
  baseline_srp: { accuracy: number; f1: number; precision: number; recall: number };
  pulse_dynamic: { accuracy: number; f1: number; precision: number; recall: number };
  improvement: { accuracy: number; f1: number; precision: number; recall: number };
  per_stock: Array<{
    symbol: string;
    n_predictions: number;
    srp_acc: number;
    pulse_acc: number;
    diff: number;
    srp_f1: number;
    pulse_f1: number;
  }>;
  yearly: { year: string; srp_acc: number; pulse_acc: number; diff: number }[];
  financial: { metric: string; static: string; dynamic: string }[];
  journey: { algorithm: string; accuracy: number; note: string }[];
  stats: ModelStats;
};

export type SelectedFeature = {
  rank: number;
  feature: string;
  importance: number;
  category: string;
  description?: string;
};

export type LivePrediction = {
  mode: "live";
  ticker: string;
  stock_name: string;
  name?: string;
  sector?: string;
  latest_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  prev_close: number;
  change: number;
  pct: number;
  up: boolean;
  high52: number;
  low52: number;
  prediction: number;
  signal: string;
  confidence: number;
  selected_features: string[];
  selected_detailed: SelectedFeature[];
  feature_importances: Record<string, number>;
  price_history: PriceBar[];
  technicals: Array<{
    name: string;
    value: number;
    category: string;
    description: string;
    selected: boolean;
    signal: string;
  }>;
  explanation: {
    action: string;
    dominant_category: string;
    text: string;
    insights: string[];
    top_features: { feature: string; description: string }[];
  };
  n_rows_used: number;
  category_distribution: { name: string; count: number }[];
};

export type HistoricalPrediction = {
  mode: "historical";
  stock: string;
  date: string;
  available_dates: string[];
  prediction: number;
  signal: string;
  confidence: number;
  actual_target: number;
  actual_label: string;
  correct: boolean;
  regime: string;
  selected_features: string[];
  selected_detailed: SelectedFeature[];
  top_feature: string | null;
  recent: Array<{
    date: string;
    signal: string;
    confidence: number;
    actual: string;
    correct: boolean;
    regime: string;
  }>;
};

export type PredictionResult = LivePrediction | HistoricalPrediction;

export function fetchOverview() {
  return apiGet<OverviewResponse>("/api/overview");
}

export function fetchStockHistory(symbol: string, range: string) {
  return apiGet<{ quote: StockQuote; history: PriceBar[] }>(
    `/api/stocks/${encodeURIComponent(symbol)}/history?range=${encodeURIComponent(range)}`,
  );
}

export function fetchFeatures(stock?: string) {
  const q = stock ? `?stock=${encodeURIComponent(stock)}` : "";
  return apiGet<FeaturesResponse>(`/api/features${q}`);
}

export function fetchRegime() {
  return apiGet<RegimeResponse>("/api/regime");
}

export function fetchResearch() {
  return apiGet<ResearchResponse>("/api/research");
}

export function runPredict(ticker: string, date?: string, mode?: "live" | "historical" | "auto") {
  return apiPost<PredictionResult>("/api/predict", { ticker, date, mode });
}

export function isLivePrediction(p: PredictionResult): p is LivePrediction {
  return p.mode === "live";
}
