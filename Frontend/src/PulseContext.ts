import { createContext, useContext } from "react";
import type { OverviewResponse, StockQuote } from "./api";

export type PulseContextValue = {
  overview: OverviewResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
  stocks: StockQuote[];
};

export const PulseContext = createContext<PulseContextValue>({
  overview: null,
  loading: true,
  error: null,
  refresh: () => {},
  stocks: [],
});

export function usePulse() {
  return useContext(PulseContext);
}
