export default function StatusBanner({
  loading,
  error,
  loadingText = "Loading live data…",
}: {
  loading?: boolean;
  error?: string | null;
  loadingText?: string;
}) {
  if (loading) {
    return (
      <div className="mb-4 rounded px-3 py-2 text-xs" style={{ background: "var(--secondary)", color: "var(--muted-foreground)", fontFamily: "Inter" }}>
        {loadingText}
      </div>
    );
  }
  if (error) {
    return (
      <div className="mb-4 rounded px-3 py-2 text-xs" style={{ background: "var(--down-bg)", color: "var(--down)", fontFamily: "Inter" }}>
        ⚠️ {error || "Live data unavailable"}
      </div>
    );
  }
  return null;
}
