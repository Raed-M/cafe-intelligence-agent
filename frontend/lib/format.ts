export function formatValue(value: unknown, locale: "ar" | "en" = "ar", unit?: string) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = typeof value === "number" ? value : Number(value);
  const rendered = Number.isFinite(numeric) ? new Intl.NumberFormat(locale === "ar" ? "ar-SA" : "en-US", { maximumFractionDigits: 2 }).format(numeric) : String(value);
  return unit ? `${rendered} ${unit}` : rendered;
}

export function formatDate(value?: string, locale: "ar" | "en" = "ar") {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(locale === "ar" ? "ar-SA" : "en-GB", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function displayError(error: unknown, fallback: string) { return error instanceof Error ? error.message : fallback; }
