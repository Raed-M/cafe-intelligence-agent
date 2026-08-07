import { describe, expect, it } from "vitest";
import { apiInternals } from "../lib/api";
import { formatDate, formatValue } from "../lib/format";
import { workflowIndexForStage } from "../components/workflow-monitor";
import { flattenRows, humanLabel, humanValue, visibleColumns } from "../lib/data-display";
import { enhanceReportHtml } from "../lib/report-format";

describe("API response adapters", () => {
  it("accepts the contract item envelope", () => {
    expect(apiInternals.items<{ id: string }>({ schema_version: "1", items: [{ id: "cafe-1" }] })).toEqual([{ id: "cafe-1" }]);
  });

  it("does not invent items for malformed responses", () => {
    expect(apiInternals.items({ result: "ok" })).toEqual([]);
    expect(apiInternals.items(null)).toEqual([]);
  });

  it("unwraps a single run from the API contract envelope", () => {
    expect(apiInternals.member<{ id: string }>({ schema_version: "1", run: { id: "run-1" } }, "run")).toEqual({ id: "run-1" });
    expect(() => apiInternals.member({}, "run")).toThrow("missing 'run'");
  });

  it("builds encoded optional query strings", () => {
    expect(apiInternals.query({ cafe_id: "قهوة سيهات", limit: 10, cursor: undefined })).toBe("?cafe_id=%D9%82%D9%87%D9%88%D8%A9+%D8%B3%D9%8A%D9%87%D8%A7%D8%AA&limit=10");
  });
});

describe("honest display formatting", () => {
  it("renders missing values as an em dash rather than zero", () => {
    expect(formatValue(undefined)).toBe("—");
    expect(formatValue(null)).toBe("—");
  });

  it("keeps invalid backend dates visible for diagnosis", () => {
    expect(formatDate("not-a-date", "en")).toBe("not-a-date");
  });
});

describe("live backend workflow mapping", () => {
  it("shows the new cross-domain synthesis as its own stage", () => {
    expect(workflowIndexForStage("cross_domain_synthesis", "running")).toBe(3);
  });
});

describe("human-readable data adapter", () => {
  it("flattens nested API rows and hides internal identifiers from visible columns", () => {
    const rows = flattenRows([{ record_id: "opaque-123", data: { item_en: "Spanish Latte", price_sar: 18 } }]);
    expect(rows[0].__recordId).toBe("opaque-123");
    expect(visibleColumns(rows, "menu")).toEqual(["item_en", "price_sar"]);
    expect(humanLabel("price_sar")).toBe("Price (SAR)");
    expect(humanValue({ date: "2026-08-07", visitors: 44 }, "en")).toContain("Date: 2026-08-07");
  });
});

describe("readable report formatting", () => {
  it("adds the reading path and folds raw local context into a technical appendix", () => {
    const formatted = enhanceReportHtml("<html><head></head><body><h1>Weekly report</h1><h2>Local, Calendar &amp; Prayer Context</h2><p>raw weather dump</p><h2>Limitations</h2></body></html>");
    expect(formatted).toContain("Verified findings");
    expect(formatted).toContain('class="technical-appendix"');
    expect(formatted).toContain("raw weather dump");
  });
});
