import { useEffect } from "react";
import type { QueryClient } from "@tanstack/react-query";

/** Statuses where the pipeline is still producing data by itself.
 *
 *  `waiting_review` is deliberately excluded: the graph is parked at the human
 *  gate, so nothing changes until someone decides. Polling it burns one of the
 *  browser's six HTTP/1.1 connections per query against a page that already
 *  holds an SSE stream open, which is enough to stall a route change -- so the
 *  transition *out* of a paused run is driven by invalidation (the decision
 *  mutation, and `useRunStatusSync` below), never by a timer. */
const IN_FLIGHT_STATUSES = new Set([
  "queued", "running", "ingesting", "analyzing", "verifying", "resuming",
]);

export function isRunInFlight(status?: string): boolean {
  return IN_FLIGHT_STATUSES.has((status ?? "").trim().toLowerCase());
}

export const LIVE_REFETCH_MS = 8_000;

/** `refetchInterval` for any query a still-moving run can change underneath.
 *
 *  Findings, the report and source row counts are all written *during* a run
 *  but each lives under its own query key, so polling only the run itself left
 *  them serving the empty result they were first fetched with at run creation. */
export function livePolling(status?: string): number | false {
  return isRunInFlight(status) ? LIVE_REFETCH_MS : false;
}

/** Everything derived from a run, excluding the run row itself. */
export function invalidateRunDerived(queryClient: QueryClient, runId: string, cafeId?: string): void {
  for (const key of [["findings", runId], ["report", runId], ["report-location", runId]]) {
    queryClient.invalidateQueries({ queryKey: key });
  }
  queryClient.invalidateQueries({ queryKey: ["runs"] });
  if (cafeId) queryClient.invalidateQueries({ queryKey: ["sources", cafeId] });
}

/** The run row plus everything derived from it. For SSE frames and decisions. */
export function invalidateRunScoped(queryClient: QueryClient, runId: string, cafeId?: string): void {
  queryClient.invalidateQueries({ queryKey: ["run", runId] });
  invalidateRunDerived(queryClient, runId, cafeId);
}

/** Re-read a run's derived data once on every status change.
 *
 *  Polling alone leaves a race at the finish line: the last in-flight tick can
 *  land a second before the report is written, and the run then goes to
 *  `waiting_review` where polling correctly stops -- freezing the UI on an
 *  empty findings list. Keying an invalidation on the status itself guarantees
 *  exactly one fresh read after the pipeline settles. */
export function useRunStatusSync(
  queryClient: QueryClient,
  runId: string | undefined,
  status: string | undefined,
  cafeId?: string,
): void {
  useEffect(() => {
    if (!runId || !status) return;
    invalidateRunDerived(queryClient, runId, cafeId);
    // Intentionally keyed on `status`: one refetch per transition, not per render.
  }, [queryClient, runId, status, cafeId]);
}
