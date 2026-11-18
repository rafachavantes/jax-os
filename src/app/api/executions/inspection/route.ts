import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getDb } from "@/server/db";
import { inspectExecutionRows, type InspectionSelector } from "@/server/collectors/execution-inspection";
import { RUN_ID_RE, PROJECT_LABEL } from "@/lib/workflow";

export const dynamic = "force-dynamic";

const MAX_SELECTORS = 20;
const MAX_QUERY_CHARS = 8 * 1024;

// `r=<destination>|<runId>|<startEventId-or-empty>`, each field URL-encoded. Arbitrary paths or
// commands never enter this route: the selectors are re-resolved through the own DB.
export function parseInspectionSearch(search: string): { ok: true; selectors: InspectionSelector[] } | { ok: false; error: string } {
  const params = new URLSearchParams(search);
  for (const key of params.keys()) {
    if (key !== "r") return { ok: false, error: "invalid selector" };
  }
  const raw = params.getAll("r");
  if (raw.length > MAX_SELECTORS) return { ok: false, error: "too many selectors" };
  const selectors: InspectionSelector[] = [];
  for (const value of raw) {
    const parts = value.split("|");
    if (parts.length !== 3) return { ok: false, error: "invalid selector" };
    let destination: string;
    let runId: string;
    try {
      destination = decodeURIComponent(parts[0]);
      runId = decodeURIComponent(parts[1]);
    } catch {
      return { ok: false, error: "invalid selector" };
    }
    if (!PROJECT_LABEL.test(destination) || !RUN_ID_RE.test(runId)) return { ok: false, error: "invalid selector" };
    const rawStart = parts[2];
    if (rawStart !== "" && !/^[1-9][0-9]{0,15}$/.test(rawStart)) return { ok: false, error: "invalid selector" };
    selectors.push({ destination, runId, startEventId: rawStart === "" ? null : Number(rawStart) });
  }
  return { ok: true, selectors };
}

export async function GET(req: Request) {
  const u = new URL(req.url);
  if (u.search.length > MAX_QUERY_CHARS) return NextResponse.json({ ok: false, error: "request too large" });
  const parsed = parseInspectionSearch(u.search);
  if (!parsed.ok) return NextResponse.json({ ok: false, error: parsed.error });
  return collectorResponse(() => inspectExecutionRows(getDb(), parsed.selectors));
}
