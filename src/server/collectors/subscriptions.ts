export type RateWindow = { usedPercent: number; resetsAt: string | null };

export type ProviderUsage = {
  provider: "anthropic" | "codex" | "opencode";
  plan: string | null;
  fiveHour: RateWindow | null;
  weekly: RateWindow | null;
  // Only OpenCode GO fills this — the other providers have no monthly window.
  monthly?: RateWindow | null;
  models: { name: string; weekly: RateWindow }[];
};

type Json = Record<string, unknown>;
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

// ---- Anthropic (GET https://api.anthropic.com/api/oauth/usage) ----

function claudeWindow(w: unknown): RateWindow | null {
  if (!w || typeof w !== "object") return null;
  const o = w as Json;
  const u = num(o.utilization);
  if (u === null) return null;
  return {
    // The endpoint reports utilization in percent (0-100): 1 means 1%, not a fraction.
    usedPercent: Math.round(u),
    resetsAt: str(o.resetsAt) ?? str(o.resets_at),
  };
}

export function parseClaudeUsage(json: unknown, plan: string | null): ProviderUsage {
  const o = (json ?? {}) as Json;
  const pick = (a: string, b: string) => (o[a] !== undefined ? o[a] : o[b]);
  const models: ProviderUsage["models"] = [];
  const opus = claudeWindow(pick("sevenDayOpus", "seven_day_opus"));
  const sonnet = claudeWindow(pick("sevenDaySonnet", "seven_day_sonnet"));
  if (opus) models.push({ name: "Opus", weekly: opus });
  if (sonnet) models.push({ name: "Sonnet", weekly: sonnet });
  return {
    provider: "anthropic",
    plan,
    fiveHour: claudeWindow(pick("fiveHour", "five_hour")),
    weekly: claudeWindow(pick("sevenDay", "seven_day")),
    models,
  };
}

export function parseClaudeCreds(json: unknown, nowMs: number): { token: string; plan: string | null } {
  const o = ((json ?? {}) as Json).claudeAiOauth as Json | undefined;
  const token = str(o?.accessToken);
  if (!token) throw new Error("missing-credentials");
  const exp = num(o?.expiresAt); // epoch ms
  if (exp !== null && exp - nowMs < 5 * 60_000) throw new Error("expired");
  return { token, plan: str(o?.subscriptionType) };
}

// ---- Codex (GET https://chatgpt.com/backend-api/wham/usage) ----

export function parseCodexCreds(json: unknown): { token: string; accountId: string | null } {
  const t = ((json ?? {}) as Json).tokens as Json | undefined;
  const token = str(t?.access_token);
  if (!token) throw new Error("missing-credentials");
  return { token, accountId: str(t?.account_id) };
}

export function parseCodexUsage(json: unknown): ProviderUsage {
  const o = (json ?? {}) as Json;
  const rl = (o.rate_limit ?? {}) as Json;
  let fiveHour: RateWindow | null = null;
  let weekly: RateWindow | null = null;
  const assign = (w: unknown, fallback: "five" | "week") => {
    if (!w || typeof w !== "object") return;
    const win = w as Json;
    const pct = num(win.used_percent) ?? num(win.usage_percent);
    if (pct === null) return;
    const reset = num(win.reset_at);
    const rw: RateWindow = {
      usedPercent: Math.round(pct),
      resetsAt: reset !== null ? new Date(reset * 1000).toISOString() : null,
    };
    const secs = num(win.limit_window_seconds);
    const isWeek = secs !== null ? secs >= 604_800 : fallback === "week";
    if (isWeek) weekly ??= rw;
    else fiveHour ??= rw;
  };
  assign(rl.primary_window, "five");
  assign(rl.secondary_window, "week");
  return { provider: "codex", plan: str(o.plan_type), fiveHour, weekly, models: [] };
}

// ---- OpenCode GO (GET https://opencode.ai/zen/go/v1/usage) ----
// 5h window = 20% of the monthly cap, weekly = 50%, monthly = 100% (docs).

function goWindow(w: unknown): RateWindow | null {
  if (!w || typeof w !== "object") return null;
  const o = w as Json;
  const pct = num(o.percent);
  if (pct === null) return null;
  return { usedPercent: Math.round(pct), resetsAt: str(o.resetsAt) };
}

export function parseOpenCodeGoUsage(json: unknown): ProviderUsage {
  const usage = (((json ?? {}) as Json).usage ?? {}) as Json;
  return {
    provider: "opencode",
    plan: null,
    fiveHour: goWindow(usage.rolling),
    weekly: goWindow(usage.weekly),
    monthly: goWindow(usage.monthly),
    models: [],
  };
}

// ---- Fetch glue (untested by design; parsers above are the tested part) ----

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type ProviderResult =
  | { ok: true; data: ProviderUsage; stale?: boolean }
  | { ok: false; error: string };

export type SubscriptionsPayload = {
  anthropic: ProviderResult;
  codex: ProviderResult;
  opencode: ProviderResult;
};

// SECURITY: tokens never leave this module — not logged, not in errors,
// not in the payload. Only derived percentages/timestamps go to the client.

const lastGood = new Map<string, ProviderUsage>();
const backoffUntil = new Map<string, number>();

function readJsonFile(path: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    throw new Error("missing-credentials");
  }
}

async function getJson(url: string, headers: Record<string, string>, name: string): Promise<unknown> {
  const res = await fetch(url, { headers, signal: AbortSignal.timeout(10_000) });
  if (res.status === 429) {
    const ra = Number(res.headers.get("retry-after"));
    backoffUntil.set(name, Date.now() + (Number.isFinite(ra) && ra > 0 ? ra * 1000 : 300_000));
    throw new Error("rate-limited");
  }
  if (res.status === 401 || res.status === 403) throw new Error("expired");
  if (!res.ok) throw new Error(`http-${res.status}`);
  return res.json();
}

async function fetchAnthropic(): Promise<ProviderUsage> {
  const creds = parseClaudeCreds(readJsonFile(join(homedir(), ".claude", ".credentials.json")), Date.now());
  const json = await getJson(
    "https://api.anthropic.com/api/oauth/usage",
    { Authorization: `Bearer ${creds.token}`, Accept: "application/json", "anthropic-beta": "oauth-2025-04-20" },
    "anthropic",
  );
  return parseClaudeUsage(json, creds.plan);
}

async function fetchCodex(): Promise<ProviderUsage> {
  const creds = parseCodexCreds(readJsonFile(join(homedir(), ".codex", "auth.json")));
  const headers: Record<string, string> = {
    Authorization: `Bearer ${creds.token}`,
    Accept: "application/json",
    "User-Agent": "CodexBar",
  };
  if (creds.accountId) headers["ChatGPT-Account-Id"] = creds.accountId;
  return parseCodexUsage(await getJson("https://chatgpt.com/backend-api/wham/usage", headers, "codex"));
}

async function fetchOpenCodeGo(): Promise<ProviderUsage> {
  const token = process.env.OPENCODE_GO_API_KEY;
  if (!token) throw new Error("missing-credentials");
  const json = await getJson(
    "https://opencode.ai/zen/go/v1/usage",
    { Authorization: `Bearer ${token}`, Accept: "application/json" },
    "opencode",
  );
  return parseOpenCodeGoUsage(json);
}

// Read-only configured/missing status for the three native credentials (F3, diff review
// e52d9e6dc555) — a boolean sibling on the settings route, computed fresh per request, never
// persisted, never the value itself. Reuses the SAME exported parsers the fetchers above call,
// discarding the returned token.
export function nativeCredentialsConfigured(): { claude: boolean; codex: boolean; opencodeGo: boolean } {
  const claude = (() => {
    try { parseClaudeCreds(readJsonFile(join(homedir(), ".claude", ".credentials.json")), Date.now()); return true; }
    catch { return false; }
  })();
  const codex = (() => {
    try { parseCodexCreds(readJsonFile(join(homedir(), ".codex", "auth.json"))); return true; }
    catch { return false; }
  })();
  return { claude, codex, opencodeGo: Boolean(process.env.OPENCODE_GO_API_KEY) };
}

async function guarded(name: string, fn: () => Promise<ProviderUsage>): Promise<ProviderResult> {
  if (Date.now() < (backoffUntil.get(name) ?? 0)) {
    const last = lastGood.get(name);
    return last ? { ok: true, data: last, stale: true } : { ok: false, error: "rate-limited" };
  }
  try {
    const data = await fn();
    lastGood.set(name, data);
    return { ok: true, data };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg === "rate-limited") {
      const last = lastGood.get(name);
      if (last) return { ok: true, data: last, stale: true };
    }
    return { ok: false, error: msg };
  }
}

export type SubscriptionsEnabled = { claude: boolean; codex: boolean; opencodeGo: boolean };
// F4 (cold review): an injectable seam over the three module-private fetchers, so tests spy on a
// disabled provider's OWN fetcher without touching real credentials/network. Production never
// passes this — the default below is the only production path; the three fns stay private.
export type SubscriptionsFetchers = { claude: typeof fetchAnthropic; codex: typeof fetchCodex; opencodeGo: typeof fetchOpenCodeGo };
const systemFetchers: SubscriptionsFetchers = { claude: fetchAnthropic, codex: fetchCodex, opencodeGo: fetchOpenCodeGo };
const DISABLED: ProviderResult = { ok: false, error: "disabled" };

// Decision 14 (spec): three independent flags — turning off exactly one leaves the other two
// slots' pass/fail behavior (including their own guarded() backoff/stale-cache logic) untouched.
// A disabled slot never reads its credential and never reaches the network.
export async function getSubscriptions(enabled: SubscriptionsEnabled, fetchers: SubscriptionsFetchers = systemFetchers): Promise<SubscriptionsPayload> {
  const [anthropic, codex, opencode] = await Promise.all([
    enabled.claude ? guarded("anthropic", fetchers.claude) : Promise.resolve(DISABLED),
    enabled.codex ? guarded("codex", fetchers.codex) : Promise.resolve(DISABLED),
    enabled.opencodeGo ? guarded("opencode", fetchers.opencodeGo) : Promise.resolve(DISABLED),
  ]);
  return { anthropic, codex, opencode };
}
