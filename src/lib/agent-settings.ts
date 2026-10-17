import modelDefaults from "../../workflow/fixtures/model-defaults-v1.json";

export const SETTINGS_CAP = 128 * 1024;
export const CONNECTION_ADAPTERS = ["xai", "openrouter", "openai-compatible"] as const;

export type ModelInput = {
  id: string;
  label?: string;
  context?: number;
  limit?: { context: number; output: number };
  reasoning?: boolean;
  tool_call?: boolean;
  effort_template?: "none" | "reasoning" | "reasoning_effort";
  efforts?: string[];
};

export type ProfileName = "default" | "fallback";
export type SourceRevisions = { "opencode.json": string; "opencode.jsonc": string };
export type EditorRevision = { settings: string; sources: SourceRevisions };
export type Health =
  | "registered"
  | "missing"
  | "oauth-managed"
  | "sync-pending"
  | "invalid"
  | "unavailable";
export type CredentialRef =
  | { kind: "native" }
  | { kind: "env"; env: string };
export type ModelChoice = {
  id: string;
  label: string;
  origin: "catalog" | "override";
  efforts: string[];
  no_effort: boolean;
  compatible: boolean;
  reason: string | null;
  context: number | null;
  effort_template: "none" | "reasoning" | "reasoning_effort" | null;
};
export type ConnectionChoice = {
  id: string;
  label: string;
  adapter: string | null;
  base_url: string | null;
  auth: "api-key" | "oauth" | "none" | "unknown";
  health: Health;
  credential: CredentialRef | null;
  editable: boolean;
  reason: string | null;
  used_by: ProfileName[];
  models: ModelChoice[];
  pending?: boolean;
};

export type PercentileKey = "p50" | "p75" | "p90" | "p99";
export type Percentile = Partial<Record<PercentileKey, number>>;
export type PriceCeiling = { prompt?: number; completion?: number };
export type Routing = {
  sort: "price";
  allow_fallbacks: boolean;
  preferred_min_throughput?: Percentile;
  preferred_max_latency?: Percentile;
  max_price?: PriceCeiling;
  only?: string[];
  ignore?: string[];
  quantizations?: string[];
};
export type RoutingDraft = {
  sort: "price";
  allow_fallbacks: boolean;
  preferred_min_throughput?: Percentile | null;
  preferred_max_latency?: Percentile | null;
  max_price?: PriceCeiling | null;
  only?: string[] | null;
  ignore?: string[] | null;
  quantizations?: string[] | null;
};
export type ReviewerSpec = { model: string; effort: string };
export type BuilderProfile = {
  connection: string;
  model: string;
  effort: string | null;
  credential: CredentialRef;
  routing: Routing | null;
};
export type AgentSettings = {
  schema_version: 1;
  revision: string;
  reviewers: { claude: ReviewerSpec; codex: ReviewerSpec };
  builders: { default: BuilderProfile; fallback: BuilderProfile };
  source_revisions: SourceRevisions;
};
export type SettingsDraft = {
  reviewers: AgentSettings["reviewers"];
  builders: AgentSettings["builders"];
};

const ROOT_KEYS = ["schema_version", "revision", "reviewers", "builders", "source_revisions"] as const;
const REVIEWER_FIELD_KEYS = ["model", "effort"] as const;
const PROFILE_KEYS = ["connection", "model", "effort", "credential", "routing"] as const;
const SOURCE_FILES = ["opencode.json", "opencode.jsonc"] as const;
const ROUTING_KEYS = new Set([
  "sort",
  "allow_fallbacks",
  "preferred_min_throughput",
  "preferred_max_latency",
  "max_price",
  "only",
  "ignore",
  "quantizations",
]);
const PERCENTILES = new Set(["p50", "p75", "p90", "p99"]);
// Wire order from OpenRouter's provider.quantizations enum; also the UI checkbox order.
export const QUANTIZATIONS = [
  "int4", "int8", "fp4", "mxfp4", "nvfp4", "fp6", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown",
] as const;
const QUANTIZATION_SET = new Set<string>(QUANTIZATIONS);
const NATIVE_ENV = new Set([
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "OPENROUTER_API_KEY",
  "XAI_API_KEY",
  "DEEPSEEK_API_KEY",
  "GOOGLE_GENERATIVE_AI_API_KEY",
  "GEMINI_API_KEY",
  "GROQ_API_KEY",
  "MISTRAL_API_KEY",
  "TOGETHER_API_KEY",
  "CEREBRAS_API_KEY",
  "COHERE_API_KEY",
  "FIREWORKS_API_KEY",
  "PERPLEXITY_API_KEY",
]);
const REVIEWER_EFFORTS: Record<"claude" | "codex", Set<string>> = {
  claude: new Set(["low", "medium", "high", "xhigh", "max"]),
  codex: new Set(["minimal", "low", "medium", "high", "xhigh"]),
};
const CONNECTION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const EFFORT_RE = /^[A-Za-z0-9._-]{1,64}$/;
const REVISION_RE = /^[0-9a-f]{32}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const JAX_ENV_RE = /^JAX_PROVIDER_[A-Z0-9_]+_API_KEY$/;

const NUM = Symbol("jsonNum");
type JsonNum = { [NUM]: true; int: boolean; v: number };

export const BOOTSTRAP_DRAFT: SettingsDraft = {
  reviewers: {
    claude: { ...modelDefaults.reviewers.claude },
    codex: { ...modelDefaults.reviewers.codex },
  },
  builders: {
    default: {
      connection: "xai",
      ...modelDefaults.builders.default,
      credential: { kind: "native" },
      routing: null,
    },
    fallback: {
      connection: "openrouter",
      ...modelDefaults.builders.fallback,
      credential: { kind: "native" },
      routing: { sort: "price", allow_fallbacks: true },
    },
  },
};

function malformed(): never {
  throw new Error("agent-settings-malformed");
}

function isNum(value: unknown): value is JsonNum {
  return typeof value === "object" && value !== null && NUM in value;
}

function isObj(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !isNum(value);
}

function exactKeys(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (!isObj(value)) malformed();
  const got = Object.keys(value);
  if (got.length !== keys.length) malformed();
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(value, key)) malformed();
  }
  return value;
}

function isControlOrSpace(ch: string): boolean {
  return /[\s\p{C}]/u.test(ch);
}

function requireModel(value: unknown): asserts value is string {
  if (typeof value !== "string" || !value || value.startsWith("-")) malformed();
  if ([...value].some(isControlOrSpace) || new TextEncoder().encode(value).length > 256) {
    malformed();
  }
}

function finiteNum(value: unknown, opts: { positive?: boolean; nonnegative?: boolean }): boolean {
  if (!isNum(value) || !Number.isFinite(value.v)) return false;
  if (opts.positive && !(value.v > 0)) return false;
  if (opts.nonnegative && value.v < 0) return false;
  return true;
}

function percentile(value: unknown): void {
  if (!isObj(value)) malformed();
  const keys = Object.keys(value);
  if (keys.length !== 1 || !PERCENTILES.has(keys[0])) malformed();
  if (!finiteNum(value[keys[0]], { positive: true })) malformed();
}

function price(value: unknown): void {
  if (!isObj(value)) malformed();
  const keys = Object.keys(value);
  if (!keys.length) malformed();
  for (const key of keys) {
    if (key !== "prompt" && key !== "completion") malformed();
    if (!finiteNum(value[key], { nonnegative: true })) malformed();
  }
}

function endpointIds(value: unknown): Set<string> {
  if (!Array.isArray(value) || !value.length || value.length > 128) malformed();
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "string" || !item || item.length > 256) malformed();
    if ([...item].some(isControlOrSpace) || seen.has(item)) malformed();
    seen.add(item);
  }
  return seen;
}

function quantizations(value: unknown): void {
  if (!Array.isArray(value) || !value.length) malformed();
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "string" || !QUANTIZATION_SET.has(item) || seen.has(item)) malformed();
    seen.add(item);
  }
}

function routing(value: unknown): void {
  if (value === null) return;
  if (!isObj(value)) malformed();
  const keys = Object.keys(value);
  if (!keys.includes("sort") || !keys.includes("allow_fallbacks")) malformed();
  for (const key of keys) {
    if (!ROUTING_KEYS.has(key)) malformed();
  }
  if (value.sort !== "price" || typeof value.allow_fallbacks !== "boolean") malformed();
  if ("preferred_min_throughput" in value) percentile(value.preferred_min_throughput);
  if ("preferred_max_latency" in value) percentile(value.preferred_max_latency);
  if ("max_price" in value) price(value.max_price);
  if ("quantizations" in value) quantizations(value.quantizations);
  const only = "only" in value ? endpointIds(value.only) : new Set<string>();
  const ignore = "ignore" in value ? endpointIds(value.ignore) : new Set<string>();
  for (const id of only) {
    if (ignore.has(id)) malformed();
  }
}

export function isValidCredentialEnvName(env: string): boolean {
  return NATIVE_ENV.has(env) || JAX_ENV_RE.test(env);
}

export function validCredentialBinding(value: unknown): value is CredentialRef {
  try {
    credential(value);
    return true;
  } catch {
    return false;
  }
}

function credential(value: unknown): void {
  if (!isObj(value) || !("kind" in value)) malformed();
  if (value.kind === "native") {
    if (Object.keys(value).length !== 1) malformed();
    return;
  }
  if (value.kind !== "env") malformed();
  if (Object.keys(value).length !== 2 || !("env" in value)) malformed();
  const env = value.env;
  if (typeof env !== "string") malformed();
  if (isValidCredentialEnvName(env)) return;
  malformed();
}

function profile(value: unknown): void {
  const obj = exactKeys(value, PROFILE_KEYS);
  if (typeof obj.connection !== "string" || !CONNECTION_RE.test(obj.connection)) malformed();
  requireModel(obj.model);
  const effort = obj.effort;
  if (effort !== null && (typeof effort !== "string" || !EFFORT_RE.test(effort))) malformed();
  credential(obj.credential);
  routing(obj.routing);
}

function reviewer(value: unknown, runtime: "claude" | "codex"): void {
  const obj = exactKeys(value, REVIEWER_FIELD_KEYS);
  requireModel(obj.model);
  if (typeof obj.effort !== "string" || !REVIEWER_EFFORTS[runtime].has(obj.effort)) malformed();
}

function parseStrictJson(text: string): unknown {
  let i = 0;
  const n = text.length;

  function skipWs(): void {
    while (i < n) {
      const c = text[i];
      if (c === " " || c === "\t" || c === "\n" || c === "\r") i++;
      else break;
    }
  }

  function fail(): never {
    throw new SyntaxError("json");
  }

  function parseLit(lit: string, value: unknown): unknown {
    if (text.slice(i, i + lit.length) !== lit) fail();
    i += lit.length;
    return value;
  }

  function parseString(): string {
    if (text[i] !== '"') fail();
    i++;
    let out = "";
    while (i < n) {
      const c = text[i];
      if (c === '"') {
        i++;
        return out;
      }
      if (c === "\\") {
        i++;
        if (i >= n) fail();
        const e = text[i++];
        if (e === "u") {
          const hex = text.slice(i, i + 4);
          if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail();
          out += String.fromCharCode(parseInt(hex, 16));
          i += 4;
          continue;
        }
        const mapped: Record<string, string> = {
          '"': '"',
          "\\": "\\",
          "/": "/",
          b: "\b",
          f: "\f",
          n: "\n",
          r: "\r",
          t: "\t",
        };
        if (mapped[e] === undefined) fail();
        out += mapped[e];
        continue;
      }
      if (c.charCodeAt(0) < 0x20) fail();
      out += c;
      i++;
    }
    fail();
  }

  function parseNumber(): JsonNum {
    const start = i;
    if (text[i] === "-") i++;
    if (i >= n) fail();
    if (text[i] === "0") i++;
    else if (text[i] >= "1" && text[i] <= "9") {
      i++;
      while (i < n && text[i] >= "0" && text[i] <= "9") i++;
    } else fail();
    let int = true;
    if (i < n && text[i] === ".") {
      int = false;
      i++;
      if (i >= n || text[i] < "0" || text[i] > "9") fail();
      while (i < n && text[i] >= "0" && text[i] <= "9") i++;
    }
    if (i < n && (text[i] === "e" || text[i] === "E")) {
      int = false;
      i++;
      if (i < n && (text[i] === "+" || text[i] === "-")) i++;
      if (i >= n || text[i] < "0" || text[i] > "9") fail();
      while (i < n && text[i] >= "0" && text[i] <= "9") i++;
    }
    const v = Number(text.slice(start, i));
    if (!Number.isFinite(v)) fail();
    return { [NUM]: true, int, v };
  }

  function parseArray(depth: number): unknown[] {
    i++;
    skipWs();
    if (text[i] === "]") {
      i++;
      return [];
    }
    const out: unknown[] = [];
    while (true) {
      out.push(parseValue(depth));
      skipWs();
      if (text[i] === "]") {
        i++;
        return out;
      }
      if (text[i] !== ",") fail();
      i++;
      skipWs();
    }
  }

  function parseObject(depth: number): Record<string, unknown> {
    i++;
    skipWs();
    if (text[i] === "}") {
      i++;
      return {};
    }
    const out: Record<string, unknown> = {};
    while (true) {
      skipWs();
      if (text[i] !== '"') fail();
      const key = parseString();
      if (Object.prototype.hasOwnProperty.call(out, key)) fail();
      skipWs();
      if (text[i] !== ":") fail();
      i++;
      out[key] = parseValue(depth);
      skipWs();
      if (text[i] === "}") {
        i++;
        return out;
      }
      if (text[i] !== ",") fail();
      i++;
    }
  }

  function parseValue(depth: number): unknown {
    if (depth > 64) fail();
    skipWs();
    if (i >= n) fail();
    const c = text[i];
    if (c === "{") return parseObject(depth + 1);
    if (c === "[") return parseArray(depth + 1);
    if (c === '"') return parseString();
    if (c === "t") return parseLit("true", true);
    if (c === "f") return parseLit("false", false);
    if (c === "n") return parseLit("null", null);
    if (c === "-" || (c >= "0" && c <= "9")) return parseNumber();
    fail();
  }

  const value = parseValue(0);
  skipWs();
  if (i !== n) fail();
  return value;
}

function unwrap(value: unknown): unknown {
  if (isNum(value)) return value.v;
  if (Array.isArray(value)) return value.map(unwrap);
  if (isObj(value)) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = unwrap(v);
    return out;
  }
  return value;
}

export function decodeAgentSettings(raw: Uint8Array): AgentSettings {
  if (!(raw instanceof Uint8Array)) malformed();
  if (raw.byteLength > SETTINGS_CAP) throw new Error("agent-settings-too-large");
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(raw);
  } catch {
    malformed();
  }
  let parsed: unknown;
  try {
    parsed = parseStrictJson(text);
  } catch {
    malformed();
  }
  const root = exactKeys(parsed, ROOT_KEYS);
  if (!isNum(root.schema_version) || !root.schema_version.int || root.schema_version.v !== 1) {
    malformed();
  }
  if (typeof root.revision !== "string" || !REVISION_RE.test(root.revision)) malformed();
  const reviewers = exactKeys(root.reviewers, ["claude", "codex"]);
  reviewer(reviewers.claude, "claude");
  reviewer(reviewers.codex, "codex");
  const builders = exactKeys(root.builders, ["default", "fallback"]);
  profile(builders.default);
  profile(builders.fallback);
  const sources = exactKeys(root.source_revisions, SOURCE_FILES);
  for (const token of Object.values(sources)) {
    if (token !== "absent" && (typeof token !== "string" || !SHA256_RE.test(token))) malformed();
  }
  return unwrap(parsed) as AgentSettings;
}

export function referencedProfiles(
  settings: AgentSettings,
  connectionId: string,
  modelId?: string,
): ProfileName[] {
  const names: ProfileName[] = [];
  for (const name of ["default", "fallback"] as const) {
    const row = settings.builders[name];
    if (row.connection !== connectionId) continue;
    if (modelId !== undefined && row.model !== modelId) continue;
    names.push(name);
  }
  return names;
}

export function isSettingsDraftDirty(baseline: SettingsDraft, draft: SettingsDraft): boolean {
  return JSON.stringify(baseline) !== JSON.stringify(draft);
}

export function pollingWouldOverwrite(
  dirty: boolean,
  baselineRevision: string,
  serverRevision: string,
): boolean {
  return dirty && baselineRevision !== serverRevision;
}

export function compactRouting(routing: RoutingDraft | null): Routing | null {
  if (routing == null) return null;
  const out: Routing = { sort: "price", allow_fallbacks: routing.allow_fallbacks };
  if (routing.preferred_min_throughput) out.preferred_min_throughput = routing.preferred_min_throughput;
  if (routing.preferred_max_latency) out.preferred_max_latency = routing.preferred_max_latency;
  if (routing.max_price) {
    const priceOut: PriceCeiling = {};
    if (routing.max_price.prompt != null) priceOut.prompt = routing.max_price.prompt;
    if (routing.max_price.completion != null) priceOut.completion = routing.max_price.completion;
    if (Object.keys(priceOut).length) out.max_price = priceOut;
  }
  if (routing.only && routing.only.length) out.only = routing.only;
  if (routing.ignore && routing.ignore.length) out.ignore = routing.ignore;
  if (routing.quantizations && routing.quantizations.length) out.quantizations = routing.quantizations;
  return out;
}

export function percentileMatches(
  goal: Percentile,
  observation: (Percentile & { window?: string }) | null,
): boolean {
  if (!observation) return false;
  const key = Object.keys(goal)[0] as PercentileKey | undefined;
  if (!key) return false;
  const observed = observation[key];
  return typeof observed === "number" && Number.isFinite(observed);
}

export const MODEL_INPUT_KEYS = [
  "id",
  "label",
  "context",
  "limit",
  "reasoning",
  "tool_call",
  "effort_template",
  "efforts",
] as const;
export const EFFORT_TEMPLATES = ["none", "reasoning", "reasoning_effort"] as const;

function validId(s: string): boolean {
  return !!s && !s.startsWith("jaxflow-builder-") && !/[\r\n\0]/.test(s) && new TextEncoder().encode(s).length <= 256;
}

export function validateModelInput(value: unknown): value is ModelInput {
  if (!isObj(value)) return false;
  const model = value as Record<string, unknown>;
  for (const key of Object.keys(model)) {
    if (!(MODEL_INPUT_KEYS as readonly string[]).includes(key)) return false;
  }
  if (typeof model.id !== "string" || !validId(model.id)) return false;
  if ("label" in model) {
    const label = model.label;
    if (!(label === undefined || (typeof label === "string" && new TextEncoder().encode(label).length <= 256))) {
      return false;
    }
  }
  if ("context" in model) {
    const context = model.context;
    if (typeof context !== "number" || !Number.isInteger(context) || context <= 0) return false;
  }
  if ("context" in model && "limit" in model) return false;
  if ("limit" in model) {
    const limit = model.limit;
    if (!isObj(limit)) return false;
    const row = limit as Record<string, unknown>;
    for (const key of Object.keys(row)) {
      if (key !== "context" && key !== "output") return false;
      if (typeof row[key] !== "number" || !Number.isFinite(row[key]) || (row[key] as number) <= 0) return false;
    }
  }
  if ("reasoning" in model && typeof model.reasoning !== "boolean") return false;
  if ("tool_call" in model && typeof model.tool_call !== "boolean") return false;
  const template = model.effort_template;
  if (template !== undefined && (typeof template !== "string" || !(EFFORT_TEMPLATES as readonly string[]).includes(template))) {
    return false;
  }
  if ("efforts" in model) {
    const efforts = model.efforts;
    if (!Array.isArray(efforts) || efforts.some((e) => typeof e !== "string")) return false;
    const list = efforts as unknown[];
    if (new Set<string>(list as string[]).size !== list.length) return false;
    if (list.some((e) => !/^[A-Za-z0-9._-]{1,64}$/.test(e as string))) return false;
  }
  if (model.reasoning === false) {
    if (template !== undefined && template !== "none") return false;
    if (model.efforts && (model.efforts as unknown[]).length) return false;
  }
  if (template !== undefined) {
    if (typeof model.reasoning !== "boolean" || typeof model.tool_call !== "boolean") return false;
    const limit = model.limit;
    if (!limit || typeof (limit as Record<string, unknown>).context !== "number" || typeof (limit as Record<string, unknown>).output !== "number") {
      return false;
    }
    if (model.tool_call !== true) return false;
    if (template === "none") {
      if (model.reasoning !== false || (model.efforts && (model.efforts as unknown[]).length)) return false;
    } else {
      if (model.reasoning !== true || !model.efforts || !(model.efforts as unknown[]).length) return false;
    }
  }
  return true;
}

export function modelInputPayload(model: {
  id: string;
  label?: string;
  context?: number | null;
  output?: number | null;
  reasoning?: boolean | null;
  toolCall?: boolean | null;
  effort?: "none" | "reasoning" | "reasoning_effort";
  efforts?: string[];
}): ModelInput | null {
  const out: ModelInput = { id: model.id };
  if (model.label) out.label = model.label;
  if (model.reasoning != null) out.reasoning = model.reasoning;
  if (model.toolCall != null) out.tool_call = model.toolCall;
  if (model.effort) out.effort_template = model.effort;
  if (model.context != null && model.output != null) {
    out.limit = { context: model.context, output: model.output };
  } else if (model.context != null) {
    out.context = model.context;
  }
  if (model.efforts && model.efforts.length) out.efforts = model.efforts;
  return validateModelInput(out) ? out : null;
}

export type RoutingLabels = {
  sort: string;
  fallbacks: string;
  throughput: string;
  latency: string;
  maxPrompt: string;
  maxCompletion: string;
  only: string;
  ignore: string;
};

function percentileValue(p: unknown): string | null {
  if (!p || typeof p !== "object") return null;
  const rec = p as Record<string, unknown>;
  for (const key of Object.keys(rec)) {
    if (typeof rec[key] === "number") return `${key}:${String(rec[key])}`;
  }
  return null;
}

export function routingPreviewText(routing: unknown, labels: RoutingLabels): string | null {
  if (!routing || typeof routing !== "object") return null;
  const row = routing as Record<string, unknown>;
  const sort = row.sort === "price" ? labels.sort : String(row.sort);
  const parts = [
    typeof row.allow_fallbacks === "boolean" ? `${sort} ${row.allow_fallbacks ? "+" : "-"}${labels.fallbacks}` : sort,
  ];
  const throughput = percentileValue(row.preferred_min_throughput);
  const latency = percentileValue(row.preferred_max_latency);
  if (throughput) parts.push(`${labels.throughput} ${throughput}`);
  if (latency) parts.push(`${labels.latency} ${latency}`);
  if (row.max_price && typeof row.max_price === "object") {
    const mp = row.max_price as Record<string, unknown>;
    if (typeof mp.prompt === "number") parts.push(`${labels.maxPrompt}=${String(mp.prompt)}`);
    if (typeof mp.completion === "number") parts.push(`${labels.maxCompletion}=${String(mp.completion)}`);
  }
  if (Array.isArray(row.only) && row.only.length) parts.push(`${labels.only}=${(row.only as string[]).join(",")}`);
  if (Array.isArray(row.ignore) && row.ignore.length) parts.push(`${labels.ignore}=${(row.ignore as string[]).join(",")}`);
  return parts.join(" · ");
}

export type PreviewBindingRow = {
  profile: ProfileName;
  connection: string;
  model: string;
  effort: string | null;
  routing: unknown;
};
export type ProfilePreviewRow = {
  profile: ProfileName;
  before: PreviewBindingRow | null;
  after: PreviewBindingRow | null;
};

function isPreviewRow(value: unknown): value is PreviewBindingRow {
  if (!isObj(value)) return false;
  const row = value as Record<string, unknown>;
  return (row.profile === "default" || row.profile === "fallback")
    && typeof row.connection === "string"
    && typeof row.model === "string"
    && (row.effort === null || row.effort === undefined || typeof row.effort === "string");
}

function previewRow(value: unknown): PreviewBindingRow {
  const row = value as PreviewBindingRow;
  return {
    profile: row.profile,
    connection: row.connection,
    model: row.model,
    effort: typeof row.effort === "string" ? row.effort : null,
    routing: row.routing ?? null,
  };
}

export function previewFingerprint(draft: SettingsDraft, revision: EditorRevision | null): string | null {
  if (!revision) return null;
  const { reviewers, builders } = draft;
  const { settings, sources } = revision;
  try {
    return JSON.stringify({ reviewers, builders, settings, sources });
  } catch {
    return null;
  }
}

export function previewIsCurrent(
  stored: string | null,
  draft: SettingsDraft,
  revision: EditorRevision | null,
): boolean {
  if (!stored || !revision) return false;
  const current = previewFingerprint(draft, revision);
  return current !== null && current === stored;
}

export type ActivationGate = {
  firstActivation: boolean;
  dirty: boolean;
  revision: EditorRevision | null;
  conflict: boolean;
  valid: boolean;
  busy: boolean;
  acknowledged: boolean;
};

export function activationSaveEnabled(gate: ActivationGate): boolean {
  if (!gate.revision || gate.conflict || !gate.valid || gate.busy) return false;
  return gate.firstActivation ? gate.acknowledged : gate.dirty;
}

function validReviewerModel(model: string): boolean {
  if (typeof model !== "string" || !model || model.startsWith("-")) return false;
  return ![...model].some(isControlOrSpace) && new TextEncoder().encode(model).length <= 256;
}

// `builders` false = the builder section is hidden (OpenCode off): its profiles are saved unchanged and not validated.
export function draftIsValid(draft: SettingsDraft, connections: ConnectionChoice[], builders = true): boolean {
  for (const runtime of ["claude", "codex"] as const) {
    const spec = draft.reviewers[runtime];
    if (!validReviewerModel(spec.model)) return false;
    if (!REVIEWER_EFFORTS[runtime].has(spec.effort)) return false;
  }
  if (!builders) return true;
  const selectable = selectableConnections(connections);
  for (const name of ["default", "fallback"] as const) {
    const profile = draft.builders[name];
    const connection = selectable.find((row) => row.id === profile.connection);
    if (!connection || connection.pending) return false;
    const model = configuredModels(connection).find((row) => row.id === profile.model);
    if (!model) return false;
    if (profile.effort !== null && (model.no_effort || !model.efforts.includes(profile.effort))) return false;
  }
  return true;
}

export function previewProfiles(value: unknown): { affected: string[]; rows: ProfilePreviewRow[] } | null {
  if (!isObj(value)) return null;
  const data = value as Record<string, unknown>;
  const affected = Array.isArray(data.affected) ? data.affected.filter((x): x is string => typeof x === "string") : [];
  const aliases = data.aliases;
  if (!isObj(aliases)) return null;
  const before = Array.isArray(aliases.before) ? aliases.before.filter(isPreviewRow) : [];
  const after = Array.isArray(aliases.after) ? aliases.after.filter(isPreviewRow) : [];
  const map = (rows: PreviewBindingRow[]): Partial<Record<ProfileName, PreviewBindingRow>> => {
    const out: Partial<Record<ProfileName, PreviewBindingRow>> = {};
    for (const row of rows) out[row.profile] = previewRow(row);
    return out;
  };
  const beforeMap = map(before);
  const afterMap = map(after);
  const rows: ProfilePreviewRow[] = (["default", "fallback"] as const).map((name) => ({
    profile: name,
    before: beforeMap[name] ?? null,
    after: afterMap[name] ?? null,
  }));
  return { affected, rows };
}

export const TOOLS_TABS = ["general", "agents", "providers", "rules", "inventory"] as const;
export type ToolsTab = (typeof TOOLS_TABS)[number];
export type CredentialOutcomeAction = "reconcile" | "retry-sync" | "status";
export type SettingsWriteKind = "ok" | "activation-pending" | "applied-unrecorded" | "unconfirmed" | "refused";

export const CLAUDE_REVIEWER_EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const;
export const CODEX_REVIEWER_EFFORTS = ["minimal", "low", "medium", "high", "xhigh"] as const;

// MOA-504 D10: the Providers tab configures OpenCode, so it exists only while OpenCode is on.
export function visibleToolsTabs(agents: { opencode: boolean }): readonly ToolsTab[] {
  return agents.opencode ? TOOLS_TABS : TOOLS_TABS.filter((tab) => tab !== "providers");
}

export function resolveToolsTab(selected: ToolsTab, tabs: readonly ToolsTab[]): ToolsTab {
  return tabs.includes(selected) ? selected : "general";
}

export function nextToolsTab(current: ToolsTab, key: string, tabs: readonly ToolsTab[] = TOOLS_TABS): ToolsTab {
  const i = tabs.indexOf(current);
  const n = tabs.length;
  if (key === "Home") return tabs[0];
  if (key === "End") return tabs[n - 1];
  if (key === "ArrowRight") return tabs[(i + 1) % n];
  if (key === "ArrowLeft") return tabs[(i - 1 + n) % n];
  return current;
}

export function saveIsStale(expected: EditorRevision, server: EditorRevision): boolean {
  return (
    expected.settings !== server.settings
    || expected.sources["opencode.json"] !== server.sources["opencode.json"]
    || expected.sources["opencode.jsonc"] !== server.sources["opencode.jsonc"]
  );
}

export function isSelectableConnection(c: ConnectionChoice): boolean {
  if (c.pending) return false;
  return (c.health === "registered" || c.health === "oauth-managed") && c.credential != null;
}

export type PendingConnectionRef = { provider_id: string };

export function applyPendingOperations(
  connections: ConnectionChoice[],
  pending: PendingConnectionRef[],
): ConnectionChoice[] {
  const ids = new Set(pending.map((row) => row.provider_id).filter(Boolean));
  if (!ids.size) return connections;
  return connections.map((row) => (ids.has(row.id) ? { ...row, pending: true } : row));
}

export function selectableConnections(rows: ConnectionChoice[]): ConnectionChoice[] {
  return rows.filter(isSelectableConnection);
}

export function configuredModels(c: ConnectionChoice | undefined): ModelChoice[] {
  return c ? c.models.filter((m) => m.compatible) : [];
}

export function credentialOutcomeActions(
  stage: string | null,
  _outcome: string | null,
): CredentialOutcomeAction[] {
  if (stage === "remote-uncertain") return ["reconcile", "status"];
  if (stage === "sync-pending" || stage === "remote-saved") return ["retry-sync", "status"];
  if (stage === "activation-pending") return ["reconcile", "status"];
  if (stage === "submitted" || stage === "reserved") return ["status"];
  return [];
}

export function shouldConfirmLeaveTools(pathname: string, href: string, dirty: boolean): boolean {
  return dirty && pathname === "/settings" && href !== "/settings";
}

export function settingsWriteKind(res: unknown): SettingsWriteKind {
  if (!res || typeof res !== "object" || Array.isArray(res)) return "unconfirmed";
  const env = res as Record<string, unknown>;
  if (env.ok === true) return "ok";
  if (env.effect === "applied") return "applied-unrecorded";
  if (env.effect === "activation-pending") return "activation-pending";
  if (env.effect === "unconfirmed") return "unconfirmed";
  if (env.ok === false) return "refused";
  return "unconfirmed";
}

const EDITOR_REVISION_RE = /^[0-9a-f]{32}$/;
const EDITOR_SOURCE_RE = /^[0-9a-f]{64}$/;

function keysExactly(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const got = Object.keys(value);
  return got.length === keys.length && keys.every((key) => Object.prototype.hasOwnProperty.call(value, key));
}

// Shared EditorRevision parser: provider/settings writers and clients reject a
// malformed or partial revision instead of trusting an unchecked cast.
export function parseEditorRevision(value: unknown): EditorRevision | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const obj = value as Record<string, unknown>;
  if (!keysExactly(obj, ["settings", "sources"])) return null;
  const settings = obj.settings;
  if (typeof settings !== "string" || (settings !== "absent" && !EDITOR_REVISION_RE.test(settings))) return null;
  const sources = obj.sources;
  if (!sources || typeof sources !== "object" || Array.isArray(sources)) return null;
  const row = sources as Record<string, unknown>;
  if (!keysExactly(row, ["opencode.json", "opencode.jsonc"])) return null;
  const a = row["opencode.json"];
  const b = row["opencode.jsonc"];
  if (typeof a !== "string" || typeof b !== "string") return null;
  if ((a !== "absent" && !EDITOR_SOURCE_RE.test(a)) || (b !== "absent" && !EDITOR_SOURCE_RE.test(b))) return null;
  return { settings, sources: { "opencode.json": a, "opencode.jsonc": b } };
}
