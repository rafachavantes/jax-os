import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import {
  applyPendingOperations,
  BOOTSTRAP_DRAFT,
  SETTINGS_CAP,
  decodeAgentSettings,
  type ConnectionChoice,
  type EditorRevision,
  type PendingConnectionRef,
  type SettingsDraft,
} from "../../../lib/agent-settings";
import { MutationRejected } from "../../../lib/mutationOutcome";
import { readBodyCapped, requireSameOrigin } from "../../../server/api";
import {
  applyAgentSettings,
  HelperRefusal,
  previewAgentSettings,
  snapshotAgentSettings,
} from "../../../server/collectors/agent-settings";
import { sanitizeConnections } from "../../../server/collectors/opencode-providers";
import { runMutation } from "../../../server/mutations";

const REVISION_RE = /^[0-9a-f]{32}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

export const dynamic = "force-dynamic";

export type AgentSettingsRouteDeps = {
  snapshot: () => Promise<unknown>;
  preview: (intent: unknown, expected: unknown) => Promise<unknown>;
  apply: (intent: unknown, expected: unknown, operationId: string) => Promise<unknown>;
  sanitize: (raw: unknown) => unknown;
  pending?: () => Promise<PendingConnectionRef[]>;
};

const systemDeps: AgentSettingsRouteDeps = {
  snapshot: () => snapshotAgentSettings({}),
  preview: previewAgentSettings,
  apply: applyAgentSettings,
  sanitize: sanitizeConnections,
};

function reply(body: unknown) {
  const res = NextResponse.json(body);
  res.headers.set("Cache-Control", "no-store");
  return res;
}

function isObj(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function keysOk(obj: Record<string, unknown>, required: string[], optional: string[] = []): boolean {
  const allow = new Set([...required, ...optional]);
  for (const key of Object.keys(obj)) if (!allow.has(key)) return false;
  return required.every((key) => Object.hasOwn(obj, key));
}

function helperCode(e: unknown): string | null {
  return e instanceof HelperRefusal ? e.code : null;
}

function parseExpected(value: unknown): EditorRevision | null {
  if (!isObj(value) || !keysOk(value, ["settings", "sources"])) return null;
  const settings = value.settings;
  if (typeof settings !== "string") return null;
  if (settings !== "absent" && !REVISION_RE.test(settings)) return null;
  if (!isObj(value.sources) || !keysOk(value.sources, ["opencode.json", "opencode.jsonc"])) return null;
  const a = value.sources["opencode.json"];
  const b = value.sources["opencode.jsonc"];
  if (typeof a !== "string" || typeof b !== "string") return null;
  if ((a !== "absent" && !SHA256_RE.test(a)) || (b !== "absent" && !SHA256_RE.test(b))) return null;
  return { settings, sources: { "opencode.json": a, "opencode.jsonc": b } };
}

function parseDraft(reviewers: unknown, builders: unknown): SettingsDraft | null {
  try {
    const decoded = decodeAgentSettings(new TextEncoder().encode(JSON.stringify({
      schema_version: 1,
      revision: "0".repeat(32),
      reviewers,
      builders,
      source_revisions: { "opencode.json": "absent", "opencode.jsonc": "absent" },
    })));
    return { reviewers: decoded.reviewers, builders: decoded.builders };
  } catch {
    return null;
  }
}

function operationId(body: Record<string, unknown>): string | null {
  if (!Object.hasOwn(body, "operation_id")) return randomUUID();
  return typeof body.operation_id === "string" && UUID_RE.test(body.operation_id) ? body.operation_id : null;
}

async function readJson(req: Request, cap: number): Promise<{ ok: true; value: unknown } | { ok: false; error: string }> {
  const type = (req.headers.get("content-type") ?? "").split(";")[0].trim().toLowerCase();
  if (type !== "application/json") return { ok: false, error: "invalid payload" };
  const raw = await readBodyCapped(req, cap);
  if (!raw.ok) return { ok: false, error: raw.error === "request too large" ? raw.error : "invalid payload" };
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw.value);
  } catch {
    return { ok: false, error: "invalid payload" };
  }
  try {
    return { ok: true, value: text === "" ? {} : JSON.parse(text) };
  } catch {
    return { ok: false, error: "invalid payload" };
  }
}

function asApply(value: unknown) {
  if (!isObj(value)) return {};
  return {
    effect: value.effect,
    editor_revision: value.editor_revision,
    settings: value.settings,
  };
}

async function mutate(
  deps: AgentSettingsRouteDeps,
  kind: string,
  action: string,
  op: string,
  intent: unknown,
  expected: EditorRevision,
) {
  const result = await runMutation(
    { ts: new Date().toISOString(), kind, action, operation_id: op },
    async () => {
      try {
        return await deps.apply(intent, expected, op);
      } catch (e) {
        const code = helperCode(e);
        if (code) throw new MutationRejected(code);
        throw e;
      }
    },
  );
  if (result.ok) {
    const data = asApply(result.value);
    if (data.effect === "activation-pending") {
      return reply({
        ok: false,
        code: "activation-pending",
        effect: "activation-pending",
        audit: "recorded",
        error: "activation pending; reconcile current state before retrying",
        data,
      });
    }
    return reply({ ok: true, data });
  }
  const { status: _status, value, ...failure } = result;
  return reply({ ...failure, ...(value !== undefined ? { data: asApply(value) } : {}) });
}

async function handleGet(_req: Request, deps: AgentSettingsRouteDeps = systemDeps) {
  try {
    const snap = await deps.snapshot();
    const settings = isObj(snap) && "settings" in snap ? snap.settings : null;
    const draft = isObj(settings) && isObj(settings.reviewers) && isObj(settings.builders)
      ? { reviewers: settings.reviewers, builders: settings.builders }
      : BOOTSTRAP_DRAFT;
    const connections = deps.sanitize(isObj(snap) ? (snap.connections ?? snap) : snap) as ConnectionChoice[];
    const pending = deps.pending ? await deps.pending() : [];
    return reply({
      ok: true,
      data: {
        editor_revision: isObj(snap) ? snap.editor_revision : undefined,
        settings,
        draft,
        connections: applyPendingOperations(Array.isArray(connections) ? connections : [], pending),
      },
    });
  } catch (e) {
    return reply({ ok: false, error: helperCode(e) ?? "unavailable" });
  }
}

async function handlePost(req: Request, deps: AgentSettingsRouteDeps = systemDeps) {
  const origin = requireSameOrigin(req);
  if (origin) {
    origin.headers.set("Cache-Control", "no-store");
    return origin;
  }
  const read = await readJson(req, SETTINGS_CAP);
  if (!read.ok) return reply({ ok: false, error: read.error });
  if (!isObj(read.value)) return reply({ ok: false, error: "invalid payload" });
  const body = read.value;
  const action = body.action;
  if (action === "preview") {
    if (!keysOk(body, ["action", "expected", "reviewers", "builders"])) {
      return reply({ ok: false, error: "invalid payload" });
    }
    const expected = parseExpected(body.expected);
    const draft = parseDraft(body.reviewers, body.builders);
    if (!expected || !draft) return reply({ ok: false, error: "invalid payload" });
    try {
      const data = await deps.preview({ kind: "save-settings", ...draft }, expected);
      return reply({ ok: true, data });
    } catch (e) {
      return reply({ ok: false, error: helperCode(e) ?? "unavailable" });
    }
  }
  if (action === "save") {
    if (!keysOk(body, ["action", "expected", "reviewers", "builders"], ["operation_id"])) {
      return reply({ ok: false, error: "invalid payload" });
    }
    const expected = parseExpected(body.expected);
    const draft = parseDraft(body.reviewers, body.builders);
    const op = operationId(body);
    if (!expected || !draft || !op) return reply({ ok: false, error: "invalid payload" });
    return mutate(deps, "agent-settings-save", "save", op, { kind: "save-settings", ...draft }, expected);
  }
  if (action === "reconcile") {
    if (!keysOk(body, ["action", "expected"], ["operation_id"])) {
      return reply({ ok: false, error: "invalid payload" });
    }
    const expected = parseExpected(body.expected);
    const op = operationId(body);
    if (!expected || !op) return reply({ ok: false, error: "invalid payload" });
    try {
      const snap = await deps.snapshot();
      const settings = isObj(snap) ? snap.settings : null;
      const draft = isObj(settings) ? parseDraft(settings.reviewers, settings.builders) : null;
      if (!draft) return reply({ ok: false, error: "invalid payload" });
      return mutate(
        deps,
        "agent-settings-reconcile",
        "reconcile",
        op,
        { kind: "save-settings", ...draft },
        expected,
      );
    } catch (e) {
      return reply({ ok: false, error: helperCode(e) ?? "unavailable" });
    }
  }
  return reply({ ok: false, error: "invalid payload" });
}

export function GET(req: Request) {
  return handleGet(req, (req as Request & { jaxDeps?: AgentSettingsRouteDeps }).jaxDeps);
}

export function POST(req: Request) {
  return handlePost(req, (req as Request & { jaxDeps?: AgentSettingsRouteDeps }).jaxDeps);
}
