import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import {
  applyPendingOperations,
  isValidCredentialEnvName,
  parseEditorRevision,
  SETTINGS_CAP,
  validCredentialBinding,
  validateModelInput,
  type ConnectionChoice,
  type EditorRevision,
  type PendingConnectionRef,
} from "../../../lib/agent-settings";
import { MutationRejected } from "../../../lib/mutationOutcome";
import { readBodyCapped, requireSameOrigin } from "../../../server/api";
import { HelperRefusal } from "../../../server/collectors/agent-settings";
import { loadJaxosEnv } from "../../../server/envFile";
import {
  applyProviders,
  parseProviderConnection,
  previewProviders,
  sanitizeConnections,
  snapshotProviders,
} from "../../../server/collectors/opencode-providers";
import { runMutation } from "../../../server/mutations";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const CONNECTION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const KINDS = ["save-connection", "save-model", "remove-model", "remove-connection", "bind-credential"] as const;

export const dynamic = "force-dynamic";

export type ProvidersRouteDeps = {
  snapshot: () => Promise<unknown>;
  preview: (intent: unknown, expected: unknown) => Promise<unknown>;
  apply: (intent: unknown, expected: unknown, operationId: string) => Promise<unknown>;
  sanitize: (raw: unknown) => unknown;
  pending?: () => Promise<PendingConnectionRef[]>;
};

const systemDeps: ProvidersRouteDeps = {
  // MOA-498: no more provider-credential mutations are ever written (Bitwarden removed) — a
  // saved binding's env name/kind now lives directly in the connection's own profile, not a
  // separate audit trail this snapshot used to decorate itself with.
  snapshot: () => snapshotProviders({}),
  preview: previewProviders,
  apply: applyProviders,
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
  return parseEditorRevision(value);
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

function parseModel(value: unknown) {
  if (!isObj(value)) return null;
  return validateModelInput(value) ? value : null;
}

function parseIntent(kind: unknown, body: Record<string, unknown>) {
  if (kind === "save-connection") {
    const connection = parseProviderConnection(body.connection);
    return connection ? { kind, connection } : null;
  }
  if (kind === "save-model") {
    if (typeof body.connection !== "string" || !CONNECTION_RE.test(body.connection)) return null;
    const model = parseModel(body.model);
    return model ? { kind, connection: body.connection, model } : null;
  }
  if (kind === "remove-model") {
    if (typeof body.connection !== "string" || !CONNECTION_RE.test(body.connection)) return null;
    if (typeof body.model !== "string" || !body.model || body.model.startsWith("jaxflow-builder-") || /[\r\n\0]/.test(body.model)) {
      return null;
    }
    return { kind, connection: body.connection, model: body.model };
  }
  if (kind === "remove-connection") {
    if (typeof body.connection !== "string" || !CONNECTION_RE.test(body.connection)) return null;
    return { kind, connection: body.connection };
  }
  if (kind === "bind-credential") {
    if (typeof body.connection !== "string" || !CONNECTION_RE.test(body.connection)) return null;
    if (!validCredentialBinding(body.credential)) return null;
    return { kind, connection: body.connection, credential: body.credential };
  }
  return null;
}

function previewKeys(kind: string): string[] {
  if (kind === "save-connection" || kind === "remove-connection") return ["action", "expected", "kind", "connection"];
  if (kind === "save-model" || kind === "remove-model") return ["action", "expected", "kind", "connection", "model"];
  if (kind === "bind-credential") return ["action", "expected", "kind", "connection", "credential"];
  return [];
}

function mutateKeys(kind: string): string[] {
  if (kind === "save-connection" || kind === "remove-connection") return ["action", "expected", "connection"];
  if (kind === "save-model" || kind === "remove-model") return ["action", "expected", "connection", "model"];
  if (kind === "bind-credential") return ["action", "expected", "connection", "credential"];
  return [];
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
  deps: ProvidersRouteDeps,
  action: (typeof KINDS)[number],
  op: string,
  intent: unknown,
  expected: EditorRevision,
) {
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: `opencode-providers-${action}`, action, operation_id: op },
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

async function handleGet(_req: Request, deps: ProvidersRouteDeps = systemDeps) {
  try {
    const snap = await deps.snapshot();
    const connections = deps.sanitize(isObj(snap) ? (snap.connections ?? snap) : snap) as ConnectionChoice[];
    const pending = deps.pending ? await deps.pending() : [];
    return reply({
      ok: true,
      data: {
        editor_revision: isObj(snap) ? snap.editor_revision : undefined,
        connections: applyPendingOperations(Array.isArray(connections) ? connections : [], pending),
      },
    });
  } catch (e) {
    return reply({ ok: false, error: helperCode(e) ?? "unavailable" });
  }
}

async function handlePost(req: Request, deps: ProvidersRouteDeps = systemDeps) {
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
    const kind = body.kind;
    if (typeof kind !== "string" || !keysOk(body, previewKeys(kind))) {
      return reply({ ok: false, error: "invalid payload" });
    }
    const expected = parseExpected(body.expected);
    const intent = parseIntent(kind, body);
    if (!expected || !intent) return reply({ ok: false, error: "invalid payload" });
    try {
      const data = await deps.preview(intent, expected);
      return reply({ ok: true, data });
    } catch (e) {
      return reply({ ok: false, error: helperCode(e) ?? "unavailable" });
    }
  }
  if (typeof action === "string" && (KINDS as readonly string[]).includes(action)) {
    const kind = action as (typeof KINDS)[number];
    if (!keysOk(body, mutateKeys(kind), ["operation_id"])) return reply({ ok: false, error: "invalid payload" });
    const expected = parseExpected(body.expected);
    const intent = parseIntent(kind, body);
    const op = operationId(body);
    if (!expected || !intent || !op) return reply({ ok: false, error: "invalid payload" });
    return mutate(deps, kind, op, intent, expected);
  }
  return reply({ ok: false, error: "invalid payload" });
}

// MOA-498 D4/12: a pure env-name status check for the credential dialog — same class as
// Part 1's envFileStatus(), non-secret, never a value, no new filesystem mechanism
// (loadJaxosEnv already exists and is reused unmodified). Never touches deps/jaxDeps.
async function credentialEnvStatus(name: string) {
  if (!isValidCredentialEnvName(name)) return reply({ ok: false, error: "invalid-env-name" });
  const probe: Record<string, string | undefined> = {};
  loadJaxosEnv([name], probe);
  return reply({ ok: true, data: { state: probe[name] ? "configured" : "missing" } });
}

export function GET(req: Request) {
  const envName = new URL(req.url).searchParams.get("credential-env");
  if (envName !== null) return credentialEnvStatus(envName);
  return handleGet(req, (req as Request & { jaxDeps?: ProvidersRouteDeps }).jaxDeps);
}

export function POST(req: Request) {
  return handlePost(req, (req as Request & { jaxDeps?: ProvidersRouteDeps }).jaxDeps);
}
