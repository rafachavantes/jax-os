import { NextResponse } from "next/server";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { APP_KEYS, compose, type AppKey, type RuleDocs } from "../../../../lib/rules";
import { applyRuleFile, rulePaths, sourceRevision, type ApplyOutcome } from "../../../../server/collectors/rules";
import { getDb } from "../../../../server/db";
import { getRuleDocs } from "../../../../server/db/rules";
import { runMutation } from "../../../../server/mutations";
import { readGeneralSettings, type SettingsResult } from "../../../../server/settings";
import { MutationRejected } from "../../../../lib/mutationOutcome";

const HEX64 = /^[0-9a-f]{64}$/;

type ApplyRevisions = { global: string; exception: string };

// Test seam (same pattern as agent-settings / the rules GET route): production
// uses these defaults, tests set `jaxDeps` on the Request so they can point at
// a temporary root instead of $HOME.
export type RulesApplyDeps = {
  getDocs: (db: ReturnType<typeof getDb>) => RuleDocs;
  apply: (targetPath: string, expected: string, clientHash: string) => ApplyOutcome;
  path: (app: AppKey) => string;
  readSettings: () => SettingsResult;
};

const systemDeps: RulesApplyDeps = {
  getDocs: (db) => getRuleDocs(db),
  apply: applyRuleFile,
  path: (app) => rulePaths()[app],
  readSettings: readGeneralSettings,
};

// Exact two-key pair (Task 2 §5): the four-key GET `sourceRevisions` object is
// NOT accepted here — the submitter must narrow it to {global, exception}.
function parseRevisions(value: unknown): ApplyRevisions | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const keys = Object.keys(value as Record<string, unknown>);
  if (keys.length !== 2 || !keys.includes("global") || !keys.includes("exception")) return null;
  const global = (value as Record<string, unknown>).global;
  const exception = (value as Record<string, unknown>).exception;
  if (typeof global !== "string" || !HEX64.test(global)) return null;
  if (typeof exception !== "string" || !HEX64.test(exception)) return null;
  return { global, exception };
}

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;

  const body = await readJsonCapped(req, 4096);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error }, { status: 400 });
  const { app, diskHash, applySourceRevisions } = (body.value ?? {}) as Record<string, unknown>;
  if (typeof app !== "string" || !(APP_KEYS as readonly string[]).includes(app) || typeof diskHash !== "string") {
    return NextResponse.json({ ok: false, error: "invalid payload" }, { status: 400 });
  }
  const revisions = parseRevisions(applySourceRevisions);
  if (!revisions) {
    return NextResponse.json({ ok: false, error: "invalid applySourceRevisions" }, { status: 400 });
  }

  const deps = (req as Request & { jaxDeps?: RulesApplyDeps }).jaxDeps ?? systemDeps;
  const appKey = app as AppKey;
  // MOA-504 D10: fail CLOSED on unreadable settings, refuse a disabled agent; both before any mutation.
  const settings = deps.readSettings();
  if (!settings.ok) return NextResponse.json({ ok: false, error: settings.error }, { status: 409 });
  if (!settings.data.integrations.agents[appKey]) {
    return NextResponse.json({ ok: false, error: "agent-disabled" }, { status: 409 });
  }
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "rules-apply", app: appKey, path: deps.path(appKey) },
    () => {
      const db = getDb();
      // Source revision check + synchronous file apply in ONE immediate DB
      // transaction: another source writer cannot slip between them.
      const outcome = db
        .transaction(() => {
          const docs = deps.getDocs(db);
          if (sourceRevision(docs.global) !== revisions.global) {
            throw new MutationRejected("canonical source changed since preview", 409);
          }
          if (sourceRevision(docs[appKey]) !== revisions.exception) {
            throw new MutationRejected("exception source changed since preview", 409);
          }
          const expected = compose(docs[appKey], docs.global);
          return deps.apply(deps.path(appKey), expected, diskHash);
        })
        .immediate();
      if (!outcome.ok) {
        if (outcome.reason === "stale") throw new MutationRejected("file changed on disk since diskHash was read", 409);
        if (outcome.reason === "symlink") throw new MutationRejected("refused: symlink in target path", 400);
        throw new Error("rules apply failed");
      }
      return outcome;
    },
    (value) => ({ bytes: value.bytes, backup: value.backup }),
  );
  if (result.ok) {
    return NextResponse.json({ ok: true, data: { bytes: result.value.bytes, backup: result.value.backup } });
  }
  const { value, status, ...failure } = result;
  return NextResponse.json(
    { ...failure, ...(value && value.ok ? { data: { bytes: value.bytes, backup: value.backup } } : {}) },
    { status: status ?? 500 },
  );
}
