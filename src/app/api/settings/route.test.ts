import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const DEFAULTS = {
  ownerName: "", locale: "en-US", reposRoot: "/home/x/repos", vaultPath: null,
  monitoredUnits: { user: [], system: [] },
  integrations: {
    linear: false, ttyd: false, webhook: false, hermesTokens: false,
    agents: { claude: false, codex: false, opencode: false },
    classifier: false, github: false, vault: false,
  },
};

// Pass-through-valid default: fills any missing top-level key from DEFAULTS (mirrors 497A's own
// "a missing field falls back to its default independently" rule), invalid only when `locale`
// isn't one of the two schema locales — that's this harness's one "malformed shape" case (F2).
// Mirrors the real `parseSettings(raw)` shape: SettingsResult, not GeneralSettings | null.
function fakeParse(value: unknown): { ok: true; data: typeof DEFAULTS } | { ok: false; error: "settings-malformed" } {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ok: false, error: "settings-malformed" };
  const merged = { ...DEFAULTS, ...(value as Record<string, unknown>) };
  if (merged.locale !== "en-US" && merged.locale !== "pt-BR") return { ok: false, error: "settings-malformed" };
  return { ok: true, data: merged as typeof DEFAULTS };
}

const harness = vi.hoisted(() => ({
  readGeneralSettings: vi.fn(),
  writeGeneralSettings: vi.fn(),
  parseSettings: vi.fn(),
  getDb: vi.fn(() => ({ tag: "fake-db" })),
  nativeCredentials: { claude: false, codex: false, opencodeGo: false },
}));
vi.mock("../../../server/settings", () => ({
  readGeneralSettings: harness.readGeneralSettings,
  writeGeneralSettings: harness.writeGeneralSettings,
  parseSettings: harness.parseSettings,
}));
vi.mock("../../../server/db", () => ({ getDb: harness.getDb }));
// Native credential detection lives in the subscriptions collector — a separate module, so it
// gets its own mock block (F3, diff review e52d9e6dc555).
vi.mock("../../../server/collectors/subscriptions", () => ({
  nativeCredentialsConfigured: () => ({ ...harness.nativeCredentials }),
}));

import { GET, PUT } from "./route";

function putRequest(body: unknown, headers: Record<string, string> = {}) {
  return new Request("http://127.0.0.1/api/settings", {
    method: "PUT", body: JSON.stringify(body),
    headers: { "content-type": "application/json", ...headers },
  });
}

const WEBHOOK_ENV_KEYS = ["NOTIFICATION_WEBHOOK_URL", "NOTIFICATION_WEBHOOK_SECRET"] as const;
let savedWebhookEnv: Record<string, string | undefined>;
// MOA-498 F2: the route now calls the real envFileStatus(), which resolves $JAXOS_HOME/.env.
// Point JAXOS_HOME at a fresh empty temp dir per test so no test can ever read the owner's
// real ~/.jax-os/.env (Global Constraints: tests never touch the real ~/.jax-os).
let envHomeDir: string;
let savedJaxosHome: string | undefined;

beforeEach(() => {
  harness.readGeneralSettings.mockReset();
  harness.writeGeneralSettings.mockReset();
  harness.parseSettings.mockReset();
  harness.getDb.mockClear();
  harness.parseSettings.mockImplementation(fakeParse);
  harness.nativeCredentials = { claude: false, codex: false, opencodeGo: false };
  savedWebhookEnv = Object.fromEntries(WEBHOOK_ENV_KEYS.map((k) => [k, process.env[k]]));
  for (const k of WEBHOOK_ENV_KEYS) delete process.env[k];
  savedJaxosHome = process.env.JAXOS_HOME;
  envHomeDir = mkdtempSync(join(tmpdir(), "settings-route-jaxos-"));
  process.env.JAXOS_HOME = envHomeDir;
});
afterEach(() => {
  for (const k of WEBHOOK_ENV_KEYS) {
    if (savedWebhookEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedWebhookEnv[k];
  }
  if (savedJaxosHome === undefined) delete process.env.JAXOS_HOME;
  else process.env.JAXOS_HOME = savedJaxosHome;
  rmSync(envHomeDir, { recursive: true, force: true });
});

describe("GET /api/settings", () => {
  it("returns the reader's result verbatim on success, plus the webhook configured/missing status", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    const res = await GET();
    expect(await res.json()).toEqual({
      ok: true, data: DEFAULTS, webhookConfigured: false,
      credentialsConfigured: { claude: false, codex: false, opencodeGo: false },
      envFileState: "missing",
    });
  });

  it("GET carries credentialsConfigured alongside webhookConfigured", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    harness.nativeCredentials = { claude: true, codex: false, opencodeGo: false };
    const json = await (await GET()).json();
    expect(json.credentialsConfigured).toEqual({ claude: true, codex: false, opencodeGo: false });
  });

  it("webhookConfigured is true only when BOTH NOTIFICATION_WEBHOOK_URL and NOTIFICATION_WEBHOOK_SECRET are set", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    process.env.NOTIFICATION_WEBHOOK_URL = "https://example.com/hook";
    expect((await (await GET()).json()).webhookConfigured).toBe(false); // secret still missing
    process.env.NOTIFICATION_WEBHOOK_SECRET = "s3cr3t";
    expect((await (await GET()).json()).webhookConfigured).toBe(true);
  });

  it.each(["settings-malformed", "settings-unreadable"])(
    "returns the reader's tagged error verbatim (%s), with no webhookConfigured field",
    async (error) => {
      harness.readGeneralSettings.mockReturnValue({ ok: false, error });
      const res = await GET();
      expect(await res.json()).toEqual({ ok: false, error });
    },
  );

  it("envFileState reflects $JAXOS_HOME/.env (MOA-498 F2)", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    const dir = mkdtempSync(join(tmpdir(), "settings-route-envfile-"));
    process.env.JAXOS_HOME = dir;
    try {
      expect((await (await GET()).json()).envFileState).toBe("missing");
      writeFileSync(join(dir, ".env"), "X=1\n", { mode: 0o600 });
      expect((await (await GET()).json()).envFileState).toBe("ok");
    } finally {
      delete process.env.JAXOS_HOME;
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("PUT /api/settings", () => {
  it("rejects a cross-site request before reading settings", async () => {
    const res = await PUT(putRequest({ ownerName: "x" }, { "sec-fetch-site": "cross-site" }));
    expect((await res.json()).ok).toBe(false);
    expect(harness.readGeneralSettings).not.toHaveBeenCalled();
  });

  it("a partial body ({ownerName} only) leaves integrations/monitoredUnits/etc untouched (decision 15 merge)", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    const res = await PUT(putRequest({ ownerName: "Owner" }));
    const json = await res.json();
    expect(json.ok).toBe(true);
    expect(json.data.ownerName).toBe("Owner");
    expect(json.data.integrations).toEqual(DEFAULTS.integrations);
    expect(json.data.monitoredUnits).toEqual(DEFAULTS.monitoredUnits);
    expect(harness.writeGeneralSettings).toHaveBeenCalledWith(harness.getDb(), json.data);
  });

  it("replaces a nested object WHOLE, never deep-merges (integrations.linear on, the other six untouched keys of a nested body still apply)", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    const res = await PUT(putRequest({ integrations: { ...DEFAULTS.integrations, linear: true } }));
    const json = await res.json();
    expect(json.data.integrations.linear).toBe(true);
  });

  it("a malformed MERGED shape refuses without writing", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    const res = await PUT(putRequest({ locale: "fr-FR" }));
    expect(await res.json()).toEqual({ ok: false, error: "settings-malformed" });
    expect(harness.writeGeneralSettings).not.toHaveBeenCalled();
  });

  it("an unreadable CURRENT file refuses the write outright, never merges onto an unknown state", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: false, error: "settings-unreadable" });
    const res = await PUT(putRequest({ ownerName: "Owner" }));
    expect(await res.json()).toEqual({ ok: false, error: "settings-unreadable" });
    expect(harness.writeGeneralSettings).not.toHaveBeenCalled();
  });

  it("a MALFORMED current file merges onto the schema defaults instead of refusing — the Geral form is how the owner repairs a malformed file (resolved ambiguity, see Self-review)", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: false, error: "settings-malformed" });
    const res = await PUT(putRequest({ ownerName: "Owner" }));
    const json = await res.json();
    expect(json.ok).toBe(true);
    expect(json.data.ownerName).toBe("Owner");
    expect(json.data.integrations.linear).toBe(false); // schema default, not carried over from nothing
  });

  it("round-trips a full-object PUT", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    const full = { ...DEFAULTS, ownerName: "Owner", locale: "pt-BR" as const };
    const res = await PUT(putRequest(full));
    expect((await res.json()).data).toEqual(full);
  });

  it("a successful save also reports webhookConfigured", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    process.env.NOTIFICATION_WEBHOOK_URL = "https://example.com/hook";
    process.env.NOTIFICATION_WEBHOOK_SECRET = "s3cr3t";
    const res = await PUT(putRequest({ ownerName: "Owner" }));
    expect((await res.json()).webhookConfigured).toBe(true);
  });

  it("a successful save also reports credentialsConfigured (mirrors webhookConfigured)", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    harness.nativeCredentials = { claude: false, codex: true, opencodeGo: false };
    const res = await PUT(putRequest({ ownerName: "Owner" }));
    expect((await res.json()).credentialsConfigured).toEqual({ claude: false, codex: true, opencodeGo: false });
  });

  it("a writer failure (audit store unavailable) becomes a {ok:false} response, never a 5xx", async () => {
    harness.readGeneralSettings.mockReturnValue({ ok: true, data: DEFAULTS });
    harness.writeGeneralSettings.mockImplementation(() => {
      throw new Error("settings-audit-unavailable");
    });
    const res = await PUT(putRequest({ ownerName: "Owner" }));
    expect(await res.json()).toEqual({ ok: false, error: "settings-audit-unavailable" });
  });
});
