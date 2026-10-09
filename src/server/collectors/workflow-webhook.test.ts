import { createHmac } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { readGeneralSettings } from "../settings";
import { forwardNotification, notificationTargetFromEnv, signBody } from "./workflow-webhook";

const TARGET = { url: "http://127.0.0.1:8644/webhooks/jax-workflow", secret: "s3cret" };

// The gate only ever reads settings.data.integrations.webhook — these fixtures are the minimal
// shapes that exercise it (cast: the real GeneralSettings has many more required fields).
type SettingsLike = ReturnType<typeof readGeneralSettings>;
const settingsWith = (webhook: boolean) => ({ ok: true, data: { integrations: { webhook } } }) as unknown as SettingsLike;

describe("signBody", () => {
  it("is sha256= + hex HMAC-SHA256 over the exact bytes", () => {
    const body = '{"event_id":1,"type":"run-finished"}';
    const expected = "sha256=" + createHmac("sha256", "s3cret").update(body).digest("hex");
    expect(signBody(body, "s3cret")).toBe(expected);
    expect(signBody(body + " ", "s3cret")).not.toBe(expected);
  });
});

describe("notificationTargetFromEnv (MOA-498 D3 — integrations.webhook gate)", () => {
  // Isolates $JAXOS_HOME from the real ~/.jax-os so these tests never touch Rafa's real .env.
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "jax-webhook-"));
    process.env.JAXOS_HOME = dir;
  });
  afterEach(() => {
    delete process.env.JAXOS_HOME;
    rmSync(dir, { recursive: true, force: true });
  });
  function writeEnvFile(text: string) {
    const p = join(dir, ".env");
    writeFileSync(p, text, { mode: 0o600 });
    chmodSync(p, 0o600);
  }

  it("returns null and never reads env when the flag is off — checked before the env read", () => {
    const env = new Proxy({}, { get() { throw new Error("must not read env when disabled"); } });
    expect(notificationTargetFromEnv(env, settingsWith(false))).toBeNull();
  });
  it("returns null when settings are unreadable (fail closed, same as off)", () => {
    const env = new Proxy({}, { get() { throw new Error("must not read env"); } });
    expect(notificationTargetFromEnv(env, { ok: false, error: "settings-unreadable" } as unknown as SettingsLike)).toBeNull();
  });
  it("needs BOTH vars when the flag is on and no .env file exists", () => {
    const on = settingsWith(true);
    expect(notificationTargetFromEnv({}, on)).toBeNull();
    expect(notificationTargetFromEnv({ NOTIFICATION_WEBHOOK_URL: "u" }, on)).toBeNull();
    expect(notificationTargetFromEnv({ NOTIFICATION_WEBHOOK_URL: "u", NOTIFICATION_WEBHOOK_SECRET: "s" }, on))
      .toEqual({ url: "u", secret: "s" });
  });
  it("reads fresh from $JAXOS_HOME/.env on every call — no restart needed after a rewrite", () => {
    const on = settingsWith(true);
    writeEnvFile("NOTIFICATION_WEBHOOK_URL=http://a\nNOTIFICATION_WEBHOOK_SECRET=sA\n");
    expect(notificationTargetFromEnv({}, on)).toEqual({ url: "http://a", secret: "sA" });
    writeEnvFile("NOTIFICATION_WEBHOOK_URL=http://b\nNOTIFICATION_WEBHOOK_SECRET=sB\n");
    expect(notificationTargetFromEnv({}, on)).toEqual({ url: "http://b", secret: "sB" });
  });
  it("prefers a genuine process-env/.env.local value over the file (D1 precedence)", () => {
    const on = settingsWith(true);
    writeEnvFile("NOTIFICATION_WEBHOOK_URL=http://file\nNOTIFICATION_WEBHOOK_SECRET=file-secret\n");
    expect(notificationTargetFromEnv({ NOTIFICATION_WEBHOOK_URL: "http://real", NOTIFICATION_WEBHOOK_SECRET: "real-secret" }, on))
      .toEqual({ url: "http://real", secret: "real-secret" });
  });
});

describe("forwardNotification", () => {
  it("POSTs the exact body with the signature header and reports 2xx as ok", async () => {
    const fetchFn = vi.fn(async () => new Response("ok", { status: 200 }));
    const body = '{"a":1}';
    const r = await forwardNotification(body, TARGET, { fetchFn: fetchFn as unknown as typeof fetch });
    expect(r).toEqual({ ok: true, status: 200 });
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(TARGET.url);
    expect(init.method).toBe("POST");
    expect(init.body).toBe(body);
    expect((init.headers as Record<string, string>)["X-Hub-Signature-256"]).toBe(signBody(body, TARGET.secret));
  });
  it("treats every 2xx as delivered — a generic receiver may answer 201/202/204", async () => {
    for (const status of [200, 201, 202, 204]) {
      const r = await forwardNotification("{}", TARGET, { fetchFn: (async () => new Response(null, { status })) as unknown as typeof fetch });
      expect(r).toEqual({ ok: true, status });
    }
  });
  it("non-2xx and throws are ok:false", async () => {
    const r401 = await forwardNotification("{}", TARGET, { fetchFn: (async () => new Response("", { status: 401 })) as unknown as typeof fetch });
    expect(r401).toEqual({ ok: false, error: "gateway responded 401" });
    const rErr = await forwardNotification("{}", TARGET, { fetchFn: (async () => { throw new Error("ECONNREFUSED"); }) as unknown as typeof fetch });
    expect(rErr).toEqual({ ok: false, error: "ECONNREFUSED" });
  });
});
