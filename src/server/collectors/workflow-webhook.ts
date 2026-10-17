import { createHmac } from "node:crypto";
import { readJaxosEnvValue } from "../envFile";
import { readGeneralSettings } from "../settings";

// The ONE place this build touches the network: a single bounded, HMAC-signed POST to the
// owner's own notification webhook, called from the workflow events forward path
// (src/app/api/workflow/events/route.ts).
// X-Hub-Signature-256 = sha256=<hex HMAC_SHA256(secret, raw body bytes)>; the caller
// serializes ONCE and we send exactly those bytes. Never throws — a failed forward leaves the
// event `pending` for the poll.

export type NotificationTarget = { url: string; secret: string };
export type ForwardResult = { ok: true; status: number } | { ok: false; error: string };

export function notificationTargetFromEnv(
  env: Record<string, string | undefined> = process.env,
  settings: ReturnType<typeof readGeneralSettings> = readGeneralSettings(),
): NotificationTarget | null {
  // Checked BEFORE the env read, not after (D3): a disabled integration never touches the
  // file, matching every other integration gate this ownership matrix already establishes.
  if (!settings.ok || !settings.data.integrations.webhook) return null;
  // D1 precedence: a genuine process-env/.env.local value (already merged into `env` before
  // this runs) wins; otherwise read $JAXOS_HOME/.env fresh so an owner edit takes effect on
  // the next call, no restart needed.
  const url = env.NOTIFICATION_WEBHOOK_URL ?? readJaxosEnvValue("NOTIFICATION_WEBHOOK_URL");
  const secret = env.NOTIFICATION_WEBHOOK_SECRET ?? readJaxosEnvValue("NOTIFICATION_WEBHOOK_SECRET");
  return url && secret ? { url, secret } : null;
}

export function signBody(body: string, secret: string): string {
  return "sha256=" + createHmac("sha256", secret).update(body).digest("hex");
}

export async function forwardNotification(
  body: string,
  target: NotificationTarget,
  opts: { timeoutMs?: number; fetchFn?: typeof fetch } = {},
): Promise<ForwardResult> {
  const { timeoutMs = 3000, fetchFn = fetch } = opts;
  try {
    const res = await fetchFn(target.url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Hub-Signature-256": signBody(body, target.secret) },
      body,
      signal: AbortSignal.timeout(timeoutMs),
    });
    // MOA-498 D3: any 2xx confirms delivery — a generic receiver may legitimately answer
    // 201/202/204, widened from the old Hermes-only exact-200 check.
    return res.status >= 200 && res.status < 300
      ? { ok: true, status: res.status }
      : { ok: false, error: `gateway responded ${res.status}` };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}
