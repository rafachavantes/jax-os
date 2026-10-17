import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";
import { loadJaxosEnv } from "./src/server/envFile";

// MOA-498 D1/D2: every jax-os-managed secret comes from $JAXOS_HOME/.env (owner-filled,
// 0600), never ~/.hermes/.env. Precedence: process env (systemd/shell) > .env.local (Next
// loads that before this file runs) > $JAXOS_HOME/.env. HERMES_WEBHOOK_SECRET (approvals,
// never a jax-os workflow secret) is deliberately NOT in this list.
// NOTIFICATION_WEBHOOK_URL/SECRET are NOT preloaded here: workflow-webhook.ts reads them
// fresh from the file per call (readJaxosEnvValue) so an owner edit takes effect without a
// restart, while still honoring this same precedence.
loadJaxosEnv(["LINEAR_API_KEY", "OPENCODE_GO_API_KEY", "TYPESAFE_API"]);

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  // stray root-owned /home/rafa/package-lock.json makes Next infer the wrong
  // workspace root — pin it here instead of "fixing" someone else's file
  outputFileTracingRoot: __dirname,
};

export default withNextIntl(nextConfig);
