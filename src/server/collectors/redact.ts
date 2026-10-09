// Mirror of scripts/jaxflow_hook.py's redact() (patterns at :25-37, application order at :39-48):
// same nine shapes, same replacement tokens, same order (the multi-line PEM block first). Pure —
// no I/O — so the Phase 2 spawn helper can run it over argv and child output before either
// reaches the mutations payload or a JSON response, and Phase 3 can reuse it for child.log tails.
// Python `(?i)` → JS `i`; Python's ASCII `\b` and JS's agree for these patterns.
const PRIVATE_KEY_RE = /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g;
const SECRET_RE = /\b((?:api[_-]?key|token|password|secret|[a-z][a-z0-9_-]*(?:api[_-]?key|token|password|secret)[a-z0-9_-]*))\s*([:=])\s*([^\s,;]+)/gi;
const BEARER_RE = /\b(authorization\s*:\s*bearer\s+)([^\s,;]+)/gi;
const COOKIE_RE = /\b((?:cookie\s*:\s*)?(?:session|sessionid|sid)\s*=\s*)([^;\s,]+)/gi;
const RAW_KEY_RE = /\b(?:AIza[A-Za-z0-9_-]{35}|sk-[A-Za-z0-9_-]{20,})\b/g;
const JWT_RE = /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g;
const CRED_URL_RE = /\b[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^\s/@:]+:[^\s/@]+@/g;
const PROVIDER_TOKEN_RE = /\b(?:ghp_|gho_|ghu_|ghs_|github_pat_|xox[baprs]-|AKIA[0-9A-Z]{16}|npm_)[A-Za-z0-9_-]{10,}\b/g;
const GENERIC_KEY_ASSIGN_RE = /\b([a-z][a-z0-9_-]*[_-](?:key|secret|credential))\s*([:=])\s*([^\s,;]{8,})/gi;

export function redactSecrets(text: string): string {
  let t = String(text).replace(PRIVATE_KEY_RE, "[REDACTED]"); // multi-line first, before line-oriented patterns
  t = t.replace(SECRET_RE, "$1$2[REDACTED]");
  t = t.replace(BEARER_RE, "$1[REDACTED]");
  t = t.replace(COOKIE_RE, "$1[REDACTED]");
  t = t.replace(RAW_KEY_RE, "[REDACTED]");
  t = t.replace(JWT_RE, "[REDACTED]");
  t = t.replace(CRED_URL_RE, "[REDACTED@]");
  t = t.replace(PROVIDER_TOKEN_RE, "[REDACTED]");
  return t.replace(GENERIC_KEY_ASSIGN_RE, "$1$2[REDACTED]");
}
