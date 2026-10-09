import { describe, expect, it } from "vitest";
import { redactSecrets } from "./redact";

const GHP = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789";
const SK = "sk-" + "abcdefghijklmnopqrstuvwxyz1234";
const JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop";

describe("redactSecrets — mirror of jaxflow_hook.py redact()", () => {
  it("strips every known shape, keeping the key/prefix that identifies it", () => {
    expect(redactSecrets(`API_KEY=abcd1234 && pnpm test`)).toBe("API_KEY=[REDACTED] && pnpm test");
    expect(redactSecrets(`OPENAI_API_KEY: ${SK}`)).toBe("OPENAI_API_KEY:[REDACTED]");
    expect(redactSecrets(`Authorization: Bearer abc.def`)).toBe("Authorization: Bearer [REDACTED]");
    expect(redactSecrets(`Cookie: sessionid=abc123; x=1`)).toBe("Cookie: sessionid=[REDACTED]; x=1");
    expect(redactSecrets(`token ${SK} here`)).toBe("token [REDACTED] here");
    expect(redactSecrets(`jwt ${JWT} end`)).toBe("jwt [REDACTED] end");
    expect(redactSecrets(`git clone https://user:pw@github.com/x/y.git`)).toBe("git clone [REDACTED@]github.com/x/y.git");
    expect(redactSecrets(`push ${GHP} now`)).toBe("push [REDACTED] now");
    expect(redactSecrets(`my_secret_credential=abcdefgh12`)).toBe("my_secret_credential=[REDACTED]");
    expect(redactSecrets("-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----\n")).toBe("[REDACTED]\n");
  });

  it("is a no-op on text with no known shape — an unknown secret is stored as typed (spec §4)", () => {
    expect(redactSecrets("pnpm test && hunter2-is-my-password-word")).toBe("pnpm test && hunter2-is-my-password-word");
    expect(redactSecrets("pnpm build")).toBe("pnpm build");
    expect(redactSecrets("")).toBe("");
  });

  it("redacts every occurrence, not just the first", () => {
    expect(redactSecrets(`${GHP} and ${GHP}`)).toBe("[REDACTED] and [REDACTED]");
  });
});
