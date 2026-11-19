import { describe, expect, it } from "vitest";
import { handleTargetGet } from "./handler";
import { SessionAccessError, type ResolvedSessionAccess } from "../../../../lib/sessionAccess";

const resolved: ResolvedSessionAccess = {
  selector: { kind: "card", project: "jax-os", harnessSession: "sess-a", pane: "%7", incarnation: "1234:1700000000" },
  expected: { target: "$2:@3.%7", incarnation: "1234:1700000000" },
  sessionName: "project agent",
  runtime: "claude",
  observedAt: "2026-11-19T10:00:00.000Z",
  roUrl: "http://127.0.0.1:10000/ro/?arg=%242%3A%403.%257&disableReconnect=true",
};

function deps(result: ResolvedSessionAccess | Error = resolved) {
  const calls: { selector: unknown; expected: unknown }[] = [];
  return {
    calls,
    resolve: async (selector: never, expected: never) => {
      calls.push({ selector, expected });
      if (result instanceof Error) throw result;
      return result;
    },
  } as never as { resolve: typeof import("../../../../server/session-access").resolveSessionAccess; calls: typeof calls };
}

const url = (query: string) => `http://127.0.0.1/api/tmux/target?${query}`;
const json = async (res: Response) => (await res.json()) as Record<string, unknown>;

describe("GET /api/tmux/target", () => {
  it("resolves an exact card query and returns the envelope without an rwUrl", async () => {
    const d = deps();
    const body = await json(await handleTargetGet(url("kind=card&project=jax-os&harnessSession=sess-a&pane=%257&incarnation=1234%3A1700000000"), d as never));
    expect(body).toMatchObject({ ok: true, data: { expected: resolved.expected, runtime: "claude" } });
    expect(JSON.stringify(body)).not.toContain("rwUrl");
    expect(d.calls[0].selector).toEqual(resolved.selector);
    expect(d.calls[0].expected).toBeUndefined();
  });

  it("maps an omitted card harness to the explicit null seat", async () => {
    const d = deps();
    await handleTargetGet(url("kind=card&project=jax-os&pane=%257&incarnation=1234%3A1700000000"), d as never);
    expect((d.calls[0].selector as { harnessSession: string | null }).harnessSession).toBeNull();
  });

  it("passes a fully valid expected binding and refuses a malformed one before resolving", async () => {
    const d = deps();
    await handleTargetGet(url("kind=card&project=p&pane=%257&incarnation=1%3A1&expectedTarget=%242%3A%403.%257&expectedIncarnation=1%3A1"), d as never);
    expect(d.calls[0].expected).toEqual({ target: "$2:@3.%7", incarnation: "1:1" });

    const d2 = deps();
    const body = await json(await handleTargetGet(url("kind=card&project=p&pane=%257&incarnation=1%3A1&expectedTarget=attach"), d2 as never));
    expect(body).toMatchObject({ ok: false, code: "invalid-selector" });
    expect(d2.calls).toHaveLength(0);
  });

  it("refuses duplicate, unknown and conflicting forms before any native/DB access", async () => {
    for (const query of [
      "kind=tmux&session=a&session=b",
      "kind=tmux&session=a&from=home",
      "kind=tmux&session=a&pane=%257",
      "kind=card&project=p&pane=%257&incarnation=1%3A1&session=x",
      "kind=card&project=p",
      "kind=other&session=a",
    ]) {
      const d = deps();
      const body = await json(await handleTargetGet(url(query), d as never));
      expect(body).toMatchObject({ ok: false, code: "invalid-selector" });
      expect(d.calls).toHaveLength(0);
    }
  });

  it("maps resolver failures to bounded codes, never raw exception text", async () => {
    const lost = await json(await handleTargetGet(url("kind=tmux&session=a"), deps(new SessionAccessError("lost-target")) as never));
    expect(lost).toEqual({ ok: false, code: "lost-target", error: "session target lost" });

    const boom = await json(await handleTargetGet(url("kind=tmux&session=a"), deps(new Error("/home/rafa/secret token=abc")) as never));
    expect(boom).toEqual({ ok: false, code: "source-failure", error: "session source unavailable" });
    expect(JSON.stringify(boom)).not.toContain("secret");
  });
});
