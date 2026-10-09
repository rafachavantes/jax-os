import { describe, expect, it } from "vitest";
import { agentsBannerKind } from "./settingsBanner";

const ok = (claude: boolean, codex: boolean, opencode: boolean) =>
  ({ ok: true, data: { integrations: { agents: { claude, codex, opencode } } } }) as never;

describe("agentsBannerKind (MOA-504 D7)", () => {
  it.each([
    ["all off", ok(false, false, false), "none"],
    ["opencode only", ok(false, false, true), "no-reviewer"],
    ["claude only", ok(true, false, false), null],
    ["codex only", ok(false, true, false), null],
    ["claude + opencode", ok(true, false, true), null],
    ["all on", ok(true, true, true), null],
  ])("%s -> %s", (_n, result, kind) => {
    expect(agentsBannerKind(result)).toBe(kind);
  });
  it("is null (no banner) while loading, on a settings error, and on a payload without agents", () => {
    expect(agentsBannerKind(undefined)).toBeNull();
    expect(agentsBannerKind({ ok: false, error: "settings-malformed" } as never)).toBeNull();
    expect(agentsBannerKind({ ok: true, data: {} } as never)).toBeNull();
  });
});
