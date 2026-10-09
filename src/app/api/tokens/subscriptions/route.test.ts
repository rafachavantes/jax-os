import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ ok: true, agents: { claude: true, codex: true, opencode: true } }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => settingsState.ok ? { ok: true, data: { integrations: { agents: settingsState.agents } } } : { ok: false, error: "settings-malformed" },
}));

vi.mock("../../../../server/collectors/subscriptions", () => ({ getSubscriptions: vi.fn() }));

import { getSubscriptions } from "../../../../server/collectors/subscriptions";
import { GET } from "./route";

const EMPTY = { anthropic: { ok: false as const, error: "x" }, codex: { ok: false as const, error: "x" }, opencode: { ok: false as const, error: "x" } };

describe("GET /api/tokens/subscriptions", () => {
  beforeEach(() => {
    vi.mocked(getSubscriptions).mockReset().mockResolvedValue(EMPTY);
    settingsState.ok = true;
    settingsState.agents = { claude: true, codex: true, opencode: true };
  });

  it("reads per-provider flags from settings and threads them into getSubscriptions", async () => {
    settingsState.agents = { claude: false, codex: true, opencode: true };
    const res = await GET();
    expect(await res.json()).toEqual({ ok: true, data: EMPTY });
    expect(vi.mocked(getSubscriptions)).toHaveBeenCalledWith({ claude: false, codex: true, opencodeGo: true });
  });

  it("passes all three flags through unchanged when enabled (regression)", async () => {
    await GET();
    expect(vi.mocked(getSubscriptions)).toHaveBeenCalledWith({ claude: true, codex: true, opencodeGo: true });
  });

  it("an unreadable settings file fails closed: every provider off", async () => {
    settingsState.ok = false;
    await GET();
    expect(vi.mocked(getSubscriptions)).toHaveBeenCalledWith({ claude: false, codex: false, opencodeGo: false });
  });

  it("maps agents.opencode onto the collector's opencodeGo flag", async () => {
    settingsState.agents = { claude: false, codex: false, opencode: true };
    await GET();
    expect(vi.mocked(getSubscriptions)).toHaveBeenCalledWith({ claude: false, codex: false, opencodeGo: true });
  });
});
