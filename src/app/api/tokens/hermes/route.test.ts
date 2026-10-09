import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ hermesTokens: true }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({ ok: true, data: { integrations: { hermesTokens: settingsState.hermesTokens } } }),
}));

vi.mock("../../../../server/collectors/hermes-meta", () => ({ getHermesMeta: vi.fn() }));

import { getHermesMeta } from "../../../../server/collectors/hermes-meta";
import { GET } from "./route";

describe("GET /api/tokens/hermes", () => {
  beforeEach(() => {
    vi.mocked(getHermesMeta).mockReset().mockResolvedValue({} as never);
    settingsState.hermesTokens = true;
  });

  it("refuses when hermesTokens is disabled, never calling getHermesMeta", async () => {
    settingsState.hermesTokens = false;
    const json = await (await GET()).json();
    expect(json).toEqual({ ok: false, error: "disabled" });
    expect(vi.mocked(getHermesMeta)).not.toHaveBeenCalled();
  });

  it("serves the collector envelope when hermesTokens is on (regression)", async () => {
    const json = await (await GET()).json();
    expect(json).toEqual({ ok: true, data: {} });
    expect(vi.mocked(getHermesMeta)).toHaveBeenCalledOnce();
  });
});
