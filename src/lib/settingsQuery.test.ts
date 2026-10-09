import { describe, expect, it, vi } from "vitest";
import { agentsOf, postGeneralSettings, vaultUsable } from "./settingsQuery";

describe("postGeneralSettings", () => {
  it("PUTs the given patch to /api/settings and resolves the JSON envelope", async () => {
    const fetchMock = vi.fn(async () => ({ json: async () => ({ ok: true, data: { ownerName: "Owner" } }) }));
    const result = await postGeneralSettings({ ownerName: "Owner" }, fetchMock as unknown as typeof fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ownerName: "Owner" }),
    });
    expect(result).toEqual({ ok: true, data: { ownerName: "Owner" } });
  });

  it("resolves {ok:false} on a rejected fetch instead of throwing", async () => {
    const fetchMock = vi.fn(async () => { throw new Error("network down"); });
    const result = await postGeneralSettings({ ownerName: "Owner" }, fetchMock as unknown as typeof fetch);
    expect(result.ok).toBe(false);
  });
});

describe("vaultUsable (diff review e52d9e6dc555 F2)", () => {
  it("false when the flag is off", () => {
    expect(vaultUsable({ ok: true, data: { vaultPath: "/x", integrations: { vault: false } } } as never)).toBe(false);
  });
  it("false when the flag is on but vaultPath is null", () => {
    expect(vaultUsable({ ok: true, data: { vaultPath: null, integrations: { vault: true } } } as never)).toBe(false);
  });
  it("true when the flag is on and vaultPath is set", () => {
    expect(vaultUsable({ ok: true, data: { vaultPath: "/x", integrations: { vault: true } } } as never)).toBe(true);
  });
  it("false when settings hasn't loaded or failed", () => {
    expect(vaultUsable(undefined)).toBe(false);
    expect(vaultUsable({ ok: false, error: "x" })).toBe(false);
  });
});

describe("agentsOf (MOA-504 D10: fail OPEN)", () => {
  it("returns the saved agents when the result is ok", () => {
    const r = { ok: true, data: { integrations: { agents: { claude: false, codex: true, opencode: false } } } } as never;
    expect(agentsOf(r)).toEqual({ claude: false, codex: true, opencode: false });
  });
  it("is all on while loading, when not ok, or for a payload with no integrations (never hide on unknown)", () => {
    const on = { claude: true, codex: true, opencode: true };
    expect(agentsOf(undefined)).toEqual(on);
    expect(agentsOf({ ok: false, error: "settings-malformed" })).toEqual(on);
    expect(agentsOf({ ok: true, data: { editor_revision: "x" } } as never)).toEqual(on);
  });
});
