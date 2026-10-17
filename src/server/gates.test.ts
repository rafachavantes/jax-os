import { describe, expect, it, vi } from "vitest";
import { readGeneralSettings } from "./settings";
import { linearEnabled, vaultUsableNow } from "./gates";

vi.mock("./settings", () => ({ readGeneralSettings: vi.fn() }));

const mocked = vi.mocked(readGeneralSettings);

const settings = (integrations: Record<string, unknown>, vaultPath: string | null) =>
  ({ ok: true, data: { integrations, vaultPath } }) as unknown as ReturnType<typeof readGeneralSettings>;

describe("vaultUsableNow", () => {
  it("is false when settings are unreadable", () => {
    mocked.mockReturnValue({ ok: false, error: "settings-unreadable" });
    expect(vaultUsableNow()).toBe(false);
  });

  it("is false when the vault integration is off", () => {
    mocked.mockReturnValue(settings({ vault: false }, "/v"));
    expect(vaultUsableNow()).toBe(false);
  });

  it("is false when vaultPath is null", () => {
    mocked.mockReturnValue(settings({ vault: true }, null));
    expect(vaultUsableNow()).toBe(false);
  });

  it("is true when the vault is on and a path is set", () => {
    mocked.mockReturnValue(settings({ vault: true }, "/v"));
    expect(vaultUsableNow()).toBe(true);
  });
});

describe("linearEnabled", () => {
  it("is false when settings are unreadable", () => {
    mocked.mockReturnValue({ ok: false, error: "settings-unreadable" });
    expect(linearEnabled()).toBe(false);
  });

  it("is false when the linear integration is off", () => {
    mocked.mockReturnValue(settings({ linear: false }, null));
    expect(linearEnabled()).toBe(false);
  });

  it("is true when the linear integration is on", () => {
    mocked.mockReturnValue(settings({ linear: true }, null));
    expect(linearEnabled()).toBe(true);
  });
});
